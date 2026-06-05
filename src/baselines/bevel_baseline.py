"""
bevel_baseline.py
=================
Baseline-метод "Beveling (distance transform)" — варіант №4 з роботи
Moreira et al. (arxiv 2212.09692) "Comparing methods to generate
normal maps for pixel-art".

Алгоритм
--------
1.  З альфа-каналу спрайта будуємо бінарну маску (alpha > thresh).
2.  Рахуємо distance transform: для кожного пікселя в середині —
    відстань до найближчого фонового пікселя.  Це і є карта висот
    `h(x, y)`: чим глибше всередині силуета, тим вище.
3.  Нормалізуємо її у [0, 1] і опціонально підносимо у степінь
    < 1 (фасочна "купольна" форма) або > 1 (різкі краї).
4.  Беремо градієнти Собелем та конвертуємо у нормаль:
        N = normalize(-gx, -gy, 1)
5.  Прозорим пікселям виставляємо плоску нормаль (0, 0, 1).

Цей простий beveling уже породжує плавний "псевдо-3D" зовнішній вигляд
без необхідності читати тіні з кольорового зображення — на відміну від
Sobel-luminance, що часто ловить тіні замість геометрії.

Реалізація distance transform
-----------------------------
* Якщо встановлено SciPy: `scipy.ndimage.distance_transform_edt`
  (євклідова відстань, найточніше).
* Інакше fallback: ітеративна ерозія 3×3 через PIL.MinFilter, що
  дає Chebyshev-відстань.  Для пікселізованого арту різниця у
  фінальному рендері помітна слабо.

CLI приклад
-----------
    python bevel_baseline.py \
        --color_dir  new_dataset/color  \
        --normal_dir new_dataset/normal \
        --output_dir baseline/bevel_eval \
        --val_split 0.10 --seed 42 \
        --num_examples 10
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

# Імпортуємо допоміжне з sobel_baseline.py щоб гарантувати ідентичний
# спліт + однаковий обчислювач метрик.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sobel_baseline import (   # noqa: E402
    SOBEL_X, SOBEL_Y, _convolve2d_same,
    build_pair_list, stratified_val_indices,
    masked_metrics, save_example_grid,
)

# Опціональний SciPy
try:
    from scipy.ndimage import distance_transform_edt   # type: ignore
    HAS_SCIPY = True
except Exception:                                       # pragma: no cover
    HAS_SCIPY = False


# ──────────────────────────────────────────────
# 1.  DISTANCE TRANSFORM (з fallback)
# ──────────────────────────────────────────────

def _edt_via_pil(mask: np.ndarray) -> np.ndarray:
    """Chebyshev distance до найближчого фонового пікселя через
    ітеративне MinFilter(3×3) — fallback коли SciPy недоступний."""
    H, W = mask.shape
    current = (mask > 0).astype(np.uint8) * 255
    dist = np.zeros((H, W), dtype=np.float32)
    step = 0
    # Максимум кроків — половина найбільшої сторони
    max_steps = max(H, W)
    while current.any() and step < max_steps:
        img = Image.fromarray(current, mode="L")
        eroded = np.asarray(img.filter(ImageFilter.MinFilter(3)))
        step += 1
        newly_eroded = (current > 0) & (eroded == 0)
        dist[newly_eroded] = step
        current = eroded
    return dist


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """EDT для бінарної маски (1 = всередині, 0 = фон).  Повертає
    відстань до найближчого фонового пікселя."""
    if HAS_SCIPY:
        return distance_transform_edt(mask).astype(np.float32)
    return _edt_via_pil(mask)


# ──────────────────────────────────────────────
# 2.  HEIGHT MAP → NORMAL MAP (Sobel) — як у sobel_baseline
# ──────────────────────────────────────────────

def height_to_normal(height: np.ndarray, strength: float = 4.0) -> np.ndarray:
    """Те саме, що sobel_baseline.height_to_normal, але виділене окремо
    щоб уникнути циклічного імпорту."""
    gx = _convolve2d_same(height, SOBEL_X)
    gy = -_convolve2d_same(height, SOBEL_Y)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height, dtype=np.float32)
    length = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= length; ny /= length; nz /= length
    return np.stack([nx, ny, nz], axis=-1).astype(np.float32)


# ──────────────────────────────────────────────
# 3.  ОСНОВНИЙ ПРЕДИКТОР
# ──────────────────────────────────────────────

def predict_normal_bevel(rgba: np.ndarray,
                         alpha_thresh: float = 0.05,
                         strength: float = 4.0,
                         shape_exp: float = 0.5,
                         fill_background: bool = True) -> np.ndarray:
    """Передбачити нормаль методом "Beveling (distance transform)".

    Параметри
    ---------
    rgba          : (H, W, 4) uint8 — вхідне RGBA.
    alpha_thresh  : поріг альфи для маски силуета (0..1).
    strength      : сила Собеля (більше = крутіші схили).
    shape_exp     : показник степені для height (форма "купола"):
                    0.5  — еліпсоїд (сферична фаска),
                    1.0  — конус (рівномірний нахил),
                    2.0  — параболоїд (плаский верх, різкі краї).
    fill_background: заповнити прозорі пікселі плоскою нормаллю (0,0,1).
    """
    if rgba.dtype != np.uint8 or rgba.shape[-1] != 4:
        raise ValueError("rgba має бути uint8 з 4 каналами (RGBA)")

    alpha = rgba[..., 3].astype(np.float32) / 255.0
    mask = alpha > alpha_thresh

    if not mask.any():
        # Порожня альфа — повертаємо плоску нормаль
        flat = np.zeros((*rgba.shape[:2], 3), dtype=np.uint8)
        flat[..., 2] = 255
        return flat

    dist = distance_transform(mask)
    max_d = float(dist.max()) if dist.max() > 0 else 1.0
    height = dist / max_d                        # ∈ [0, 1]
    if shape_exp != 1.0:
        height = np.power(height, shape_exp).astype(np.float32)

    normal = height_to_normal(height, strength=strength)

    if fill_background:
        normal[~mask] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    return ((normal + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# 4.  EVAL ПАЙПЛАЙН (структура — як у sobel_baseline)
# ──────────────────────────────────────────────

def save_bevel_example_grid(input_rgba: np.ndarray,
                            height: np.ndarray,
                            bevel_normal: np.ndarray,
                            gt_normal_rgb: np.ndarray,
                            out_path: Path,
                            upscale: int = 4) -> None:
    """Зберегти 4-панельне зображення:
       вхід | height-map | Bevel-нормаль | GT.
    """
    h, w = input_rgba.shape[:2]

    # Конвертуємо карту висот у gray RGB для візуалізації
    height_vis = (height.clip(0, 1) * 255).astype(np.uint8)
    height_rgb = np.stack([height_vis] * 3, axis=-1)

    panels = [
        Image.fromarray(input_rgba, mode="RGBA").convert("RGB"),
        Image.fromarray(height_rgb, mode="RGB"),
        Image.fromarray(bevel_normal, mode="RGB"),
        Image.fromarray(gt_normal_rgb, mode="RGB"),
    ]
    new_w, new_h = w * upscale, h * upscale
    panels = [p.resize((new_w, new_h), Image.NEAREST) for p in panels]
    pad = 6
    canvas = Image.new("RGB",
                       (new_w * len(panels) + pad * (len(panels) - 1), new_h),
                       (24, 24, 24))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += new_w + pad
    canvas.save(out_path)


def run_evaluation(color_dir: Path, normal_dir: Path, output_dir: Path,
                   val_split: float, seed: int,
                   strength: float, shape_exp: float,
                   num_examples: int,
                   alpha_thresh: float = 0.05) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    edt_kind = "scipy.distance_transform_edt" if HAS_SCIPY \
        else "PIL.MinFilter (Chebyshev fallback)"
    print(f"[bevel] distance transform: {edt_kind}")
    print(f"[bevel] сканую {color_dir} та {normal_dir}…")

    pairs = build_pair_list(color_dir, normal_dir)
    if not pairs:
        raise SystemExit("Не знайдено жодної пари color↔normal.")
    sizes = [s for *_, s in pairs]
    val_idx = stratified_val_indices(sizes, val_split, seed)
    val_pairs = [pairs[i] for i in val_idx]
    val_size_counts = {s: [x[2] for x in val_pairs].count(s)
                       for s in sorted({x[2] for x in val_pairs})}
    print(f"[bevel] val-вибірка: {len(val_pairs)} пар; "
          f"по розмірах: {val_size_counts}")

    rows = []
    for k, (c_path, n_path, sz) in enumerate(val_pairs):
        try:
            with Image.open(c_path) as cimg:
                rgba = np.asarray(cimg.convert("RGBA"))
            with Image.open(n_path) as nimg:
                target = np.asarray(nimg.convert("RGB"))
        except Exception as e:
            print(f"  ! пропуск {c_path.name}: {e}")
            continue

        pred = predict_normal_bevel(rgba,
                                    alpha_thresh=alpha_thresh,
                                    strength=strength,
                                    shape_exp=shape_exp)
        alpha = rgba[..., 3]
        m = masked_metrics(pred, target, alpha, alpha_thresh)
        rows.append(dict(
            file=c_path.name, size=sz,
            n_valid=m["n_valid"],
            bevel_l1=m["l1"],
            bevel_ang=m["ang"],
            bevel_mae_deg=m["mae_deg"],
        ))
        if (k + 1) % 200 == 0:
            print(f"  оброблено {k + 1}/{len(val_pairs)}")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "metrics_per_image.csv", index=False)

    # Per-size + summary
    per_size_rows = []
    for s in sorted(df["size"].unique()):
        sub = df[df["size"] == s]
        per_size_rows.append(dict(
            size=int(s), n=len(sub),
            bevel_mae_deg=float(sub["bevel_mae_deg"].mean()),
            bevel_ang=float(sub["bevel_ang"].mean()),
            bevel_l1=float(sub["bevel_l1"].mean()),
        ))
    pd.DataFrame(per_size_rows).to_csv(output_dir / "metrics_per_size.csv",
                                       index=False)

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Beveling (distance transform) baseline\n")
        f.write("=" * 60 + "\n")
        f.write(f"Кількість зображень   : {len(df)}\n")
        f.write(f"Валідних пікселів     : {int(df['n_valid'].sum())}\n")
        f.write(f"val_split / seed      : {val_split} / {seed}\n")
        f.write(f"alpha_thresh          : {alpha_thresh}\n")
        f.write(f"strength (Собель)     : {strength}\n")
        f.write(f"shape_exp             : {shape_exp}\n")
        f.write(f"EDT backend           : {edt_kind}\n\n")
        f.write("                       L1        (1−cos)    MAE°\n")
        f.write(f"  bevel              "
                f"{df['bevel_l1'].mean():.4f}    "
                f"{df['bevel_ang'].mean():.4f}    "
                f"{df['bevel_mae_deg'].mean():.3f}\n\n")
        f.write("По розмірах:\n")
        f.write(pd.DataFrame(per_size_rows).to_string(index=False))
        f.write("\n")

    print(f"[bevel] зведення -> {output_dir/'summary.txt'}")

    # Приклади
    rng = random.Random(seed)
    by_size_pairs: dict[int, list] = {}
    for c, n, s in val_pairs:
        by_size_pairs.setdefault(s, []).append((c, n, s))
    sizes_sorted = sorted(by_size_pairs.keys())
    chosen = []
    per_bucket = max(1, num_examples // max(len(sizes_sorted), 1))
    for s in sizes_sorted:
        bucket = by_size_pairs[s][:]
        rng.shuffle(bucket)
        chosen.extend(bucket[:per_bucket])
    chosen = chosen[:num_examples]

    for i, (c, n, s) in enumerate(chosen, 1):
        with Image.open(c) as cimg:
            rgba = np.asarray(cimg.convert("RGBA"))
        with Image.open(n) as nimg:
            target = np.asarray(nimg.convert("RGB"))

        # Перерахуємо щоб збереги height-візуалізацію
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        mask = alpha > alpha_thresh
        if mask.any():
            dist = distance_transform(mask)
            max_d = float(dist.max()) if dist.max() > 0 else 1.0
            height = (dist / max_d) ** shape_exp
            height = height * mask
        else:
            height = np.zeros_like(alpha)

        pred = predict_normal_bevel(rgba,
                                    alpha_thresh=alpha_thresh,
                                    strength=strength,
                                    shape_exp=shape_exp)

        Image.fromarray(pred, mode="RGB").save(
            predictions_dir / f"{c.stem}_bevel.png")
        save_bevel_example_grid(
            rgba, height, pred, target,
            examples_dir / f"example_{i:02d}_s{s:03d}_{c.stem}.png")
    print(f"[bevel] приклади у {examples_dir}")


# ──────────────────────────────────────────────
# 5.  CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Beveling (distance transform) baseline")
    p.add_argument("--color_dir",    default="new_dataset/color")
    p.add_argument("--normal_dir",   default="new_dataset/normal")
    p.add_argument("--output_dir",   default="baseline/bevel_eval")
    p.add_argument("--val_split",    type=float, default=0.10)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--strength",     type=float, default=4.0)
    p.add_argument("--shape_exp",    type=float, default=0.5,
                   help="Степінь форми купола (0.5=еліпсоїд, 1=конус, 2=параболоїд)")
    p.add_argument("--num_examples", type=int,   default=10)
    p.add_argument("--alpha_thresh", type=float, default=0.05)

    p.add_argument("--predict",      action="store_true")
    p.add_argument("--input_image",  type=str, default=None)
    p.add_argument("--output_image", type=str, default="bevel_normal.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.predict:
        if not args.input_image:
            raise SystemExit("--predict потребує --input_image")
        with Image.open(args.input_image) as img:
            rgba = np.asarray(img.convert("RGBA"))
        normal = predict_normal_bevel(rgba,
                                      alpha_thresh=args.alpha_thresh,
                                      strength=args.strength,
                                      shape_exp=args.shape_exp)
        Image.fromarray(normal, mode="RGB").save(args.output_image)
        print(f"[predict] {args.input_image} -> {args.output_image}")
        return

    run_evaluation(
        color_dir   = Path(args.color_dir),
        normal_dir  = Path(args.normal_dir),
        output_dir  = Path(args.output_dir),
        val_split   = args.val_split,
        seed        = args.seed,
        strength    = args.strength,
        shape_exp   = args.shape_exp,
        num_examples = args.num_examples,
        alpha_thresh = args.alpha_thresh,
    )


if __name__ == "__main__":
    main()
