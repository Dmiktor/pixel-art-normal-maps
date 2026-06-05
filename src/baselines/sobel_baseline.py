"""
sobel_baseline.py
=================
Класичний бейзлайн для генерації мап нормалі з пікселізованих
зображень за допомогою оператора Собеля.

Реалізовано два варіанти:
  * "luminance"  — як висота використовується яскравість пікселя (Y' з RGB).
                   Це класичний метод, що порівнюється у статті
                   arxiv 2212.09692.
  * "alpha"      — як висота використовується альфа-канал (силует спрайта).
                   Корисний як простий контур-only метод.

Маска формується з альфа-каналу вхідного зображення: усі метрики
(angular error, L1) рахуються лише по непрозорих пікселях, аналогічно
як у train_normalmap.py (MaskedNormalLoss).

Виходи нормалей зберігаються у тій самій системі координат, що й
рендерені «target» нормалі: камера-спейс, RGB-кодування
N_rgb = (N + 1) / 2 * 255, де N ∈ [-1, 1]^3 — одиничний вектор.

CLI приклад
-----------
Прогнати валідаційну вибірку з new_dataset і зберегти CSV+приклади:

    python sobel_baseline.py \
        --color_dir  new_dataset/color  \
        --normal_dir new_dataset/normal \
        --output_dir baseline/sobel_eval \
        --val_split 0.10 --seed 42 \
        --num_examples 10

Прогнати тільки на одне зображення:

    python sobel_baseline.py \
        --predict --input_image sprite.png \
        --output_image sobel_normal.png \
        --height_mode luminance
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ──────────────────────────────────────────────
# 1.  HEIGHT MAP → NORMAL MAP (Sobel)
# ──────────────────────────────────────────────

# Стандартні ядра оператора Собеля 3×3.
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)


def _convolve2d_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Звичайна 2D-згортка (mode='same') без зовнішніх бібліотек.

    Використовує дзеркальне відображення на краях, щоб уникнути темних
    рамок типових для нульового заповнення.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + img.shape[0],
                                          j:j + img.shape[1]]
    return out


def height_to_normal(height: np.ndarray, strength: float = 4.0) -> np.ndarray:
    """Перетворити карту висот (H, W) ∈ [0, 1] на мапу нормалі (H, W, 3) ∈ [-1, 1].

    Конвенція камера-спейс:
      * X (red)   — горизонталь, +X праворуч
      * Y (green) — вертикаль, +Y вгору (рендерер пакує саме так)
      * Z (blue)  — нормаль "до камери", +Z

    `strength` керує висотою рельєфу — більше = різкіші схили.
    """
    if height.ndim != 2:
        raise ValueError("height має бути двовимірним масивом (H, W)")

    gx = _convolve2d_same(height, SOBEL_X)
    # У Sobel_Y +Y направлено вниз (рядки збільшуються вниз), тож для
    # «+Y вгору» у нормалі необхідне інвертування знаку.
    gy = -_convolve2d_same(height, SOBEL_Y)

    nx = -gx * strength          # схил → нормаль направлена *проти* схилу
    ny = -gy * strength
    nz = np.ones_like(height, dtype=np.float32)

    length = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= length
    ny /= length
    nz /= length
    return np.stack([nx, ny, nz], axis=-1).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Стандартна Rec.601 яскравість: 0.299R + 0.587G + 0.114B."""
    return (0.299 * rgb[..., 0] +
            0.587 * rgb[..., 1] +
            0.114 * rgb[..., 2]).astype(np.float32)


def predict_normal_sobel(rgba: np.ndarray,
                         height_mode: str = "luminance",
                         strength: float = 4.0,
                         fill_background: bool = True) -> np.ndarray:
    """Передбачити мапу нормалі для пікселізованого зображення.

    Параметри
    ---------
    rgba          : (H, W, 4) uint8 — вхідне зображення RGBA.
    height_mode   : "luminance" | "alpha".
    strength      : чутливість Собеля (більше = різкіший рельєф).
    fill_background: чи замінювати прозорі пікселі на (0,0,1) (плоска
                     нормаль "до камери") — як це робить наш рендерер.

    Повертає
    -------
    normal_rgb    : (H, W, 3) uint8 — мапа нормалі у форматі RGB.
    """
    if rgba.dtype != np.uint8:
        raise TypeError("rgba має бути uint8")
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("rgba має форму (H, W, 4)")

    arr = rgba.astype(np.float32) / 255.0
    alpha = arr[..., 3]

    if height_mode == "luminance":
        # Премультиплікуємо альфою, щоб краї спрайта не давали ложних
        # градієнтів від прозорого фону.
        rgb = arr[..., :3] * alpha[..., None]
        height = luminance(rgb)
    elif height_mode == "alpha":
        height = alpha
    else:
        raise ValueError("height_mode має бути 'luminance' або 'alpha'")

    normal = height_to_normal(height, strength=strength)        # H W 3 ∈ [-1, 1]

    if fill_background:
        bg_mask = alpha <= (5.0 / 255.0)
        normal[bg_mask] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    normal_rgb = ((normal + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    return normal_rgb


# ──────────────────────────────────────────────
# 2.  МЕТРИКИ (точно як у train_normalmap.py — masked)
# ──────────────────────────────────────────────

def _to_unit(n_rgb: np.ndarray) -> np.ndarray:
    """Декодувати RGB-нормаль (uint8) у одиничний вектор float32 (H, W, 3)."""
    v = n_rgb.astype(np.float32) / 255.0
    v = v * 2.0 - 1.0
    length = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6
    return v / length


def masked_metrics(pred_rgb: np.ndarray,
                   target_rgb: np.ndarray,
                   alpha: np.ndarray,
                   alpha_thresh: float = 0.05) -> dict:
    """Обчислити маскований L1, angular (1 − cos) та MAE у градусах.

    pred_rgb, target_rgb : (H, W, 3) uint8
    alpha                : (H, W) uint8 — альфа вхідного спрайта
    """
    pred  = _to_unit(pred_rgb)
    tgt   = _to_unit(target_rgb)
    mask  = (alpha.astype(np.float32) / 255.0) > alpha_thresh
    n_valid = max(int(mask.sum()), 1)

    diff = np.abs(pred - tgt).sum(axis=-1)                 # H W (сума по 3 каналах)
    l1   = float((diff * mask).sum() / (n_valid * 3.0))

    cos  = (pred * tgt).sum(axis=-1)                       # H W
    ang  = float(((1.0 - cos) * mask).sum() / n_valid)

    cos_c = np.clip(cos, -1.0 + 1e-6, 1.0 - 1e-6)
    deg   = np.arccos(cos_c) * (180.0 / math.pi)
    mae_deg = float((deg * mask).sum() / n_valid)

    return dict(l1=l1, ang=ang, mae_deg=mae_deg, n_valid=n_valid)


# ──────────────────────────────────────────────
# 3.  ПОБУДОВА СПИСКУ ПАР І СТРАТИФІКОВАНИЙ SPLIT
# ──────────────────────────────────────────────

_SIZE_RX = re.compile(r"_s(\d{2,4})")


def _parse_size_from_name(name: str) -> int:
    """Витягнути розмір з імені файлу формату `..._s064...`."""
    m = _SIZE_RX.search(name)
    return int(m.group(1)) if m else 0


def build_pair_list(color_dir: Path, normal_dir: Path) -> list[tuple[Path, Path, int]]:
    """Перелік усіх пар (color, normal) з однаковими іменами."""
    color_files = {p.name: p for p in color_dir.glob("*.png")}
    normal_files = {p.name: p for p in normal_dir.glob("*.png")}
    common = sorted(set(color_files) & set(normal_files))
    pairs = []
    for name in common:
        sz = _parse_size_from_name(name)
        if sz == 0:
            continue
        pairs.append((color_files[name], normal_files[name], sz))
    return pairs


def stratified_val_indices(sizes: list[int], val_split: float,
                           seed: int) -> list[int]:
    """Та сама логіка, що в train_normalmap.stratified_split_indices.

    Кожна корзина розміру отримує val_split * n валідаційних зразків
    (мінімум 1). Повертає список індексів val-вибірки.
    """
    rng = random.Random(seed)
    by_size: dict[int, list[int]] = {}
    for i, s in enumerate(sizes):
        by_size.setdefault(s, []).append(i)

    val_idx: list[int] = []
    for s, idxs in by_size.items():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_split)))
        val_idx.extend(idxs[:n_val])
    rng.shuffle(val_idx)
    return val_idx


# ──────────────────────────────────────────────
# 4.  ЗБЕРЕЖЕННЯ ПРИКЛАДІВ
# ──────────────────────────────────────────────

def save_example_grid(input_rgba: np.ndarray,
                      sobel_lum: np.ndarray,
                      sobel_alpha: np.ndarray,
                      gt_normal_rgb: np.ndarray,
                      out_path: Path,
                      upscale: int = 4) -> None:
    """Зберегти бок-о-бок зображення:
       вхід | Sobel(lum) | Sobel(alpha) | ground-truth.

    Для збереження пікселізованого вигляду застосовується NEAREST-апскейл.
    """
    h, w = input_rgba.shape[:2]
    panels = []
    panels.append(Image.fromarray(input_rgba, mode="RGBA").convert("RGB"))
    panels.append(Image.fromarray(sobel_lum, mode="RGB"))
    panels.append(Image.fromarray(sobel_alpha, mode="RGB"))
    panels.append(Image.fromarray(gt_normal_rgb, mode="RGB"))

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


# ──────────────────────────────────────────────
# 5.  ОСНОВНИЙ ПАЙПЛАЙН
# ──────────────────────────────────────────────

def run_evaluation(color_dir: Path, normal_dir: Path, output_dir: Path,
                   val_split: float, seed: int,
                   strength: float, num_examples: int,
                   alpha_thresh: float = 0.05) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    print(f"[sobel] сканую {color_dir} та {normal_dir}…")
    pairs = build_pair_list(color_dir, normal_dir)
    if not pairs:
        raise SystemExit("Не знайдено жодної пари color↔normal.")
    sizes = [p[2] for p in pairs]
    size_counts = {s: sizes.count(s) for s in sorted(set(sizes))}
    print(f"[sobel] усього пар: {len(pairs)}; по розмірах: {size_counts}")

    val_idx = stratified_val_indices(sizes, val_split, seed)
    val_pairs = [pairs[i] for i in val_idx]
    val_sizes = [s for *_, s in val_pairs]
    val_size_counts = {s: val_sizes.count(s) for s in sorted(set(val_sizes))}
    print(f"[sobel] валідаційна вибірка: {len(val_pairs)} пар; "
          f"по розмірах: {val_size_counts}")

    # ── обробка кожної пари ────────────────────────────────────────────
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

        pred_lum   = predict_normal_sobel(rgba, "luminance", strength)
        pred_alpha = predict_normal_sobel(rgba, "alpha",     strength)

        alpha = rgba[..., 3]
        m_lum   = masked_metrics(pred_lum,   target, alpha, alpha_thresh)
        m_alpha = masked_metrics(pred_alpha, target, alpha, alpha_thresh)

        rows.append(dict(
            file=c_path.name, size=sz,
            n_valid=m_lum["n_valid"],
            lum_l1=m_lum["l1"],   lum_ang=m_lum["ang"],   lum_mae_deg=m_lum["mae_deg"],
            alpha_l1=m_alpha["l1"], alpha_ang=m_alpha["ang"], alpha_mae_deg=m_alpha["mae_deg"],
        ))

        if (k + 1) % 200 == 0:
            print(f"  оброблено {k + 1}/{len(val_pairs)}")

    df = pd.DataFrame(rows)
    csv_path = output_dir / "metrics_per_image.csv"
    df.to_csv(csv_path, index=False)
    print(f"[sobel] записав покадрові метрики у {csv_path}")

    # ── зведена статистика ─────────────────────────────────────────────
    def agg(col: str) -> dict:
        return dict(mean=float(df[col].mean()),
                    median=float(df[col].median()),
                    std=float(df[col].std()))

    summary = {
        "n_images":          int(len(df)),
        "n_valid_pixels":    int(df["n_valid"].sum()),
        "val_split":         val_split,
        "seed":              seed,
        "alpha_thresh":      alpha_thresh,
        "strength":          strength,
        "luminance":         {"l1": agg("lum_l1"),
                              "ang": agg("lum_ang"),
                              "mae_deg": agg("lum_mae_deg")},
        "alpha":             {"l1": agg("alpha_l1"),
                              "ang": agg("alpha_ang"),
                              "mae_deg": agg("alpha_mae_deg")},
    }

    # по розмірах
    per_size_rows = []
    for s in sorted(df["size"].unique()):
        sub = df[df["size"] == s]
        per_size_rows.append(dict(
            size=int(s), n=len(sub),
            lum_mae_deg=float(sub["lum_mae_deg"].mean()),
            lum_ang=float(sub["lum_ang"].mean()),
            lum_l1=float(sub["lum_l1"].mean()),
            alpha_mae_deg=float(sub["alpha_mae_deg"].mean()),
            alpha_ang=float(sub["alpha_ang"].mean()),
            alpha_l1=float(sub["alpha_l1"].mean()),
        ))
    per_size_df = pd.DataFrame(per_size_rows)
    per_size_df.to_csv(output_dir / "metrics_per_size.csv", index=False)

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Sobel baseline — підсумкові метрики на val-split\n")
        f.write("=" * 60 + "\n")
        f.write(f"Кількість зображень   : {summary['n_images']}\n")
        f.write(f"Валідних пікселів     : {summary['n_valid_pixels']}\n")
        f.write(f"val_split / seed      : {val_split} / {seed}\n")
        f.write(f"alpha_thresh          : {alpha_thresh}\n")
        f.write(f"strength (Собель)     : {strength}\n\n")
        f.write("                       L1        (1−cos)    MAE°\n")
        for mode in ("luminance", "alpha"):
            m = summary[mode]
            f.write(f"  {mode:<10}        "
                    f"{m['l1']['mean']:.4f}    "
                    f"{m['ang']['mean']:.4f}    "
                    f"{m['mae_deg']['mean']:.3f}\n")
        f.write("\nПо розмірах:\n")
        f.write(per_size_df.to_string(index=False))
        f.write("\n")
    print(f"[sobel] зведення збережено у {summary_path}")

    # ── приклади ───────────────────────────────────────────────────────
    # Вибираємо по 1-2 з кожного розміру щоб охопити всі типи
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

    print(f"[sobel] зберігаю {len(chosen)} прикладів…")
    for i, (c, n, s) in enumerate(chosen, 1):
        with Image.open(c) as cimg:
            rgba = np.asarray(cimg.convert("RGBA"))
        with Image.open(n) as nimg:
            target = np.asarray(nimg.convert("RGB"))

        pred_lum   = predict_normal_sobel(rgba, "luminance", strength)
        pred_alpha = predict_normal_sobel(rgba, "alpha",     strength)

        # збережемо передбачення окремо (потім допоможе при діагностиці)
        Image.fromarray(pred_lum,   mode="RGB").save(predictions_dir / f"{c.stem}_sobel_lum.png")
        Image.fromarray(pred_alpha, mode="RGB").save(predictions_dir / f"{c.stem}_sobel_alpha.png")

        save_example_grid(rgba, pred_lum, pred_alpha, target,
                          examples_dir / f"example_{i:02d}_s{s:03d}_{c.stem}.png")
    print(f"[sobel] приклади у {examples_dir}")


# ──────────────────────────────────────────────
# 6.  CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sobel-baseline для нормалей з піксель-арту")
    p.add_argument("--color_dir",   default="new_dataset/color")
    p.add_argument("--normal_dir",  default="new_dataset/normal")
    p.add_argument("--output_dir",  default="baseline/sobel_eval")
    p.add_argument("--val_split",   type=float, default=0.10)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--strength",    type=float, default=4.0)
    p.add_argument("--num_examples", type=int,  default=10)
    p.add_argument("--alpha_thresh", type=float, default=0.05)

    p.add_argument("--predict",      action="store_true",
                   help="Згенерувати одне зображення замість прогону датасета")
    p.add_argument("--input_image",  type=str, default=None)
    p.add_argument("--output_image", type=str, default="sobel_normal.png")
    p.add_argument("--height_mode",  choices=("luminance", "alpha"),
                   default="luminance")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.predict:
        if not args.input_image:
            raise SystemExit("--predict потребує --input_image")
        with Image.open(args.input_image) as img:
            rgba = np.asarray(img.convert("RGBA"))
        normal = predict_normal_sobel(rgba, args.height_mode, args.strength)
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
        num_examples = args.num_examples,
        alpha_thresh = args.alpha_thresh,
    )


if __name__ == "__main__":
    main()
