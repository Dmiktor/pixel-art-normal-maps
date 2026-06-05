"""
eval_model.py
=============
Запускає навчену U-Net модель (train_normalmap.py) на ТІЙ САМІЙ
валідаційній вибірці, що й sobel_baseline.py, та обчислює маскований
angular error, L1, MAE у градусах для кожного зображення.

Це дає прямий per-image зіставник із Sobel-бейзлайном.

Залежності
----------
* torch (як для train_normalmap.py)
* PIL, numpy, pandas

Виклик
------
    python eval_model.py \
        --color_dir  new_dataset/color  \
        --normal_dir new_dataset/normal \
        --checkpoint runs/exp_balanced/checkpoints/best.pth \
        --output_dir baseline/model_eval \
        --val_split 0.10 --seed 42 \
        --num_examples 10
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F

# Імпортуємо архітектуру з train_normalmap.py, що поруч.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_normalmap import UNet  # noqa: E402

# Імпортуємо ті ж самі допоміжні функції з sobel_baseline.py,
# щоб гарантувати ідентичний split та формат прикладів.
from sobel_baseline import (              # noqa: E402
    build_pair_list,
    stratified_val_indices,
    save_example_grid,
    masked_metrics,
)


def _round_up_to(n: int, mult: int) -> int:
    return ((n + mult - 1) // mult) * mult


@torch.no_grad()
def model_predict(model: torch.nn.Module, rgba: np.ndarray,
                  device: torch.device) -> np.ndarray:
    """Прогнати модель на одному RGBA-зображенні, повернути RGB-нормаль.

    Логіка ідентична `predict_single` з train_normalmap.py: розмір
    зображення доводиться до найближчого кратного 8 паддингом нулями,
    модель повертає одиничні вектори, потім картинка кропиться назад
    до оригінального розміру.
    """
    h, w = rgba.shape[:2]
    side = _round_up_to(max(h, w), 8)
    py, px = (side - h) // 2, (side - w) // 2
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    canvas[py:py + h, px:px + w] = rgba

    arr = canvas.astype(np.float32) / 255.0          # H W 4
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

    pred = model(t)[0].cpu().numpy()                 # 3 H W ∈ [-1, 1] (unit-length)
    out = ((pred + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    out = out.transpose(1, 2, 0)                     # H W 3
    return out[py:py + h, px:px + w]


def main() -> None:
    p = argparse.ArgumentParser(description="Прогон U-Net на val-split (для порівняння з Sobel)")
    p.add_argument("--color_dir",   default="new_dataset/color")
    p.add_argument("--normal_dir",  default="new_dataset/normal")
    p.add_argument("--checkpoint",  default="runs/exp_balanced/checkpoints/best.pth")
    p.add_argument("--output_dir",  default="baseline/model_eval")
    p.add_argument("--val_split",   type=float, default=0.10)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_examples", type=int,  default=10)
    p.add_argument("--alpha_thresh", type=float, default=0.05)
    p.add_argument("--device",      default=None,
                   help="cuda | cpu (за замовчуванням — автодетект)")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[eval] device={device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    # ── завантажуємо чекпойнт ─────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    model = UNet(
        in_ch     = 4, out_ch = 3,
        base      = cfg.get("base_channels", 64),
        enc_blocks = cfg.get("enc_blocks", 2),
        bot_blocks = cfg.get("bot_blocks", 4),
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"[eval] чекпойнт {args.checkpoint}; епоха={ckpt.get('epoch', '?')}")

    # ── формуємо ТОЙ САМИЙ val-split, що й sobel_baseline ─────────────
    color_dir  = Path(args.color_dir)
    normal_dir = Path(args.normal_dir)
    pairs = build_pair_list(color_dir, normal_dir)
    sizes = [s for *_, s in pairs]
    val_idx = stratified_val_indices(sizes, args.val_split, args.seed)
    val_pairs = [pairs[i] for i in val_idx]
    print(f"[eval] val-вибірка: {len(val_pairs)} пар")

    # ── прохід ────────────────────────────────────────────────────────
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

        pred = model_predict(model, rgba, device)

        alpha = rgba[..., 3]
        m = masked_metrics(pred, target, alpha, args.alpha_thresh)
        rows.append(dict(
            file=c_path.name, size=sz,
            n_valid=m["n_valid"],
            model_l1=m["l1"],
            model_ang=m["ang"],
            model_mae_deg=m["mae_deg"],
        ))

        if (k + 1) % 100 == 0:
            print(f"  оброблено {k + 1}/{len(val_pairs)}")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "metrics_per_image.csv", index=False)

    # ── зведення ──────────────────────────────────────────────────────
    def agg(col: str) -> dict:
        return dict(mean=float(df[col].mean()),
                    median=float(df[col].median()),
                    std=float(df[col].std()))

    per_size_rows = []
    for s in sorted(df["size"].unique()):
        sub = df[df["size"] == s]
        per_size_rows.append(dict(
            size=int(s), n=len(sub),
            model_mae_deg=float(sub["model_mae_deg"].mean()),
            model_ang=float(sub["model_ang"].mean()),
            model_l1=float(sub["model_l1"].mean()),
        ))
    pd.DataFrame(per_size_rows).to_csv(output_dir / "metrics_per_size.csv",
                                       index=False)

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("U-Net (exp_balanced) — метрики на val-split\n")
        f.write("=" * 60 + "\n")
        f.write(f"Кількість зображень   : {len(df)}\n")
        f.write(f"Валідних пікселів     : {int(df['n_valid'].sum())}\n")
        f.write(f"val_split / seed      : {args.val_split} / {args.seed}\n")
        f.write(f"alpha_thresh          : {args.alpha_thresh}\n\n")
        f.write("                  mean    median    std\n")
        for col in ("model_l1", "model_ang", "model_mae_deg"):
            a = agg(col)
            f.write(f"  {col:<14}  {a['mean']:.4f}   "
                    f"{a['median']:.4f}   {a['std']:.4f}\n")

    print(f"[eval] зведення -> {output_dir/'summary.txt'}")

    # ── приклади (ті ж самі індекси, що й Sobel) ──────────────────────
    rng = random.Random(args.seed)
    by_size_pairs: dict[int, list] = {}
    for c, n, s in val_pairs:
        by_size_pairs.setdefault(s, []).append((c, n, s))
    sizes_sorted = sorted(by_size_pairs.keys())
    chosen = []
    per_bucket = max(1, args.num_examples // max(len(sizes_sorted), 1))
    for s in sizes_sorted:
        bucket = by_size_pairs[s][:]
        rng.shuffle(bucket)
        chosen.extend(bucket[:per_bucket])
    chosen = chosen[:args.num_examples]

    for i, (c, n, s) in enumerate(chosen, 1):
        with Image.open(c) as cimg:
            rgba = np.asarray(cimg.convert("RGBA"))
        with Image.open(n) as nimg:
            target = np.asarray(nimg.convert("RGB"))

        pred = model_predict(model, rgba, device)
        Image.fromarray(pred, mode="RGB").save(
            predictions_dir / f"{c.stem}_unet.png")

        # save_example_grid очікує 4 панелі: input, sobel_lum, sobel_alpha, GT —
        # тут підставляємо: input | model | model | GT, щоб формат збігся.
        save_example_grid(rgba, pred, pred, target,
                          examples_dir / f"example_{i:02d}_s{s:03d}_{c.stem}.png")
    print(f"[eval] приклади у {examples_dir}")


if __name__ == "__main__":
    main()
