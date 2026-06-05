"""
compare_methods.py
==================
Зводить разом метрики чотирьох методів:
  * Sobel (luminance)   — baseline/sobel_eval/metrics_per_image.csv
  * Sobel (alpha)       — baseline/sobel_eval/metrics_per_image.csv
  * Bevel (EDT)         — baseline/bevel_eval/metrics_per_image.csv
  * U-Net (exp_balanced)— baseline/model_eval/metrics_per_image.csv

Виходи
------
  baseline/comparison/comparison_summary.csv
  baseline/comparison/comparison_summary.txt
  baseline/comparison/examples_quint/example_NN.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def panels_grid(panels, labels, upscale: int = 4) -> Image.Image:
    """Скласти підписані панелі пліч-о-пліч."""
    h, w = panels[0].shape[:2]
    pad = 6
    label_h = 18
    new_w, new_h = w * upscale, h * upscale
    canvas = Image.new(
        "RGB",
        (new_w * len(panels) + pad * (len(panels) - 1), new_h + label_h),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    x = 0
    for i, p in enumerate(panels):
        if p.ndim == 3 and p.shape[2] == 4:
            img = Image.fromarray(p, mode="RGBA").convert("RGB")
        else:
            img = Image.fromarray(p)
        img = img.resize((new_w, new_h), Image.NEAREST)
        canvas.paste(img, (x, label_h))
        try:
            draw.text((x + 4, 2), labels[i], fill=(220, 220, 220))
        except Exception:
            pass
        x += new_w + pad
    return canvas


def _agg(sub: pd.DataFrame, col: str) -> float:
    if col not in sub:
        return float("nan")
    return float(sub[col].mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sobel_dir",   default="baseline/sobel_eval")
    p.add_argument("--bevel_dir",   default="baseline/bevel_eval")
    p.add_argument("--model_dir",   default="baseline/model_eval")
    p.add_argument("--color_dir",   default="new_dataset/color")
    p.add_argument("--normal_dir",  default="new_dataset/normal")
    p.add_argument("--out_dir",     default="baseline/comparison")
    p.add_argument("--num_examples", type=int, default=10)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = out_dir / "examples_quint"
    examples_dir.mkdir(exist_ok=True)

    sobel = pd.read_csv(Path(args.sobel_dir) / "metrics_per_image.csv")
    bevel = pd.read_csv(Path(args.bevel_dir) / "metrics_per_image.csv")
    model_path = Path(args.model_dir) / "metrics_per_image.csv"
    has_model = model_path.exists()
    model = pd.read_csv(model_path) if has_model else None

    merged = pd.merge(sobel, bevel, on=["file", "size", "n_valid"], how="inner")
    if has_model:
        merged = pd.merge(merged, model,
                          on=["file", "size", "n_valid"], how="inner")
    merged.to_csv(out_dir / "merged_per_image.csv", index=False)

    def gather(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for s in list(sorted(df["size"].unique())) + [-1]:
            sub = df if s == -1 else df[df["size"] == s]
            row = dict(size=int(s), n=len(sub))
            row["sobel_lum_mae"]   = _agg(sub, "lum_mae_deg")
            row["sobel_alpha_mae"] = _agg(sub, "alpha_mae_deg")
            row["bevel_mae"]       = _agg(sub, "bevel_mae_deg")
            row["sobel_lum_ang"]   = _agg(sub, "lum_ang")
            row["sobel_alpha_ang"] = _agg(sub, "alpha_ang")
            row["bevel_ang"]       = _agg(sub, "bevel_ang")
            row["sobel_lum_l1"]    = _agg(sub, "lum_l1")
            row["sobel_alpha_l1"]  = _agg(sub, "alpha_l1")
            row["bevel_l1"]        = _agg(sub, "bevel_l1")
            if has_model:
                row["unet_mae"] = _agg(sub, "model_mae_deg")
                row["unet_ang"] = _agg(sub, "model_ang")
                row["unet_l1"]  = _agg(sub, "model_l1")
            rows.append(row)
        return pd.DataFrame(rows)

    summary = gather(merged)
    summary.to_csv(out_dir / "comparison_summary.csv", index=False)

    cols_model = ["Sobel-lum", "Sobel-α", "Bevel(EDT)"]
    if has_model:
        cols_model.append("U-Net")

    def tbl(f, title, lum_c, a_c, b_c, u_c, fmt="{:>9.3f}"):
        f.write(title + "\n")
        header = "size     n   " + "  ".join(f"{m:>10}" for m in cols_model)
        f.write(header + "\n")
        for _, r in summary.iterrows():
            tag = "ALL" if r["size"] == -1 else f"{int(r['size'])}px"
            vals = [r[lum_c], r[a_c], r[b_c]]
            if has_model:
                vals.append(r[u_c])
            f.write(f"  {tag:<6} {int(r['n']):<5} " +
                    "  ".join(fmt.format(v) for v in vals) + "\n")
        f.write("\n")

    with (out_dir / "comparison_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Порівняння методів (масковані метрики, лише непрозорі пікселі)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Зображень   : {len(merged)}\n")
        f.write(f"U-Net у наборі: {'так' if has_model else 'ні (запустіть eval_model.py)'}\n\n")

        tbl(f, "MAE кута (градуси), нижче — краще",
            "sobel_lum_mae", "sobel_alpha_mae", "bevel_mae", "unet_mae")
        tbl(f, "Кут (1 − cos θ), нижче — краще",
            "sobel_lum_ang", "sobel_alpha_ang", "bevel_ang", "unet_ang",
            fmt="{:>9.4f}")
        tbl(f, "L1 (нормалі у [-1,1]), нижче — краще",
            "sobel_lum_l1", "sobel_alpha_l1", "bevel_l1", "unet_l1",
            fmt="{:>9.4f}")

    print(f"[compare] зведення -> {out_dir / 'comparison_summary.txt'}")

    color_dir  = Path(args.color_dir)
    normal_dir = Path(args.normal_dir)
    sobel_pred_dir = Path(args.sobel_dir) / "predictions"
    bevel_pred_dir = Path(args.bevel_dir) / "predictions"
    model_pred_dir = Path(args.model_dir) / "predictions"

    sobel_examples_dir = Path(args.sobel_dir) / "examples"
    if not sobel_examples_dir.exists():
        print("[compare] не знайдено Sobel-прикладів, пропускаю композиції")
        return

    n_saved = 0
    for example_path in sorted(sobel_examples_dir.glob("example_*.png")):
        parts = example_path.stem.split("_", 3)
        if len(parts) < 4:
            continue
        stem = parts[3]
        input_path = color_dir / f"{stem}.png"
        gt_path    = normal_dir / f"{stem}.png"
        sob_path   = sobel_pred_dir / f"{stem}_sobel_lum.png"
        sob_a_path = sobel_pred_dir / f"{stem}_sobel_alpha.png"
        bev_path   = bevel_pred_dir / f"{stem}_bevel.png"
        unet_path  = model_pred_dir / f"{stem}_unet.png"

        required = [input_path, gt_path, sob_path, sob_a_path, bev_path]
        if not all(pth.exists() for pth in required):
            continue

        rgba  = np.asarray(Image.open(input_path).convert("RGBA"))
        gt    = np.asarray(Image.open(gt_path).convert("RGB"))
        sob   = np.asarray(Image.open(sob_path).convert("RGB"))
        sob_a = np.asarray(Image.open(sob_a_path).convert("RGB"))
        bev   = np.asarray(Image.open(bev_path).convert("RGB"))

        panels = [rgba, sob, sob_a, bev]
        labels = ["input", "Sobel (lum)", "Sobel (α)", "Bevel (EDT)"]
        if unet_path.exists():
            unet = np.asarray(Image.open(unet_path).convert("RGB"))
            panels.append(unet)
            labels.append("U-Net")
        panels.append(gt)
        labels.append("ground truth")

        grid = panels_grid(panels, labels)
        n_saved += 1
        grid.save(examples_dir / f"example_{n_saved:02d}_{stem}.png")
        if n_saved >= args.num_examples:
            break

    print(f"[compare] збережено {n_saved} мульти-панельних прикладів у {examples_dir}")


if __name__ == "__main__":
    main()
