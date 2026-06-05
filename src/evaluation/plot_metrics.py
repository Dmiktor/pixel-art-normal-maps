"""
plot_metrics.py
===============
Generates publication-quality plots from the metrics.csv produced by the
training scripts (train_normalmap.py / train_normalmap_old.py).

    python plot_metrics.py --csv results/exp_balanced/metrics.csv \
                           --out results/exp_balanced/plots

The two training scripts emit slightly different column names, so this script
normalises them and skips any plot whose columns are absent:

  * angular column : "train_angular"/"val_angular" (old) or
                     "train_ang"/"val_ang" (new) -- both accepted.
  * PSNR / SSIM    : present only for the old U-Net run; the PSNR/SSIM plot
                     is produced only when those columns exist.

Produces (when the required columns are present):
  - loss_curves.png      - train vs val combined loss
  - l1_angular.png       - L1 and angular loss components
  - angular_error.png    - mean angular error in degrees (validation)
  - psnr_ssim.png        - PSNR and SSIM (validation)   [old schema only]
  - lr_schedule.png      - learning-rate schedule
  - summary_grid.png     - all available metrics in one figure (for the thesis)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})

COLORS = {
    "train": "#2563EB", "val": "#DC2626", "l1": "#059669",
    "ang": "#D97706", "psnr": "#7C3AED", "ssim": "#DB2777", "lr": "#6B7280",
}

# Aliases so both training-script schemas load identically.
ALIASES = {
    "train_angular": "train_ang",
    "val_angular":   "val_ang",
}


def load_csv(path: str) -> dict:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    if not rows:
        raise RuntimeError("CSV is empty")
    d = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    # Make both naming conventions available.
    for long_name, short_name in ALIASES.items():
        if long_name in d and short_name not in d:
            d[short_name] = d[long_name]
        if short_name in d and long_name not in d:
            d[long_name] = d[short_name]
    return d


def has(d, *keys):
    return all(k in d for k in keys)


def smooth(arr, w=5):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")


def save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def plot_loss(d, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    ep = d["epoch"]
    ax.plot(ep, d["train_loss"], color=COLORS["train"], alpha=0.3, lw=1)
    ax.plot(ep, smooth(d["train_loss"]), color=COLORS["train"], lw=2, label="Train loss")
    ax.plot(ep, d["val_loss"], color=COLORS["val"], alpha=0.3, lw=1)
    ax.plot(ep, smooth(d["val_loss"]), color=COLORS["val"], lw=2, label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Combined loss")
    ax.set_title("Training and validation loss"); ax.legend()
    save(fig, out / "loss_curves.png")


def plot_l1_angular(d, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = d["epoch"]
    for ax, key, title in [(ax1, "l1", "L1 loss"), (ax2, "ang", "Angular loss (rad)")]:
        ax.plot(ep, d[f"train_{key}"], color=COLORS["train"], alpha=0.3, lw=1)
        ax.plot(ep, smooth(d[f"train_{key}"]), color=COLORS["train"], lw=2, label="Train")
        ax.plot(ep, d[f"val_{key}"], color=COLORS["val"], alpha=0.3, lw=1)
        ax.plot(ep, smooth(d[f"val_{key}"]), color=COLORS["val"], lw=2, label="Val")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title); ax.set_title(title); ax.legend()
    fig.suptitle("Loss components", fontsize=13, fontweight="bold")
    save(fig, out / "l1_angular.png")


def plot_angular_error(d, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    ep = d["epoch"]
    ax.plot(ep, d["val_mae_deg"], color=COLORS["ang"], alpha=0.3, lw=1)
    ax.plot(ep, smooth(d["val_mae_deg"]), color=COLORS["ang"], lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean angular error (deg)")
    ax.set_title("Mean angular error on validation set")
    save(fig, out / "angular_error.png")


def plot_psnr_ssim(d, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = d["epoch"]
    ax1.plot(ep, d["val_psnr"], color=COLORS["psnr"], alpha=0.3, lw=1)
    ax1.plot(ep, smooth(d["val_psnr"]), color=COLORS["psnr"], lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Peak Signal-to-Noise Ratio (validation)")
    ax2.plot(ep, d["val_ssim"], color=COLORS["ssim"], alpha=0.3, lw=1)
    ax2.plot(ep, smooth(d["val_ssim"]), color=COLORS["ssim"], lw=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("SSIM"); ax2.set_ylim(0, 1)
    ax2.set_title("Structural Similarity Index (validation)")
    save(fig, out / "psnr_ssim.png")


def plot_lr(d, out):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(d["epoch"], d["lr"], color=COLORS["lr"], lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
    ax.set_title("OneCycleLR schedule")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    save(fig, out / "lr_schedule.png")


def plot_summary_grid(d, out):
    panels = [("Combined loss", [("train_loss", "Train", COLORS["train"]),
                                 ("val_loss", "Val", COLORS["val"])]),
              ("L1 loss", [("train_l1", "Train", COLORS["train"]),
                           ("val_l1", "Val", COLORS["val"])]),
              ("Angular loss (rad)", [("train_ang", "Train", COLORS["train"]),
                                      ("val_ang", "Val", COLORS["val"])]),
              ("Mean angular error (deg)", [("val_mae_deg", "Val", COLORS["ang"])])]
    if has(d, "val_psnr"):
        panels.append(("PSNR (dB)", [("val_psnr", "Val", COLORS["psnr"])]))
    if has(d, "val_ssim"):
        panels.append(("SSIM", [("val_ssim", "Val", COLORS["ssim"])]))

    panels = [p for p in panels if all(k in d for k, _, _ in p[1])]
    ncol = 2
    nrow = (len(panels) + ncol - 1) // ncol
    fig = plt.figure(figsize=(7 * ncol, 3.3 * nrow))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, hspace=0.45, wspace=0.3)
    ep = d["epoch"]
    for i, (title, series) in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        for key, label, color in series:
            ax.plot(ep, d[key], color=color, alpha=0.25, lw=1)
            ax.plot(ep, smooth(d[key]), color=color, lw=2, label=label)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        if len(series) > 1:
            ax.legend(fontsize=8)
    fig.suptitle("Training metrics overview", fontsize=14, fontweight="bold", y=1.01)
    save(fig, out / "summary_grid.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/exp_balanced/metrics.csv")
    p.add_argument("--out", default="results/exp_balanced/plots")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    d = load_csv(args.csv)
    print(f"Loaded {int(d['epoch'][-1])} epochs from {args.csv}")

    plot_loss(d, out)
    if has(d, "train_l1", "val_l1", "train_ang", "val_ang"):
        plot_l1_angular(d, out)
    if has(d, "val_mae_deg"):
        plot_angular_error(d, out)
    if has(d, "val_psnr", "val_ssim"):
        plot_psnr_ssim(d, out)
    if has(d, "lr"):
        plot_lr(d, out)
    plot_summary_grid(d, out)
    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
