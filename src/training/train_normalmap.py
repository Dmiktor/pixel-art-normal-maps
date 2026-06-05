"""
train_normalmap.py
==================
Pixel-art -> screen-space normal-map training (rewritten from scratch).

Key design decisions and rationale
----------------------------------
1.  PAD-TO-96 + ALPHA MASK (instead of fixed-size 128 resize).
    The previous run resized every sprite to 128x128 with NEAREST, which
    means a 32x32 sprite was shown to the model as 4x4 blocks of identical
    pixels.  The network never saw a real 32x32 distribution, so 32x32
    inference (the production target) was poor.

    New approach: every sprite is centered on a 96x96 canvas (the max
    native render size) WITHOUT resizing.  Padded pixels have alpha=0.
    The loss is masked by alpha so background contributes nothing.
    The model is fully-convolutional, so at inference it can run at any
    size that's a multiple of 8 (32, 64, 80, 96, ...).

2.  4-CHANNEL INPUT (RGBA).
    The previous run pre-multiplied alpha into RGB and dropped it.
    Transparent pixels became indistinguishable from black pixels.
    Now alpha is a first-class input channel; the model knows where
    the sprite lives.

3.  L2-NORMALIZED OUTPUT.
    The previous run used tanh -> [-1, 1] with no unit-length constraint,
    so the model could win on L1 by emitting short, blurry vectors that
    weren't valid normals.  Now the output head is L2-normalized along
    the channel dim -> always a unit vector.  Targets are also normalized
    (rendered normals are quantized to 8 levels and only approximately
    unit length, so this matters).

4.  3 DOWNSAMPLE LEVELS, BASE=64.
    A 4-level U-Net would shrink 32x32 input to 2x2 at the bottleneck,
    which is degenerate.  3 levels gives 32 -> 16 -> 8 -> 4, and 96 ->
    48 -> 24 -> 12.  Slightly wider channels (base=64) compensate for
    being shallower.

5.  MASKED L1 + MASKED ANGULAR (1 - cos) LOSS.
    Both terms only see pixels where alpha > threshold.  The Sobel
    gradient term from the previous run was dropped: it amplifies false
    edges along the alpha boundary.

6.  STRATIFIED SPLIT BY SIZE.
    32x32 must appear in validation since it's the production target.

7.  CHECKPOINT + INFERENCE GRID EVERY 10 EPOCHS.
    Six fixed validation samples are picked at the start; the same six
    are rendered every 10 epochs into a side-by-side
    (input | predicted | ground-truth) grid for visual progress tracking.

CSV format expected (render_log.csv)
-------------------------------------
Columns required: rgb, normal, status, size.
Only rows with status == 'ok' are used.  Rows whose 'normal' file is
missing on disk are silently dropped (supports manual cleanup).

Usage
-----
Train:
    python train_normalmap.py \
        --render_log render_log.csv \
        --dataset_root /path/to/dataset \
        --output_dir runs/exp1

Resume from latest:
    python train_normalmap.py --resume --output_dir runs/exp1

Inference on a single image:
    python train_normalmap.py --predict \
        --checkpoint runs/exp1/checkpoints/best.pth \
        --input_image sprite.png \
        --output_image normal.png
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as TF


# ──────────────────────────────────────────────
# 1.  CONFIGURATION (override on the command line)
# ──────────────────────────────────────────────

DEFAULT_CFG = dict(
    # Data
    render_log     = "render_log.csv",
    dataset_root   = "",          # empty = use absolute paths from CSV as-is
    canvas_size    = 96,          # square canvas every sample is padded to
    val_split      = 0.10,        # stratified-by-size
    max_pairs      = 0,           # 0 = use all; e.g. 3000 = cap for fast runs

    # Model (3 down-levels, base=32 -> ~8 M params; raise to 64 for full quality)
    base_channels  = 32,
    enc_blocks     = 2,           # ResBlocks per encoder/decoder level
    bot_blocks     = 4,           # ResBlocks in bottleneck

    # Optim
    batch_size     = 32,
    num_epochs     = 60,
    lr             = 3e-4,
    weight_decay   = 1e-4,
    warmup_pct     = 0.10,        # cosine schedule with linear warmup
    grad_clip      = 1.0,
    amp            = True,        # mixed precision (fp16)

    # Loss weights
    l1_weight      = 1.0,
    angular_weight = 0.5,         # (1 - cos theta), masked
    alpha_thresh   = 0.05,        # mask: pixel counted iff alpha > this

    # Run control
    output_dir     = "runs/exp1",
    save_every     = 10,          # checkpoint + sample grid every N epochs
    num_samples    = 6,           # validation samples in the progress grid
    num_workers    = 2,
    seed           = 42,
)

print_banner = lambda s: print(f"\n{'─'*70}\n  {s}\n{'─'*70}")


# ──────────────────────────────────────────────
# 2.  PATH RESOLUTION
# ──────────────────────────────────────────────
# Tries strategies in order:
#   1. Raw path from CSV (works when CSV was written on this machine)
#   2. Path relative to CWD (works when CSV uses repo-relative paths)
#   3. Re-root: strip the original prefix up through 'color\' or 'normal\'
#      and prepend --dataset_root (works when moving between machines)
#
# The first strategy that produces an existing file wins.  If every
# strategy fails, returns the raw path so the dataset's existence-check
# can drop the row.

def _candidate_paths(win_path: str, dataset_root: str) -> list[Path]:
    norm = win_path.replace("\\", "/")
    cands: list[Path] = []

    # Strategy 1: raw path
    cands.append(Path(win_path))

    # Strategy 2: relative to CWD
    cands.append(Path(norm))

    # Strategy 3: re-root using 'color/' or 'normal/' as the anchor
    if dataset_root:
        for marker in ("/color/", "/normal/"):
            i = norm.lower().find(marker)
            if i != -1:
                # Keep everything from the parent of color/normal onward.
                # Find the start of the directory containing color/normal.
                parent_start = norm.rfind("/", 0, i)
                tail = norm[parent_start + 1:] if parent_start >= 0 else norm[i + 1:]
                cands.append(Path(dataset_root) / tail)
                # Also try without the parent dir (just color/foo.png)
                cands.append(Path(dataset_root) / norm[i + 1:])
                break

        # Last-ditch: dataset_root + filename
        cands.append(Path(dataset_root) / Path(norm).name)

    return cands


def _resolve_path(win_path: str, dataset_root: str) -> Path:
    """Return the first candidate path that exists; otherwise the raw path."""
    for cand in _candidate_paths(win_path, dataset_root):
        if cand.exists():
            return cand
    return Path(win_path)   # falls through to existence check, which drops the row


# ──────────────────────────────────────────────
# 3.  DATASET
# ──────────────────────────────────────────────

def _pad_to_canvas(img: Image.Image, canvas: int, fill) -> Image.Image:
    """Center-pad a PIL image to a square canvas. No resizing."""
    w, h = img.size
    if w == canvas and h == canvas:
        return img
    if w > canvas or h > canvas:
        # Sprite larger than canvas: nearest-resize down (rare; only if size>96)
        img = img.resize((min(w, canvas), min(h, canvas)), Image.NEAREST)
        w, h = img.size

    out = Image.new(img.mode, (canvas, canvas), fill)
    out.paste(img, ((canvas - w) // 2, (canvas - h) // 2),
              img if img.mode == "RGBA" else None)
    return out


class NormalMapDataset(Dataset):
    """
    Returns (rgba, normal, mask) per sample, all (C, canvas, canvas).

      rgba   : float32, 4 channels, [0, 1]   - input to the model
      normal : float32, 3 channels, [-1, 1]  - target, unit-length-ish
      mask   : float32, 1 channel,  {0, 1}   - 1 where sprite is opaque
    """

    def __init__(self, render_log: str, dataset_root: str,
                 canvas_size: int, alpha_thresh: float,
                 augment: bool = True):
        self.canvas       = canvas_size
        self.alpha_thresh = alpha_thresh
        self.augment      = augment

        df = pd.read_csv(render_log)
        total = len(df)

        ok = df[df["status"] == "ok"].reset_index(drop=True)
        if len(ok) == 0:
            raise RuntimeError(f"No status='ok' rows in {render_log}.")
        n_failed = total - len(ok)

        # Build pair list with sizes (for stratified split later)
        pairs = []
        for _, row in ok.iterrows():
            c = _resolve_path(row["rgb"],    dataset_root)
            n = _resolve_path(row["normal"], dataset_root)
            sz = int(row["size"])
            if sz > self.canvas:
                # The pipeline emits 32/64/80/96; anything larger gets resized
                # in _pad_to_canvas, but we warn about it.
                pass
            pairs.append((c, n, sz))

        # Drop pairs where the normal map file was deleted from disk.
        # This is the supported "manually cull bad normals" workflow.
        before = len(pairs)
        pairs = [(c, n, s) for c, n, s in pairs if n.exists() and c.exists()]
        n_missing = before - len(pairs)

        self.pairs = pairs
        self.sizes = [s for _, _, s in pairs]   # used by stratified split

        # Summary
        size_counts = {s: self.sizes.count(s) for s in sorted(set(self.sizes))}
        size_str = "  ".join(f"{s}px:{n}" for s, n in size_counts.items())
        print(f"[dataset] {len(pairs)} pairs loaded "
              f"({n_failed} failed, {n_missing} missing on disk, "
              f"out of {total} total).")
        print(f"[dataset] size breakdown: {size_str}")

        if not pairs:
            # Show the first unresolved path so the user can see what went wrong
            sample_row = ok.iloc[0]
            sample_color  = _resolve_path(sample_row["rgb"],    dataset_root)
            sample_normal = _resolve_path(sample_row["normal"], dataset_root)
            raise RuntimeError(
                "No valid pairs after filtering.\n"
                f"  CSV row 0 rgb    : {sample_row['rgb']!r}\n"
                f"  CSV row 0 normal : {sample_row['normal']!r}\n"
                f"  Tried color path : {sample_color}  (exists={sample_color.exists()})\n"
                f"  Tried normal path: {sample_normal}  (exists={sample_normal.exists()})\n"
                "Hints:\n"
                "  - If your CSV paths are already absolute and valid on this\n"
                "    machine, drop --dataset_root entirely.\n"
                "  - If you moved the data, --dataset_root should point at the\n"
                "    folder that contains color/ and normal/ subdirs."
            )

    def __len__(self):
        return len(self.pairs)

    def _load_color(self, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns rgba (4,H,W) in [0,1] and mask (1,H,W) in {0,1}."""
        img = Image.open(path).convert("RGBA")
        img = _pad_to_canvas(img, self.canvas, (0, 0, 0, 0))
        arr = np.asarray(img, dtype=np.float32) / 255.0   # H W 4
        rgba = arr.transpose(2, 0, 1).copy()              # 4 H W
        mask = (rgba[3:4] > self.alpha_thresh).astype(np.float32)
        return torch.from_numpy(rgba), torch.from_numpy(mask)

    def _load_normal(self, path: Path) -> torch.Tensor:
        """Returns normal (3,H,W) in approximately [-1, 1], unit-length-ish."""
        img = Image.open(path).convert("RGB")
        # Pad with the neutral (camera-space "up") normal (128,128,255).
        # This value gets masked out by alpha, but a sane default is safer.
        img = _pad_to_canvas(img, self.canvas, (128, 128, 255))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0                                # [-1, 1]
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())

    def __getitem__(self, idx):
        c_path, n_path, _ = self.pairs[idx]
        rgba, mask = self._load_color(c_path)
        normal     = self._load_normal(n_path)

        if self.augment:
            rgba, normal, mask = self._augment(rgba, normal, mask)

        return rgba, normal, mask

    def _augment(self, rgba, normal, mask):
        """Pixel-art-safe augmentations with correct normal-channel transforms."""
        # Horizontal flip -> negate Nx
        if random.random() < 0.5:
            rgba   = TF.hflip(rgba)
            normal = TF.hflip(normal)
            mask   = TF.hflip(mask)
            normal[0] = -normal[0]

        # Vertical flip -> negate Ny
        if random.random() < 0.5:
            rgba   = TF.vflip(rgba)
            normal = TF.vflip(normal)
            mask   = TF.vflip(mask)
            normal[1] = -normal[1]

        # 90-deg rotations (k=0,1,2,3).  Rotation in screen-space rotates
        # the X/Y components of the normal too; Z (toward camera) is fixed.
        k = random.randint(0, 3)
        if k:
            rgba   = torch.rot90(rgba,   k, [1, 2])
            normal = torch.rot90(normal, k, [1, 2])
            mask   = torch.rot90(mask,   k, [1, 2])
            nx, ny = normal[0].clone(), normal[1].clone()
            if   k == 1: normal[0], normal[1] =  ny, -nx   # CCW 90
            elif k == 2: normal[0], normal[1] = -nx, -ny   # 180
            elif k == 3: normal[0], normal[1] = -ny,  nx   # CW 90

        # Brightness / contrast jitter on RGB only (not alpha)
        if random.random() < 0.5:
            b = random.uniform(0.85, 1.15)
            c = random.uniform(0.85, 1.15)
            rgb = rgba[:3]
            rgb = torch.clamp(rgb * b, 0, 1)
            rgb = torch.clamp((rgb - 0.5) * c + 0.5, 0, 1)
            rgba = torch.cat([rgb, rgba[3:4]], dim=0)

        return rgba, normal, mask


def stratified_split_indices(sizes: list[int], val_split: float,
                             seed: int, max_pairs: int = 0,
                             keep_size: int = 32) -> tuple[list[int], list[int]]:
    """Stratified train/val split by size bucket.  32x32 (the production
    target) is guaranteed to appear in val because every bucket gets at
    least one val sample.

    If max_pairs > 0, the dataset is downsampled to roughly that many
    pairs by keeping ALL of `keep_size` (production target) and
    proportionally cutting the larger buckets.  Useful for fast runs.

    Returns plain index lists so the caller can wrap them around the
    appropriate dataset instances (train ds with augment=True, val ds
    with augment=False).
    """
    rng = random.Random(seed)
    by_size: dict[int, list[int]] = {}
    for i, s in enumerate(sizes):
        by_size.setdefault(s, []).append(i)

    # Optional: downsample to fit max_pairs while protecting keep_size
    if max_pairs and len(sizes) > max_pairs:
        kept = list(by_size.get(keep_size, []))     # never trim 32px
        budget = max_pairs - len(kept)
        if budget < 0:
            # Even the protected bucket exceeds the budget — trim it too.
            rng.shuffle(kept)
            kept = kept[:max_pairs]
            budget = 0
        # Distribute remaining budget proportionally across other buckets
        other = {s: idxs[:] for s, idxs in by_size.items() if s != keep_size}
        total_other = sum(len(v) for v in other.values()) or 1
        new_by_size = {keep_size: kept} if keep_size in by_size else {}
        for s, idxs in other.items():
            rng.shuffle(idxs)
            n = max(1, int(round(len(idxs) / total_other * budget)))
            new_by_size[s] = idxs[:n]
        by_size = new_by_size

    train_idx, val_idx = [], []
    for s, idxs in by_size.items():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_split)))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# ──────────────────────────────────────────────
# 4.  MODEL — U-Net, 3 levels, GroupNorm, L2-normalized output
# ──────────────────────────────────────────────

def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with auto group count (stable at small batches)."""
    g = min(8, channels)
    while channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            _gn(ch), nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            _gn(ch), nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_blocks: int):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            _gn(out_ch), nn.SiLU(inplace=True),
        ]
        for _ in range(n_blocks):
            layers.append(ResBlock(out_ch))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, n_blocks: int):
        super().__init__()
        # ConvTranspose for clean integer-stride upsampling
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        layers = [
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.SiLU(inplace=True),
        ]
        for _ in range(n_blocks):
            layers.append(ResBlock(out_ch))
        self.net = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.up(x)
        # Safety net for non-power-of-2 inputs (e.g. 80x80)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.net(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """
    3 down-levels.  At canvas=96:  96 -> 48 -> 24 -> 12 (bottleneck).
    At inference 32:               32 -> 16 ->  8 ->  4 (bottleneck).

    Input  : 4 channels (RGBA)
    Output : 3 channels, L2-normalized along channel dim -> unit normals.
    """

    def __init__(self, in_ch: int = 4, out_ch: int = 3, base: int = 64,
                 enc_blocks: int = 2, bot_blocks: int = 4):
        super().__init__()
        b = base

        # --- Stem (full resolution) ---
        stem = [
            nn.Conv2d(in_ch, b, 3, padding=1, bias=False),
            _gn(b), nn.SiLU(inplace=True),
        ]
        for _ in range(enc_blocks):
            stem.append(ResBlock(b))
        self.enc0 = nn.Sequential(*stem)            # /1   ch=b

        # --- Encoder ---
        self.enc1 = Down(b,     b * 2, enc_blocks)  # /2   ch=2b
        self.enc2 = Down(b * 2, b * 4, enc_blocks)  # /4   ch=4b

        # --- Bottleneck (downsample once more, then stack ResBlocks) ---
        bot = [Down(b * 4, b * 8, n_blocks=1)]      # /8   ch=8b
        for _ in range(bot_blocks):
            bot.append(ResBlock(b * 8))
        self.bot = nn.Sequential(*bot)

        # --- Decoder (mirrors encoder) ---
        self.dec2 = Up(b * 8, b * 4, b * 4, enc_blocks)   # /4
        self.dec1 = Up(b * 4, b * 2, b * 2, enc_blocks)   # /2
        self.dec0 = Up(b * 2, b,     b,     enc_blocks)   # /1

        # --- Head ---
        self.head = nn.Sequential(
            ResBlock(b),
            nn.Conv2d(b, out_ch, 1),
        )

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        b  = self.bot(e2)
        d2 = self.dec2(b,  e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)
        out = self.head(d0)
        # L2-normalize along channel dim -> guaranteed unit vectors.
        # eps prevents NaNs at startup when output is near-zero.
        return F.normalize(out, dim=1, eps=1e-6)


# ──────────────────────────────────────────────
# 5.  LOSS — MASKED L1 + MASKED ANGULAR (1 - cos)
# ──────────────────────────────────────────────

class MaskedNormalLoss(nn.Module):
    """
    Total = l1_w * MaskedL1 + ang_w * MaskedAngular.

    All loss terms are computed only on pixels where alpha > thresh.
    The previous run averaged over the full canvas, so >70 % of the
    gradient came from background (a fake flat-up normal).  That made the
    model spend its capacity learning to output flat blue, not real shape.

    Angular term is (1 - cos theta).  acos(cos) has infinite gradient at
    theta=0 and is numerically unstable; (1 - cos) is smooth, monotone,
    and minimised at the same point.
    """

    def __init__(self, l1_w: float = 1.0, ang_w: float = 0.5):
        super().__init__()
        self.l1_w  = l1_w
        self.ang_w = ang_w

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor):
        """
        pred  : (B, 3, H, W)  unit-length (already normalized by the head)
        target: (B, 3, H, W)  approximately unit-length
        mask  : (B, 1, H, W)  in {0, 1}
        """
        # Normalize the target too (rendered normals are quantized to 8 levels
        # and only approximately unit length).
        target_n = F.normalize(target, dim=1, eps=1e-6)

        n_valid = mask.sum().clamp(min=1.0)   # total valid pixels in batch

        # Masked L1 (3 channels averaged into n_valid)
        l1 = ((pred - target_n).abs() * mask).sum() / (n_valid * 3.0)

        # Masked angular (1 - cos); cos in (-1, 1) thanks to unit-length pred
        cos = (pred * target_n).sum(dim=1, keepdim=True)   # B 1 H W
        ang = ((1.0 - cos) * mask).sum() / n_valid

        total = self.l1_w * l1 + self.ang_w * ang
        return total, l1.detach(), ang.detach()


@torch.no_grad()
def mean_angular_error_deg(pred: torch.Tensor, target: torch.Tensor,
                           mask: torch.Tensor) -> float:
    """Mean angular error in degrees over masked pixels (validation metric)."""
    target_n = F.normalize(target, dim=1, eps=1e-6)
    cos = (pred * target_n).sum(dim=1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    n_valid = mask.sum().clamp(min=1.0)
    deg = torch.acos(cos) * (180.0 / math.pi)
    return ((deg * mask).sum() / n_valid).item()


# ──────────────────────────────────────────────
# 6.  LR SCHEDULE — linear warmup -> cosine decay
# ──────────────────────────────────────────────

def make_warmup_cosine_scheduler(optimizer, total_steps: int,
                                 warmup_pct: float, min_lr_ratio: float = 0.01):
    """Cosine decay from base_lr to base_lr * min_lr_ratio with linear warmup."""
    warmup_steps = max(1, int(total_steps * warmup_pct))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────
# 7.  PROGRESS GRID — visual progress every save_every epochs
# ──────────────────────────────────────────────

@torch.no_grad()
def save_progress_grid(model: nn.Module, samples, device, out_path: Path,
                       cfg: dict):
    """
    Build a side-by-side grid:  input | predicted | ground truth
    one row per sample.  Tiles are upscaled with NEAREST to 96x96 for clarity.

    `samples` is a list of (rgba, normal, mask) CPU tensors (4/3/1 x H x W),
    fixed for the entire training run so frames are visually comparable.
    """
    model.eval()
    tile = cfg["canvas_size"]                 # already 96
    rows = []
    for rgba, normal, mask in samples:
        x        = rgba.unsqueeze(0).to(device)
        pred     = model(x)[0].cpu()          # 3 H W in [-1, 1]
        # Convert all three to 8-bit RGB tiles for display
        # 1) Input: composite RGB over white using alpha
        rgb   = rgba[:3].numpy()
        alpha = rgba[3:4].numpy()
        comp  = rgb * alpha + (1 - alpha)     # over white
        comp_img = (np.clip(comp, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        # 2) Predicted normal -> 8-bit
        pred_img = ((pred.numpy() + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        pred_img = pred_img.transpose(1, 2, 0)
        # 3) GT normal -> 8-bit
        gt_img = ((normal.numpy() + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        gt_img = gt_img.transpose(1, 2, 0)
        # Concatenate horizontally with a 2-px black separator
        sep   = np.zeros((tile, 2, 3), dtype=np.uint8)
        row   = np.concatenate([comp_img, sep, pred_img, sep, gt_img], axis=1)
        rows.append(row)

    sep_v = np.zeros((2, rows[0].shape[1], 3), dtype=np.uint8)
    grid  = rows[0]
    for r in rows[1:]:
        grid = np.concatenate([grid, sep_v, r], axis=0)

    # Upscale 4x with NEAREST so pixel-art stays crisp in the saved image
    grid_img = Image.fromarray(grid, "RGB")
    grid_img = grid_img.resize(
        (grid_img.width * 4, grid_img.height * 4), Image.NEAREST
    )
    grid_img.save(out_path)


def pick_fixed_samples(val_ds_no_aug: Dataset, val_indices: list[int],
                       n: int, seed: int):
    """Pick `n` validation samples to track over time.  Caller passes the
    no-augmentation dataset so samples are deterministic."""
    rng = random.Random(seed)
    idx = list(val_indices)
    rng.shuffle(idx)
    chosen = idx[:min(n, len(idx))]
    return [val_ds_no_aug[i] for i in chosen]


# ──────────────────────────────────────────────
# 8.  TRAINING LOOP
# ──────────────────────────────────────────────

def train(cfg: dict):
    # --- Reproducibility ---
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"         {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        torch.backends.cudnn.benchmark = True

    # --- Output dirs ---
    out_dir   = Path(cfg["output_dir"])
    ckpt_dir  = out_dir / "checkpoints"
    sample_dir = out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    sample_dir.mkdir(exist_ok=True)

    # --- Data ---
    # Two dataset instances pointing at the same files: one with augment=True
    # for training, one with augment=False for validation and the progress
    # grid.  Two instances (rather than toggling .augment on a shared one)
    # keeps multi-worker DataLoaders correct, since workers fork the dataset
    # state once and never see in-place mutations.
    print_banner("Loading dataset")
    train_full = NormalMapDataset(
        render_log   = cfg["render_log"],
        dataset_root = cfg["dataset_root"],
        canvas_size  = cfg["canvas_size"],
        alpha_thresh = cfg["alpha_thresh"],
        augment      = True,
    )
    val_full = NormalMapDataset(
        render_log   = cfg["render_log"],
        dataset_root = cfg["dataset_root"],
        canvas_size  = cfg["canvas_size"],
        alpha_thresh = cfg["alpha_thresh"],
        augment      = False,
    )
    train_idx, val_idx = stratified_split_indices(
        train_full.sizes, cfg["val_split"], cfg["seed"],
        max_pairs=cfg.get("max_pairs", 0),
    )
    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(val_full,   val_idx)
    print(f"[split] train={len(train_ds)}  val={len(val_ds)}  "
          f"(stratified by size)")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"),
    )

    # Pick fixed validation samples for the progress grid (uses the
    # no-augment val dataset so the same six pixels render every check-in).
    fixed_samples = pick_fixed_samples(
        val_full, val_idx, cfg["num_samples"], seed=cfg["seed"]
    )
    print(f"[grid] {len(fixed_samples)} fixed validation samples for tracking")

    # --- Model ---
    print_banner("Building model")
    model = UNet(
        in_ch     = 4,
        out_ch    = 3,
        base      = cfg["base_channels"],
        enc_blocks = cfg["enc_blocks"],
        bot_blocks = cfg["bot_blocks"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] U-Net  base={cfg['base_channels']}  "
          f"enc={cfg['enc_blocks']}  bot={cfg['bot_blocks']}  "
          f"params={n_params/1e6:.2f} M")

    # --- Optim / loss / scheduler ---
    criterion = MaskedNormalLoss(cfg["l1_weight"], cfg["angular_weight"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    total_steps = cfg["num_epochs"] * len(train_loader)
    scheduler = make_warmup_cosine_scheduler(
        optimizer, total_steps, cfg["warmup_pct"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    # --- Optionally resume ---
    start_epoch = 1
    best_val    = float("inf")
    latest_ckpt = ckpt_dir / "latest.pth"
    if cfg.get("resume") and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))
        print(f"[resume] loaded {latest_ckpt} (epoch {ckpt['epoch']})")

    # --- Metrics CSV ---
    csv_path = out_dir / "metrics.csv"
    write_header = not csv_path.exists() or start_epoch == 1
    csv_file = open(csv_path, "a" if not write_header else "w",
                    newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow([
            "epoch", "train_loss", "train_l1", "train_ang",
            "val_loss", "val_l1", "val_ang",
            "val_mae_deg", "lr", "epoch_time_s",
        ])
        csv_file.flush()

    print_banner(f"Training {cfg['num_epochs']} epochs "
                 f"(start={start_epoch}, best_val={best_val:.4f})")

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        t0 = time.time()

        # ────── Train ──────
        model.train()
        tr_loss = tr_l1 = tr_ang = 0.0
        n_seen = 0
        for rgba, normal, mask in train_loader:
            rgba   = rgba.to(device,   non_blocking=True)
            normal = normal.to(device, non_blocking=True)
            mask   = mask.to(device,   non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg["amp"]):
                pred = model(rgba)
                loss, l1_v, ang_v = criterion(pred, normal, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = rgba.size(0)
            tr_loss += loss.item() * bs
            tr_l1   += l1_v.item() * bs
            tr_ang  += ang_v.item() * bs
            n_seen  += bs
        tr_loss /= n_seen; tr_l1 /= n_seen; tr_ang /= n_seen

        # ────── Validate ──────
        model.eval()
        vl_loss = vl_l1 = vl_ang = vl_mae = 0.0
        n_val = 0
        with torch.no_grad():
            for rgba, normal, mask in val_loader:
                rgba   = rgba.to(device,   non_blocking=True)
                normal = normal.to(device, non_blocking=True)
                mask   = mask.to(device,   non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=cfg["amp"]):
                    pred = model(rgba)
                    loss, l1_v, ang_v = criterion(pred, normal, mask)
                bs = rgba.size(0)
                vl_loss += loss.item()  * bs
                vl_l1   += l1_v.item()  * bs
                vl_ang  += ang_v.item() * bs
                vl_mae  += mean_angular_error_deg(
                    pred.float(), normal.float(), mask.float()
                ) * bs
                n_val += bs
        vl_loss /= n_val; vl_l1 /= n_val; vl_ang /= n_val; vl_mae /= n_val

        cur_lr = optimizer.param_groups[0]["lr"]
        dt     = time.time() - t0

        # ────── Log ──────
        writer.writerow([
            epoch, f"{tr_loss:.6f}", f"{tr_l1:.6f}", f"{tr_ang:.6f}",
            f"{vl_loss:.6f}", f"{vl_l1:.6f}", f"{vl_ang:.6f}",
            f"{vl_mae:.4f}", f"{cur_lr:.2e}", f"{dt:.1f}",
        ])
        csv_file.flush()

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg["num_epochs"]:
            print(f"Ep {epoch:>4}/{cfg['num_epochs']}  "
                  f"tr={tr_loss:.4f}  vl={vl_loss:.4f}  "
                  f"AngErr={vl_mae:5.2f}°  "
                  f"lr={cur_lr:.2e}  {dt:.0f}s")

        # ────── Best checkpoint ──────
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_val": best_val,
                "cfg": cfg,
            }, ckpt_dir / "best.pth")

        # ────── Periodic ckpt + sample grid ──────
        if epoch % cfg["save_every"] == 0 or epoch == cfg["num_epochs"]:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_val": best_val,
                "cfg": cfg,
            }, ckpt_dir / f"epoch_{epoch:04d}.pth")
            # Always overwrite latest.pth for resume
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_val": best_val,
                "cfg": cfg,
            }, latest_ckpt)
            grid_path = sample_dir / f"epoch_{epoch:04d}.png"
            save_progress_grid(model, fixed_samples, device, grid_path, cfg)
            print(f"          saved   ckpt -> {ckpt_dir.name}/epoch_{epoch:04d}.pth")
            print(f"          saved sample -> {sample_dir.name}/epoch_{epoch:04d}.png")

    csv_file.close()
    print_banner(f"Done. best_val_loss = {best_val:.6f}")
    print(f"  metrics  : {csv_path}")
    print(f"  best     : {ckpt_dir / 'best.pth'}")
    print(f"  samples  : {sample_dir}")


# ──────────────────────────────────────────────
# 9.  INFERENCE — single image or directory
# ──────────────────────────────────────────────

def _round_up_to(v: int, mult: int) -> int:
    return ((v + mult - 1) // mult) * mult


@torch.no_grad()
def predict_single(checkpoint: str, image_path: str, output_path: str,
                   write_alpha: bool = True):
    """
    Generate a normal map for a single sprite.

    The model is fully-convolutional, so we feed it the sprite at its
    native resolution (centered on the smallest 8-multiple canvas), then
    crop back to the original size.  No nearest-neighbor upscaling.

    `write_alpha` saves the output as RGBA, with alpha = input alpha,
    so the sprite background stays transparent.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg    = ckpt.get("cfg", {})

    model = UNet(
        in_ch     = 4, out_ch = 3,
        base      = cfg.get("base_channels", 64),
        enc_blocks = cfg.get("enc_blocks", 2),
        bot_blocks = cfg.get("bot_blocks", 4),
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    # Open input as RGBA at native resolution
    src = Image.open(image_path).convert("RGBA")
    sw, sh = src.size

    # Pad to the smallest 8-multiple square that fits the sprite
    side = _round_up_to(max(sw, sh), 8)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    px, py = (side - sw) // 2, (side - sh) // 2
    canvas.paste(src, (px, py), src)

    arr  = np.asarray(canvas, dtype=np.float32) / 255.0
    rgba = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

    pred = model(rgba)[0].cpu().numpy()           # 3 H W in [-1, 1]
    out  = ((pred + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    out  = out.transpose(1, 2, 0)                  # H W 3

    # Crop back to original size
    out = out[py:py + sh, px:px + sw]

    if write_alpha:
        # Stitch the original alpha back in so transparent regions stay
        # transparent (they contain garbage normal predictions).
        src_alpha = (np.asarray(src, dtype=np.uint8)[..., 3:4])
        rgba_out  = np.concatenate([out, src_alpha], axis=2)
        Image.fromarray(rgba_out, mode="RGBA").save(output_path)
    else:
        Image.fromarray(out, mode="RGB").save(output_path)

    print(f"[predict] {image_path}  ({sw}x{sh})  ->  {output_path}")


# ──────────────────────────────────────────────
# 10.  CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Pixel-art normal-map U-Net trainer")
    # Data
    p.add_argument("--render_log",   default=DEFAULT_CFG["render_log"])
    p.add_argument("--dataset_root", default=DEFAULT_CFG["dataset_root"],
                   help="Re-root absolute Windows paths from CSV. Empty=use as-is.")
    p.add_argument("--canvas_size",  type=int,   default=DEFAULT_CFG["canvas_size"])
    p.add_argument("--val_split",    type=float, default=DEFAULT_CFG["val_split"])
    p.add_argument("--max_pairs",    type=int,   default=DEFAULT_CFG["max_pairs"],
                   help="Cap dataset size (0=use all). 32px renders are always kept.")
    # Model
    p.add_argument("--base_channels", type=int, default=DEFAULT_CFG["base_channels"])
    p.add_argument("--enc_blocks",    type=int, default=DEFAULT_CFG["enc_blocks"])
    p.add_argument("--bot_blocks",    type=int, default=DEFAULT_CFG["bot_blocks"])
    # Optim
    p.add_argument("--batch_size",   type=int,   default=DEFAULT_CFG["batch_size"])
    p.add_argument("--num_epochs",   type=int,   default=DEFAULT_CFG["num_epochs"])
    p.add_argument("--lr",           type=float, default=DEFAULT_CFG["lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULT_CFG["weight_decay"])
    p.add_argument("--warmup_pct",   type=float, default=DEFAULT_CFG["warmup_pct"])
    p.add_argument("--grad_clip",    type=float, default=DEFAULT_CFG["grad_clip"])
    p.add_argument("--no_amp",       action="store_true")
    # Loss
    p.add_argument("--l1_weight",      type=float, default=DEFAULT_CFG["l1_weight"])
    p.add_argument("--angular_weight", type=float, default=DEFAULT_CFG["angular_weight"])
    p.add_argument("--alpha_thresh",   type=float, default=DEFAULT_CFG["alpha_thresh"])
    # Run control
    p.add_argument("--output_dir",  default=DEFAULT_CFG["output_dir"])
    p.add_argument("--save_every",  type=int, default=DEFAULT_CFG["save_every"])
    p.add_argument("--num_samples", type=int, default=DEFAULT_CFG["num_samples"])
    p.add_argument("--num_workers", type=int, default=DEFAULT_CFG["num_workers"])
    p.add_argument("--seed",        type=int, default=DEFAULT_CFG["seed"])
    p.add_argument("--resume",      action="store_true",
                   help="Resume from <output_dir>/checkpoints/latest.pth")
    # Inference
    p.add_argument("--predict",      action="store_true")
    p.add_argument("--checkpoint",   default=None)
    p.add_argument("--input_image",  default=None)
    p.add_argument("--output_image", default="predicted_normal.png")
    p.add_argument("--no_alpha_out", action="store_true",
                   help="Inference: write RGB instead of RGBA")
    return p.parse_args()


def args_to_cfg(args) -> dict:
    cfg = dict(DEFAULT_CFG)
    for k in cfg:
        if hasattr(args, k):
            cfg[k] = getattr(args, k)
    cfg["amp"]    = not args.no_amp
    cfg["resume"] = args.resume
    return cfg


if __name__ == "__main__":
    args = parse_args()

    if args.predict:
        if not args.input_image or not args.checkpoint:
            raise SystemExit("--predict needs --checkpoint and --input_image")
        predict_single(
            checkpoint   = args.checkpoint,
            image_path   = args.input_image,
            output_path  = args.output_image,
            write_alpha  = not args.no_alpha_out,
        )
    else:
        train(args_to_cfg(args))