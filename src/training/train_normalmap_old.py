"""
train_normalmap.py
==================
Pixel-art normal-map generator – U-Net training script.

Design choices
--------------
* Architecture : U-Net with residual blocks (no BatchNorm → GroupNorm to keep
                 stable stats at small batch sizes and tiny spatial dims).
* Loss         : L1 + Angular (cosine) loss.  L1 keeps edges sharp; angular
                 loss understands that XYZ channels are a unit-vector field.
* Optimizer    : AdamW + OneCycleLR – fast convergence, reliable on small datasets.
* AMP          : torch.autocast (fp16) → fits comfortably in 8 GB VRAM.
* Augmentation : horizontal/vertical flip, 90° rotations, small colour jitter on
                 input only.  No spatial warps – pixel art must stay pixel-perfect.
* Metrics      : MAE, PSNR, SSIM, Mean Angular Error saved to CSV every epoch.

Dataset format (render_log.csv)
--------------------------------
The CSV produced by the Blender render pipeline has these columns:
    model, stem, suffix, size, elevation, rotation, rgb, normal, status, error

Only rows where status == 'ok' are used.
The 'rgb' and 'normal' columns contain absolute Windows paths
(e.g. D:\\...\\dataset\\color\\foo.png).  At runtime the script re-roots them
under --dataset_root by extracting the relative tail starting from
'dataset/color/' or 'dataset/normal/'.  If --dataset_root is not given,
the script tries to use the paths as-is (useful when running on the same
Windows machine that did the rendering).
"""

import os
import csv
import math
import time
import random
import argparse
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import functional as TF

try:
    from pytorch_msssim import ssim as pt_ssim   # pip install pytorch-msssim
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False
    print("[warn] pytorch-msssim not found – SSIM will be computed via numpy fallback.")

# ──────────────────────────────────────────────
# 1.  CONFIGURATION  (edit here or pass --args)
# ──────────────────────────────────────────────

DEFAULT_CFG = dict(
    render_log      = "render_log.csv",   # path to the CSV manifest
    dataset_root    = "",                 # local root that contains color/ and normal/
                                          # leave empty to use absolute paths from CSV
    output_dir      = "runs/exp7",        # checkpoints + metrics logs go here
    image_size      = 128,                # resize everything to this square
    batch_size      = 16,
    num_epochs      = 200,
    lr              = 3e-4,
    weight_decay    = 1e-4,
    val_split       = 0.1,               # 10 % validation
    l1_weight       = 1.0,
    angular_weight  = 0.5,
    num_workers     = 4,
    seed            = 42,
    save_every      = 10,                 # save checkpoint every N epochs
    amp             = True,               # mixed precision (fp16)
    base_channels   = 32,                 # U-Net width – increase if VRAM allows
    num_bot_blocks  = 4,                  # ResBlocks in bottleneck (2=original, 4=recommended)
    enc_blocks      = 2,                  # ResBlocks per encoder/decoder level (1=original, 2=recommended)
    grad_weight     = 1.0,                # raised from 0.5 for sharper edges
)


# ──────────────────────────────────────────────
# 2.  DATASET
# ──────────────────────────────────────────────

def _resolve_path(win_path: str, dataset_root: str) -> Path:
    """
    Convert a Windows absolute path from the CSV to a local filesystem path.

    Strategy
    --------
    1. If dataset_root is given, extract the tail of the Windows path starting
       from 'dataset\\color\\' or 'dataset\\normal\\' (or the forward-slash
       equivalents) and join it under dataset_root.
    2. If no dataset_root, return the path as-is (works when running on the
       same Windows machine that rendered the data).
    """
    if not dataset_root:
        return Path(win_path)

    # Normalise to forward slashes for reliable splitting
    normalised = win_path.replace("\\", "/")
    # Find the sub-path starting from 'dataset/color/' or 'dataset/normal/'
    for marker in ("dataset/color/", "dataset/normal/"):
        idx = normalised.lower().find(marker)
        if idx != -1:
            rel = normalised[idx:]          # e.g. "dataset/color/foo.png"
            return Path(dataset_root) / rel

    # Fallback: take everything after the last occurrence of 'dataset/'
    idx = normalised.lower().rfind("dataset/")
    if idx != -1:
        return Path(dataset_root) / normalised[idx:]

    # Last resort: just use the basename under dataset_root
    return Path(dataset_root) / Path(normalised).name


class NormalMapDataset(Dataset):
    """
    Reads image pairs from a render_log.csv manifest.

    The CSV must have at least these columns:
        rgb     – path to the colour sprite (RGBA PNG)
        normal  – path to the normal map (RGB PNG)
        status  – only rows with status == 'ok' are loaded

    All images are resized to image_size × image_size using nearest-neighbour
    interpolation (correct for pixel art – no blending between pixels).
    """

    def __init__(self, render_log: str, dataset_root: str,
                 image_size: int, augment: bool = True):
        self.image_size   = image_size
        self.augment      = augment
        self.dataset_root = dataset_root

        df = pd.read_csv(render_log)
        total = len(df)

        ok = df[df["status"] == "ok"].reset_index(drop=True)
        failed = total - len(ok)

        # Drop 48x48 renders: upscaling from 48px produces blocky aliased
        # images that add inconsistent signal compared to native 64px+ renders.
        skipped_48 = int((ok["size"] == 48).sum())
        ok = ok[ok["size"] != 48].reset_index(drop=True)

        if len(ok) == 0:
            raise RuntimeError(
                f"No rows with status='ok' (and size != 48) found in {render_log}."
            )

        self.pairs = [
            (
                _resolve_path(row["rgb"],    dataset_root),
                _resolve_path(row["normal"], dataset_root),
            )
            for _, row in ok.iterrows()
        ]

        # Drop pairs where the normal map was manually removed from disk
        before_filter = len(self.pairs)
        self.pairs = [(c, n) for c, n in self.pairs if n.exists()]
        removed = before_filter - len(self.pairs)
        if removed:
            print(f"[dataset] Skipped {removed} pairs: normal map file missing on disk.")

        size_counts = ok["size"].value_counts().sort_index().to_dict()
        size_str = "  ".join(f"{s}px:{n}" for s, n in size_counts.items())
        print(f"[dataset] {len(self.pairs)} valid pairs loaded "
              f"({failed} failed + {skipped_48} at 48px skipped out of {total} total).")
        print(f"[dataset] size breakdown: {size_str}")

        # Sanity-check the first file exists so we fail fast with a clear message
        first_rgb = self.pairs[0][0]
        if not first_rgb.exists():
            raise FileNotFoundError(
                f"First RGB file not found: {first_rgb}\n"
                "Tip: set --dataset_root to the local folder that contains "
                "the 'dataset/color/' and 'dataset/normal/' sub-directories."
            )

    def __len__(self):
        return len(self.pairs)

    def _load_color(self, path: Path) -> torch.Tensor:
        """Load RGBA sprite, pre-multiply alpha, return float32 [0,1] (3, H, W)."""
        img = Image.open(path).convert("RGBA")
        img = img.resize((self.image_size, self.image_size), Image.NEAREST)
        arr   = np.array(img, dtype=np.float32) / 255.0   # H W 4
        alpha = arr[..., 3:4]
        rgb   = arr[..., :3] * alpha                        # pre-multiply
        return torch.from_numpy(rgb.transpose(2, 0, 1))     # 3 H W

    def _load_normal(self, path: Path) -> torch.Tensor:
        """Load RGB normal map, return float32 [-1, 1] (3, H, W)."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.NEAREST)
        arr = np.array(img, dtype=np.float32) / 255.0       # H W 3  [0,1]
        arr = arr * 2.0 - 1.0                                # remap to [-1, 1]
        return torch.from_numpy(arr.transpose(2, 0, 1))      # 3 H W

    def __getitem__(self, idx: int):
        c_path, n_path = self.pairs[idx]
        color  = self._load_color(c_path)
        normal = self._load_normal(n_path)

        if self.augment:
            # Horizontal flip – must negate X component of normal map
            if random.random() > 0.5:
                color  = TF.hflip(color)
                normal = TF.hflip(normal)
                normal[0] = -normal[0]

            # Vertical flip – must negate Y component of normal map
            if random.random() > 0.5:
                color  = TF.vflip(color)
                normal = TF.vflip(normal)
                normal[1] = -normal[1]

            # 90° rotations (pixel-art safe – no interpolation)
            # XY normal channels must be rotated to match spatial rotation:
            # k=1 CCW 90°: X=+oldY  Y=-oldX
            # k=2    180°: X=-oldX  Y=-oldY
            # k=3 CW  90°: X=-oldY  Y=+oldX
            # Z channel (toward camera) is always unchanged.
            k = random.choice([0, 1, 2, 3])
            if k:
                color  = torch.rot90(color,  k, [1, 2])
                normal = torch.rot90(normal, k, [1, 2])
                nx, ny = normal[0].clone(), normal[1].clone()
                if k == 1:
                    normal[0], normal[1] =  ny, -nx
                elif k == 2:
                    normal[0], normal[1] = -nx, -ny
                elif k == 3:
                    normal[0], normal[1] = -ny,  nx

            # Random crop + resize (75-100%% window, nearest-neighbour resize)
            # Forces generalisation across zoom levels without blurring edges.
            if random.random() > 0.5:
                H, W   = color.shape[1], color.shape[2]
                scale  = random.uniform(0.75, 1.0)
                ch, cw = int(H * scale), int(W * scale)
                top    = random.randint(0, H - ch)
                left   = random.randint(0, W - cw)
                color  = color[:,  top:top+ch, left:left+cw]
                normal = normal[:, top:top+ch, left:left+cw]
                color  = F.interpolate(color.unsqueeze(0),  size=(H, W), mode="nearest").squeeze(0)
                normal = F.interpolate(normal.unsqueeze(0), size=(H, W), mode="nearest").squeeze(0)

            # Colour jitter on input sprite only
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                contrast   = random.uniform(0.8, 1.2)
                color = torch.clamp(color * brightness, 0, 1)
                color = torch.clamp((color - 0.5) * contrast + 0.5, 0, 1)

        return color, normal


# ──────────────────────────────────────────────
# 2b.  FINETUNE DATASET  (non-square images, pad → square → augment)
# ──────────────────────────────────────────────

def _pad_image_to_square(img: Image.Image, fill_rgb=(0, 0, 0), fill_a=0) -> Image.Image:
    """
    Pad a PIL image to square by adding transparent/black pixels on the shorter
    axis, distributing evenly on both sides (half top+bottom or half left+right).

    For RGBA images the padding is transparent (fill_a=0).
    For RGB images the padding is black (fill_rgb).
    """
    w, h   = img.size
    side   = max(w, h)
    mode   = img.mode

    if mode == "RGBA":
        canvas = Image.new("RGBA", (side, side), (*fill_rgb, fill_a))
    else:
        canvas = Image.new("RGB",  (side, side), fill_rgb)

    paste_x = (side - w) // 2
    paste_y = (side - h) // 2
    if mode == "RGBA":
        canvas.paste(img, (paste_x, paste_y), img)   # use alpha as mask
    else:
        canvas.paste(img, (paste_x, paste_y))

    return canvas


class FinetuneDataset(Dataset):
    """
    Dataset for high-quality fine-tune pairs that may be non-square.

    Loading pipeline per pair:
      1. Open color (RGBA) and normal (RGB).
      2. Pad both to square with transparent/black pixels (nearest, no blending).
      3. Resize to image_size × image_size with nearest-neighbour.
      4. Apply the same augmentation pipeline as NormalMapDataset.

    The CSV must have at least:
        rgb    – path to the colour sprite (RGBA PNG)
        normal – path to the normal map   (RGB PNG)
        status – only 'ok' rows are used
    No 'size' column is required (non-square images are all accepted).
    """

    def __init__(self, render_log: str, image_size: int, augment: bool = True):
        self.image_size = image_size
        self.augment    = augment

        df    = pd.read_csv(render_log)
        total = len(df)
        ok    = df[df["status"] == "ok"].reset_index(drop=True)
        failed = total - len(ok)

        # Build pairs using raw paths (fine-tune data lives on the same machine)
        self.pairs = [(Path(row["rgb"]), Path(row["normal"])) for _, row in ok.iterrows()]

        # Drop any rows whose files are missing on disk
        before = len(self.pairs)
        self.pairs = [(c, n) for c, n in self.pairs if c.exists() and n.exists()]
        removed = before - len(self.pairs)

        print(f"[finetune] {len(self.pairs)} pairs loaded "
              f"({failed} failed, {removed} missing on disk, out of {total} total).")

        if not self.pairs:
            raise RuntimeError(f"No valid fine-tune pairs found in {render_log}.")

    def __len__(self):
        return len(self.pairs)

    def _load_color(self, path: Path) -> torch.Tensor:
        """Load RGBA, pad to square, resize, pre-multiply alpha → float32 [0,1] (3,H,W)."""
        img  = Image.open(path).convert("RGBA")
        img  = _pad_image_to_square(img, fill_rgb=(0, 0, 0), fill_a=0)
        img  = img.resize((self.image_size, self.image_size), Image.NEAREST)
        arr  = np.array(img, dtype=np.float32) / 255.0
        rgb  = arr[..., :3] * arr[..., 3:4]             # pre-multiply alpha
        return torch.from_numpy(rgb.transpose(2, 0, 1))  # 3 H W

    def _load_normal(self, path: Path) -> torch.Tensor:
        """Load RGB normal map, pad to square (black = Z-up neutral), resize → [-1,1]."""
        img = Image.open(path).convert("RGB")
        # Black padding (0,0,0) maps to normal (-1,-1,-1) after remapping,
        # but since padded regions have alpha=0 in color, they're masked out
        # during training anyway. A neutral Z-up normal (128,128,255) is better.
        w, h   = img.size
        side   = max(w, h)
        canvas = Image.new("RGB", (side, side), (128, 128, 255))  # neutral Z-up
        paste_x = (side - w) // 2
        paste_y = (side - h) // 2
        canvas.paste(img, (paste_x, paste_y))
        img  = canvas.resize((self.image_size, self.image_size), Image.NEAREST)
        arr  = np.array(img, dtype=np.float32) / 255.0
        arr  = arr * 2.0 - 1.0
        return torch.from_numpy(arr.transpose(2, 0, 1))  # 3 H W

    def __getitem__(self, idx: int):
        c_path, n_path = self.pairs[idx]
        color  = self._load_color(c_path)
        normal = self._load_normal(n_path)

        if self.augment:
            # Horizontal flip + negate X
            if random.random() > 0.5:
                color     = TF.hflip(color)
                normal    = TF.hflip(normal)
                normal[0] = -normal[0]

            # Vertical flip + negate Y
            if random.random() > 0.5:
                color     = TF.vflip(color)
                normal    = TF.vflip(normal)
                normal[1] = -normal[1]

            # 90° rotations with correct XY channel transform
            k = random.choice([0, 1, 2, 3])
            if k:
                color  = torch.rot90(color,  k, [1, 2])
                normal = torch.rot90(normal, k, [1, 2])
                nx, ny = normal[0].clone(), normal[1].clone()
                if k == 1:
                    normal[0], normal[1] =  ny, -nx
                elif k == 2:
                    normal[0], normal[1] = -nx, -ny
                elif k == 3:
                    normal[0], normal[1] = -ny,  nx

            # Random crop + resize (75-100%)
            if random.random() > 0.5:
                H, W   = color.shape[1], color.shape[2]
                scale  = random.uniform(0.75, 1.0)
                ch, cw = int(H * scale), int(W * scale)
                top    = random.randint(0, H - ch)
                left   = random.randint(0, W - cw)
                color  = color[:,  top:top+ch, left:left+cw]
                normal = normal[:, top:top+ch, left:left+cw]
                color  = F.interpolate(color.unsqueeze(0),  size=(H, W), mode="nearest").squeeze(0)
                normal = F.interpolate(normal.unsqueeze(0), size=(H, W), mode="nearest").squeeze(0)

            # Colour jitter on RGB input only
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                contrast   = random.uniform(0.8, 1.2)
                color = torch.clamp(color * brightness, 0, 1)
                color = torch.clamp((color - 0.5) * contrast + 0.5, 0, 1)

        return color, normal


# ──────────────────────────────────────────────
# 3.  MODEL  (U-Net with residual blocks + GroupNorm)
# ──────────────────────────────────────────────

def _gn(channels):
    """GroupNorm with automatic group count."""
    groups = min(8, channels)
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            _gn(ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            _gn(ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks=1):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            _gn(out_ch), nn.SiLU(),
        ]
        for _ in range(n_blocks):
            layers.append(ResBlock(out_ch))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, n_blocks=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        layers = [
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.SiLU(),
        ]
        for _ in range(n_blocks):
            layers.append(ResBlock(out_ch))
        self.conv = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net tuned for pixel-art normal-map prediction.

    Parameters
    ----------
    in_ch          : input channels (3 = RGB after alpha pre-multiply)
    out_ch         : output channels (3 = XYZ normal)
    base           : base channel count; doubles at each encoder level
    num_bot_blocks : number of ResBlocks stacked inside the bottleneck.
                     2 = original; 4 = recommended (more global reasoning
                     at 8×8 with almost no extra VRAM cost).
    """

    def __init__(self, in_ch=3, out_ch=3, base=32, num_bot_blocks=4, enc_blocks=2):
        super().__init__()
        b  = base
        eb = enc_blocks  # ResBlocks per encoder/decoder level

        # ── Encoder ──────────────────────────────
        # enc0 gets eb blocks at full resolution — this is where fine
        # pixel-art edge features are extracted, so extra depth helps.
        enc0_layers = [
            nn.Conv2d(in_ch, b, 3, padding=1, bias=False),
            _gn(b), nn.SiLU(),
        ]
        for _ in range(eb):
            enc0_layers.append(ResBlock(b))
        self.enc0 = nn.Sequential(*enc0_layers)     # 128 × 128

        self.enc1 = Down(b,   b*2, n_blocks=eb)     #  64 × 64
        self.enc2 = Down(b*2, b*4, n_blocks=eb)     #  32 × 32
        self.enc3 = Down(b*4, b*8, n_blocks=eb)     #  16 × 16

        # ── Bottleneck ───────────────────────────
        # Downsample to 8×8, then run num_bot_blocks ResBlocks.
        # Each block is cheap here (8×8 = 64 positions) but gives
        # the network more depth to reason about global context.
        bot_layers = [Down(b*8, b*8, n_blocks=1)]   #   8 × 8
        for _ in range(num_bot_blocks):
            bot_layers.append(ResBlock(b*8))
        self.bot = nn.Sequential(*bot_layers)

        # ── Decoder ──────────────────────────────
        self.dec3 = Up(b*8, b*8, b*4, n_blocks=eb)  #  16 × 16
        self.dec2 = Up(b*4, b*4, b*2, n_blocks=eb)  #  32 × 32
        self.dec1 = Up(b*2, b*2, b,   n_blocks=eb)  #  64 × 64
        self.dec0 = Up(b,   b,   b,   n_blocks=eb)  # 128 × 128

        # ── Output head ──────────────────────────
        self.head = nn.Sequential(
            ResBlock(b),
            nn.Conv2d(b, out_ch, 1),
        )

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bot(e3)
        d  = self.dec3(b,  e3)
        d  = self.dec2(d,  e2)
        d  = self.dec1(d,  e1)
        d  = self.dec0(d,  e0)
        return torch.tanh(self.head(d))             # output in [−1, 1]


# ──────────────────────────────────────────────
# 4.  LOSS
# ──────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    L1 + Angular + Gradient-difference loss.

    Gradient-difference loss (GDL) computes the L1 difference between the
    spatial gradients (∂x, ∂y) of pred and target.  Because gradients are
    large at edges and near-zero on flat surfaces, GDL heavily penalises
    blurry or misplaced edges — exactly what pixel-art normal maps need.
    It costs one extra pair of conv ops per forward pass (negligible).
    """

    def __init__(self, l1_w=1.0, ang_w=1.0, grad_w=0.5):
        super().__init__()
        self.l1_w   = l1_w
        self.ang_w  = ang_w
        self.grad_w = grad_w

        # Sobel kernels for ∂x and ∂y — registered as buffers so they
        # move to the right device automatically with .to(device).
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradients(self, t: torch.Tensor):
        """Compute ∂x and ∂y for each channel independently."""
        B, C, H, W = t.shape
        t_flat = t.reshape(B * C, 1, H, W)
        sx = self.sobel_x.to(device=t.device, dtype=t.dtype)
        sy = self.sobel_y.to(device=t.device, dtype=t.dtype)
        gx = F.conv2d(t_flat, sx, padding=1).reshape(B, C, H, W)
        gy = F.conv2d(t_flat, sy, padding=1).reshape(B, C, H, W)
        return gx, gy

    def forward(self, pred, target):
        # L1 pixel loss
        l1 = F.l1_loss(pred, target)

        # Angular loss (geometric — treats XYZ as a unit-vector field)
        cos     = F.cosine_similarity(pred, target, dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
        angular = torch.acos(cos).mean()

        # Gradient-difference loss (sharpness)
        pred_gx,   pred_gy   = self._gradients(pred)
        target_gx, target_gy = self._gradients(target)
        grad = (F.l1_loss(pred_gx, target_gx) +
                F.l1_loss(pred_gy, target_gy)) * 0.5

        total = self.l1_w * l1 + self.ang_w * angular + self.grad_w * grad
        return total, l1.item(), angular.item(), grad.item()


# ──────────────────────────────────────────────
# 5.  METRICS
# ──────────────────────────────────────────────

def mean_angular_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Returns mean angular error in degrees."""
    cos = F.cosine_similarity(pred, target, dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos).mean().item() * (180.0 / math.pi)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10(4.0 / mse)             # signal range is [−1,1] → max=2, max²=4


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    # normalise from [−1,1] to [0,1] for SSIM
    p = (pred.clamp(-1, 1) + 1) / 2
    t = (target.clamp(-1, 1) + 1) / 2
    if HAS_MSSSIM:
        return pt_ssim(p, t, data_range=1.0, size_average=True).item()
    # Numpy fallback (slow but correct)
    p_np = p.cpu().numpy()
    t_np = t.cpu().numpy()
    from skimage.metrics import structural_similarity
    vals = []
    for i in range(p_np.shape[0]):
        v = structural_similarity(
            p_np[i].transpose(1,2,0), t_np[i].transpose(1,2,0),
            data_range=1.0, channel_axis=-1
        )
        vals.append(v)
    return float(np.mean(vals))


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(pred, target).item()


# ──────────────────────────────────────────────
# 6.  TRAINING LOOP
# ──────────────────────────────────────────────

def train(cfg: dict):
    # ── Reproducibility ──
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"         {torch.cuda.get_device_name(0)}")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ── Data ──
    full_ds = NormalMapDataset(
        render_log   = cfg["render_log"],
        dataset_root = cfg["dataset_root"],
        image_size   = cfg["image_size"],
        augment      = True,
    )
    n_val   = max(1, int(len(full_ds) * cfg["val_split"]))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )
    # Disable augmentation for the validation subset
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"],
                              pin_memory=True)

    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    # ── Model ──
    model = UNet(in_ch=3, out_ch=3, base=cfg["base_channels"],
                 num_bot_blocks=cfg["num_bot_blocks"],
                 enc_blocks=cfg["enc_blocks"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] U-Net  params={n_params/1e6:.2f} M")

    # ── Loss / Optimiser / Scheduler ──
    criterion = CombinedLoss(cfg["l1_weight"], cfg["angular_weight"], cfg["grad_weight"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr        = cfg["lr"],
        epochs        = cfg["num_epochs"],
        steps_per_epoch = len(train_loader),
        pct_start     = 0.20,
        div_factor    = 10,
        final_div_factor = 50,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    # ── Metrics CSV ──
    csv_path = out_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "epoch",
        "train_loss", "train_l1", "train_angular",
        "val_loss",   "val_l1",   "val_angular",
        "val_mae",    "val_psnr", "val_ssim",
        "val_mae_deg",
        "lr",         "epoch_time_s",
    ])
    csv_file.flush()

    best_val_loss = float("inf")
    print(f"\n{'─'*65}")
    print(f"  Starting training  →  {cfg['num_epochs']} epochs")
    print(f"{'─'*65}\n")

    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        tr_loss = tr_l1 = tr_ang = 0.0
        for color, normal in train_loader:
            color, normal = color.to(device), normal.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg["amp"]):
                pred = model(color)
                loss, l1_v, ang_v, grad_v = criterion(pred, normal)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            bs = color.size(0)
            tr_loss += loss.item() * bs
            tr_l1   += l1_v        * bs
            tr_ang  += ang_v       * bs

        N = len(train_ds)
        tr_loss /= N; tr_l1 /= N; tr_ang /= N

        # ── Validate ──
        model.eval()
        vl_loss = vl_l1 = vl_ang = vl_mae = vl_psnr = vl_ssim = vl_mae_deg = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for color, normal in val_loader:
                color, normal = color.to(device), normal.to(device)
                with torch.autocast(device_type=device.type, enabled=cfg["amp"]):
                    pred = model(color)
                    loss, l1_v, ang_v, _ = criterion(pred, normal)
                bs = color.size(0)
                # Cast to float32 for metric computation
                pred_f  = pred.float()
                normal_f = normal.float()
                vl_loss    += loss.item()  * bs
                vl_l1      += l1_v         * bs
                vl_ang     += ang_v        * bs
                vl_mae     += compute_mae(pred_f, normal_f)     * bs
                vl_psnr    += psnr(pred_f, normal_f)             * bs
                vl_ssim    += compute_ssim(pred_f, normal_f)     * bs
                vl_mae_deg += mean_angular_error(pred_f, normal_f) * bs
                n_val_samples += bs

        vl_loss    /= n_val_samples
        vl_l1      /= n_val_samples
        vl_ang     /= n_val_samples
        vl_mae     /= n_val_samples
        vl_psnr    /= n_val_samples
        vl_ssim    /= n_val_samples
        vl_mae_deg /= n_val_samples

        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - t0

        # ── Log ──
        writer.writerow([
            epoch,
            f"{tr_loss:.6f}", f"{tr_l1:.6f}", f"{tr_ang:.6f}",
            f"{vl_loss:.6f}", f"{vl_l1:.6f}", f"{vl_ang:.6f}",
            f"{vl_mae:.6f}",  f"{vl_psnr:.4f}", f"{vl_ssim:.6f}",
            f"{vl_mae_deg:.4f}",
            f"{current_lr:.2e}", f"{epoch_time:.1f}",
        ])
        csv_file.flush()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Ep {epoch:>4}/{cfg['num_epochs']}  "
                f"tr={tr_loss:.4f}  vl={vl_loss:.4f}  "
                f"PSNR={vl_psnr:.2f}  SSIM={vl_ssim:.3f}  "
                f"AngErr={vl_mae_deg:.2f}°  "
                f"lr={current_lr:.2e}  {epoch_time:.0f}s"
            )

        # ── Checkpoints ──
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": vl_loss,
                "cfg": cfg,
            }, ckpt_dir / "best.pth")

        if epoch % cfg["save_every"] == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg,
            }, ckpt_dir / f"epoch_{epoch:04d}.pth")

    csv_file.close()
    print(f"\n[done] Best val loss: {best_val_loss:.6f}")
    print(f"       Metrics saved to: {csv_path}")
    print(f"       Best checkpoint:  {ckpt_dir / 'best.pth'}")


# ──────────────────────────────────────────────
# 6b.  FINE-TUNE  (continue from a checkpoint on new high-quality data)
# ──────────────────────────────────────────────

def finetune(cfg: dict):
    """
    Fine-tune a pretrained checkpoint on a new high-quality dataset.

    Key differences from train():
    - Loads model weights (and optionally optimizer state) from cfg["finetune_ckpt"].
    - Uses FinetuneDataset, which pads non-square images to square before resizing.
    - Lower default LR (cfg["lr"] is divided by 10 unless --lr is passed explicitly).
    - CosineAnnealingLR instead of OneCycleLR – gentler decay for small datasets.
    - All augmentations are enabled (same as training).
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])

    # ── Dataset ──
    ds = FinetuneDataset(
        render_log = cfg["finetune_log"],
        image_size = cfg["image_size"],
        augment    = True,
    )
    n_val   = max(1, int(len(ds) * cfg["val_split"]))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"],
                              pin_memory=True)
    print(f"[finetune] train={len(train_ds)}  val={len(val_ds)}")

    # ── Model – load from checkpoint ──
    ckpt = torch.load(cfg["finetune_ckpt"], map_location=device)
    # Architecture params: prefer saved cfg, fall back to current cfg
    saved_cfg      = ckpt.get("cfg", {})
    base_channels  = saved_cfg.get("base_channels",  cfg["base_channels"])
    num_bot_blocks = saved_cfg.get("num_bot_blocks",  cfg["num_bot_blocks"])
    enc_blocks     = saved_cfg.get("enc_blocks",      cfg["enc_blocks"])
    model = UNet(in_ch=3, out_ch=3, base=base_channels,
                 num_bot_blocks=num_bot_blocks,
                 enc_blocks=enc_blocks).to(device)
    model.load_state_dict(ckpt["model_state"])
    resumed_epoch = ckpt.get("epoch", 0)
    print(f"[finetune] Loaded checkpoint from epoch {resumed_epoch}: {cfg['finetune_ckpt']}")
    print(f"[finetune] Architecture: base={base_channels}  bot={num_bot_blocks}  enc={enc_blocks}")

    # ── Loss / Optimiser / Scheduler ──
    criterion = CombinedLoss(cfg["l1_weight"], cfg["angular_weight"], cfg["grad_weight"])
    ft_lr     = cfg["lr"]   # caller should pass a small LR via --lr (e.g. 3e-5)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=ft_lr,
                                  weight_decay=cfg["weight_decay"])
    # CosineAnnealingLR: smooth decay from ft_lr → ft_lr/100 over all epochs.
    # Much gentler than OneCycleLR for fine-tuning – avoids destroying pretrained weights.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = cfg["num_epochs"] * len(train_loader),
        eta_min = ft_lr / 100,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    # ── Metrics CSV ──
    csv_path = out_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "epoch",
        "train_loss", "train_l1", "train_angular",
        "val_loss",   "val_l1",   "val_angular",
        "val_mae",    "val_psnr", "val_ssim",
        "val_mae_deg",
        "lr",         "epoch_time_s",
    ])
    csv_file.flush()

    best_val_loss = float("inf")
    print(f"\n{'─'*65}")
    print(f"  Fine-tuning  →  {cfg['num_epochs']} epochs  (lr={ft_lr:.1e})")
    print(f"{'─'*65}\n")

    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        tr_loss = tr_l1 = tr_ang = 0.0
        for color, normal in train_loader:
            color, normal = color.to(device), normal.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg["amp"]):
                pred = model(color)
                loss, l1_v, ang_v, grad_v = criterion(pred, normal)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            bs = color.size(0)
            tr_loss += loss.item() * bs
            tr_l1   += l1_v        * bs
            tr_ang  += ang_v       * bs

        N = len(train_ds)
        tr_loss /= N; tr_l1 /= N; tr_ang /= N

        # ── Validate ──
        model.eval()
        vl_loss = vl_l1 = vl_ang = vl_mae = vl_psnr = vl_ssim = vl_mae_deg = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for color, normal in val_loader:
                color, normal = color.to(device), normal.to(device)
                with torch.autocast(device_type=device.type, enabled=cfg["amp"]):
                    pred = model(color)
                    loss, l1_v, ang_v, _ = criterion(pred, normal)
                bs = color.size(0)
                pred_f   = pred.float()
                normal_f = normal.float()
                vl_loss    += loss.item()  * bs
                vl_l1      += l1_v         * bs
                vl_ang     += ang_v        * bs
                vl_mae     += compute_mae(pred_f, normal_f)          * bs
                vl_psnr    += psnr(pred_f, normal_f)                  * bs
                vl_ssim    += compute_ssim(pred_f, normal_f)          * bs
                vl_mae_deg += mean_angular_error(pred_f, normal_f)    * bs
                n_val_samples += bs

        vl_loss    /= n_val_samples
        vl_l1      /= n_val_samples
        vl_ang     /= n_val_samples
        vl_mae     /= n_val_samples
        vl_psnr    /= n_val_samples
        vl_ssim    /= n_val_samples
        vl_mae_deg /= n_val_samples

        current_lr  = scheduler.get_last_lr()[0]
        epoch_time  = time.time() - t0

        writer.writerow([
            epoch,
            f"{tr_loss:.6f}", f"{tr_l1:.6f}", f"{tr_ang:.6f}",
            f"{vl_loss:.6f}", f"{vl_l1:.6f}", f"{vl_ang:.6f}",
            f"{vl_mae:.6f}",  f"{vl_psnr:.4f}", f"{vl_ssim:.6f}",
            f"{vl_mae_deg:.4f}",
            f"{current_lr:.2e}", f"{epoch_time:.1f}",
        ])
        csv_file.flush()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Ep {epoch:>4}/{cfg['num_epochs']}  "
                f"tr={tr_loss:.4f}  vl={vl_loss:.4f}  "
                f"PSNR={vl_psnr:.2f}  SSIM={vl_ssim:.3f}  "
                f"AngErr={vl_mae_deg:.2f}°  "
                f"lr={current_lr:.2e}  {epoch_time:.0f}s"
            )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({
                "epoch": resumed_epoch + epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": vl_loss,
                "cfg": cfg,
            }, ckpt_dir / "best.pth")

        if epoch % cfg["save_every"] == 0:
            torch.save({
                "epoch": resumed_epoch + epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg,
            }, ckpt_dir / f"epoch_{epoch:04d}.pth")

    csv_file.close()
    print(f"\n[done] Best val loss: {best_val_loss:.6f}")
    print(f"       Metrics saved to: {csv_path}")
    print(f"       Best checkpoint:  {ckpt_dir / 'best.pth'}")


# ──────────────────────────────────────────────
# 7.  INFERENCE HELPER
# ──────────────────────────────────────────────

def _pad_to_square(img: "Image.Image"):
    """
    Pad an RGBA image to a square canvas (centred, transparent padding).
    Returns (square_image, crop_box) where crop_box is (left, top, right, bottom)
    in square-image coordinates that recovers the original content.
    """
    w, h   = img.size
    side   = max(w, h)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    left   = (side - w) // 2
    top    = (side - h) // 2
    square.paste(img, (left, top))
    return square, (left, top, left + w, top + h)


@torch.no_grad()
def predict_single(checkpoint_path: str, image_path: str, output_path: str,
                   image_size: int = 128, base_channels: int = 32,
                   num_bot_blocks: int = 4, enc_blocks: int = 2):
    """
    Generate a normal map for a single colour sprite.

    Non-square inputs are centred on a transparent square canvas before
    inference, and the output is cropped back to the original dimensions.
    Output is an 8-bit RGB PNG in [0, 255] (Unity / Godot compatible).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg    = ckpt.get("cfg", {})
    base      = cfg.get("base_channels", base_channels)
    size      = cfg.get("image_size", image_size)
    n_bot     = cfg.get("num_bot_blocks", num_bot_blocks)
    n_enc     = cfg.get("enc_blocks", enc_blocks)

    model = UNet(in_ch=3, out_ch=3, base=base,
                 num_bot_blocks=n_bot, enc_blocks=n_enc).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    orig = Image.open(image_path).convert("RGBA")
    orig_w, orig_h = orig.size

    if orig_w == orig_h:
        padded   = orig
        crop_box = None
    else:
        padded, crop_box = _pad_to_square(orig)
        print(f"[predict] Non-square ({orig_w}x{orig_h}) padded to {padded.size[0]}x{padded.size[1]}")

    resized = padded.resize((size, size), Image.NEAREST)
    arr     = np.array(resized, dtype=np.float32) / 255.0
    rgb     = (arr[..., :3] * arr[..., 3:4]).transpose(2, 0, 1)
    x       = torch.from_numpy(rgb).unsqueeze(0).to(device)

    pred = model(x).squeeze(0).cpu().numpy()
    out  = ((pred + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(out.transpose(1, 2, 0), mode="RGB")

    if crop_box is not None:
        pad_side  = padded.size[0]
        scale     = size / pad_side
        scaled_box = tuple(int(round(v * scale)) for v in crop_box)
        result = result.crop(scaled_box)
        result = result.resize((orig_w, orig_h), Image.NEAREST)
        print(f"[predict] Cropped output back to {orig_w}x{orig_h}")

    result.save(output_path)
    print(f"[predict] Saved -> {output_path}")


# ──────────────────────────────────────────────
# 8.  ENTRY POINT
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train pixel-art normal-map U-Net")
    # ── Dataset ──
    p.add_argument("--render_log",      default=DEFAULT_CFG["render_log"],
                   help="Path to render_log.csv produced by the Blender pipeline")
    p.add_argument("--dataset_root",    default=DEFAULT_CFG["dataset_root"],
                   help="Local root folder that contains dataset/color/ and "
                        "dataset/normal/. Leave empty when running on the same "
                        "machine that rendered the data (absolute paths are used).")
    # ── Output ──
    p.add_argument("--output_dir",      default=DEFAULT_CFG["output_dir"])
    # ── Training ──
    p.add_argument("--image_size",      type=int,   default=DEFAULT_CFG["image_size"])
    p.add_argument("--batch_size",      type=int,   default=DEFAULT_CFG["batch_size"])
    p.add_argument("--num_epochs",      type=int,   default=DEFAULT_CFG["num_epochs"])
    p.add_argument("--lr",              type=float, default=DEFAULT_CFG["lr"])
    p.add_argument("--weight_decay",    type=float, default=DEFAULT_CFG["weight_decay"])
    p.add_argument("--val_split",       type=float, default=DEFAULT_CFG["val_split"])
    p.add_argument("--l1_weight",       type=float, default=DEFAULT_CFG["l1_weight"])
    p.add_argument("--angular_weight",  type=float, default=DEFAULT_CFG["angular_weight"])
    p.add_argument("--num_workers",     type=int,   default=DEFAULT_CFG["num_workers"])
    p.add_argument("--seed",            type=int,   default=DEFAULT_CFG["seed"])
    p.add_argument("--save_every",      type=int,   default=DEFAULT_CFG["save_every"])
    p.add_argument("--base_channels",   type=int,   default=DEFAULT_CFG["base_channels"])
    p.add_argument("--num_bot_blocks",  type=int,   default=DEFAULT_CFG["num_bot_blocks"],
                   help="ResBlocks in the bottleneck (default 4; original was 2)")
    p.add_argument("--enc_blocks",      type=int,   default=DEFAULT_CFG["enc_blocks"],
                   help="ResBlocks per encoder/decoder level (1=original, 2=recommended)")
    p.add_argument("--grad_weight",     type=float, default=DEFAULT_CFG["grad_weight"],
                   help="Gradient-difference loss weight (0=off, 0.5=recommended)")
    p.add_argument("--no_amp",          action="store_true",
                   help="Disable mixed-precision training (slower, uses more VRAM)")
    # ── Fine-tune ──
    p.add_argument("--finetune",        action="store_true",
                   help="Fine-tune mode: load a pretrained checkpoint and train on new data")
    p.add_argument("--finetune_ckpt",   default="runs/exp5/checkpoints/best.pth",
                   help="Checkpoint to start fine-tuning from")
    p.add_argument("--finetune_log",    default="new_dataset/render_log.csv",
                   help="render_log.csv for the new fine-tune dataset")
    # ── Inference ──
    p.add_argument("--predict",         action="store_true",
                   help="Run inference on a single image instead of training")
    p.add_argument("--checkpoint",      default="runs/exp5/checkpoints/best.pth")
    p.add_argument("--input_image",     default=None)
    p.add_argument("--output_image",    default="predicted_normal.png")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.predict:
        if not args.input_image:
            raise ValueError("--input_image required for --predict mode")
        predict_single(
            checkpoint_path = args.checkpoint,
            image_path      = args.input_image,
            output_path     = args.output_image,
            image_size      = args.image_size,
            base_channels   = args.base_channels,
            num_bot_blocks  = args.num_bot_blocks,
            enc_blocks      = args.enc_blocks,
        )
    elif args.finetune:
        cfg = {k: getattr(args, k, DEFAULT_CFG[k]) for k in DEFAULT_CFG}
        cfg["amp"]           = not args.no_amp
        cfg["finetune_ckpt"] = args.finetune_ckpt
        cfg["finetune_log"]  = args.finetune_log
        finetune(cfg)
    else:
        cfg = {k: getattr(args, k, DEFAULT_CFG[k]) for k in DEFAULT_CFG}
        cfg["amp"] = not args.no_amp
        train(cfg)