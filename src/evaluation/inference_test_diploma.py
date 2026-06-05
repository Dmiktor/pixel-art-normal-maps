"""
inference_test_diploma.py
-------------------------
Generates the figure for Додаток В (Приклад передбачення моделi).

Inputs : the three RGBA test sprites in "Test for diploma/":
           - db32-Char.png
           - ghost-1.png
           - hero-idle-1.png
Model  : runs/exp_balanced/checkpoints/best.pth   (UNet trained earlier).

For each input sprite the script:
  1. Loads the RGBA image at its natural resolution.
  2. Pads with NEUTRAL-NORMAL background  (0.5, 0.5, 1.0)  on the colour
     side and with TRANSPARENT BLACK (0, 0, 0, 0) on the mask side.
     This is required by the U-Net architecture (input must be square
     and divisible by 8) but keeps the output visually clean.
  3. Runs the U-Net.
  4. Re-applies the alpha mask: predicted normals are kept inside the
     silhouette, everywhere else we paint the neutral normal so the
     output looks like a real game normal-map asset.
  5. Saves a side-by-side  [input | predicted normal]  pair for each
     sprite, and finally stitches all three pairs into ONE wide row
     image ("appendix_predictions.png") at exactly the same scale as
     the originals (no smoothing, NEAREST upscaling for clarity).

The final image goes to ../../Диплом/Зображення/appendix_predictions.png
so that the LaTeX figure picks it up automatically.

Usage (Windows venv):
    python inference_test_diploma.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── make UNet importable ─────────────────────────────────────────────
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_normalmap import UNet  # noqa: E402


# ── paths ────────────────────────────────────────────────────────────
TEST_DIR     = HERE / "Test for diploma"
CKPT_PATH    = HERE / "runs" / "exp_balanced" / "checkpoints" / "best.pth"
OUT_DIR      = HERE.parent.parent / "Диплом" / "Зображення"
OUT_NAME     = "appendix_predictions.png"
DISPLAY_SCALE = 6        # NEAREST upscale for the final figure (px → px*N)

# ── settings (must match training) ───────────────────────────────────
ALPHA_THRESH = 0.05
# Neutral normal (0,0,1) encoded to [0,1] is (0.5, 0.5, 1.0)
# In 8-bit RGB that is (128, 128, 255).
NEUTRAL_RGB  = (128, 128, 255)


def round_up_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def pad_to_square_multiple_of_8(img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Center-pads an RGBA image to a square canvas whose side is divisible
    by 8 (required by the U-Net's 3 downsamplings). The padding is
    transparent black on the input side; the surrounding background
    will be re-painted with the neutral normal AFTER the prediction.

    Returns the padded image AND the (left, top, w, h) crop box of the
    original sprite so we can paste the prediction back at its location.
    """
    w, h = img.size
    side = round_up_to_multiple(max(w, h), 8)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    left = (side - w) // 2
    top  = (side - h) // 2
    canvas.paste(img, (left, top), img)
    return canvas, (left, top, w, h)


@torch.no_grad()
def predict_normal(model: torch.nn.Module, rgba: Image.Image, device: torch.device) -> np.ndarray:
    """
    rgba: PIL RGBA image, already padded to (S, S) with S % 8 == 0.

    Returns a (S, S, 3) float array in [0, 1] suitable for direct
    saving as an 8-bit normal map.
    """
    arr = np.asarray(rgba, dtype=np.float32) / 255.0       # H W 4
    x   = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    n   = model(x)                                         # 1 3 H W, unit
    n   = n[0].cpu().numpy().transpose(1, 2, 0)            # H W 3 in [-1, 1]
    # Encode to [0, 1]
    return (n * 0.5) + 0.5


def compose_normal_with_mask(pred01: np.ndarray, rgba: Image.Image) -> Image.Image:
    """
    Paint the neutral normal everywhere alpha is below the threshold,
    keep the predicted normal inside the silhouette. Returns an RGB PIL
    image of the same size as `rgba`.
    """
    alpha = np.asarray(rgba, dtype=np.float32)[:, :, 3] / 255.0
    mask  = (alpha > ALPHA_THRESH)[:, :, None]             # H W 1
    neutral = np.array(NEUTRAL_RGB, dtype=np.float32) / 255.0
    rgb = np.where(mask, pred01, neutral).clip(0.0, 1.0)
    rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(rgb_u8, mode="RGB")


def nearest_upscale(img: Image.Image, factor: int) -> Image.Image:
    if factor <= 1:
        return img
    w, h = img.size
    return img.resize((w * factor, h * factor), Image.NEAREST)


def process_one(path: Path, model: torch.nn.Module, device: torch.device) -> tuple[Image.Image, Image.Image]:
    """
    Returns (input_display, prediction_display), both at the SAME size,
    NEAREST-upscaled by DISPLAY_SCALE for the figure.
    The prediction is cropped back to the original sprite bounds and
    has neutral normal as a background.
    """
    rgba = Image.open(path).convert("RGBA")
    orig_w, orig_h = rgba.size

    # 1. Pad to a square divisible by 8
    padded, (left, top, w, h) = pad_to_square_multiple_of_8(rgba)
    # 2. Run the U-Net
    pred01 = predict_normal(model, padded, device)         # (S,S,3) [0,1]
    # 3. Crop prediction back to the original sprite rectangle
    pred_crop = pred01[top:top + h, left:left + w, :]
    # 4. Compose with the original alpha mask (neutral normal outside)
    pred_img  = compose_normal_with_mask(pred_crop, rgba)
    # 5. NEAREST-upscale both panels equally for the figure
    return (nearest_upscale(rgba.convert("RGB"), DISPLAY_SCALE),
            nearest_upscale(pred_img,            DISPLAY_SCALE))


def stitch_horizontal(panels: list[Image.Image], gap: int = 12,
                      bg: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Place panels side-by-side on a white background, vertically centered."""
    max_h = max(p.height for p in panels)
    total_w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    out = Image.new("RGB", (total_w, max_h), bg)
    x = 0
    for p in panels:
        y = (max_h - p.height) // 2
        out.paste(p, (x, y))
        x += p.width + gap
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device = {device}")

    # Load checkpoint first to auto-detect the base width
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)   # support both raw state_dict & checkpoint dict

    # Infer `base` from the stem conv: enc0.0.weight has shape (base, in_ch=4, 3, 3)
    stem_w = state.get("enc0.0.weight")
    if stem_w is None:
        raise RuntimeError("Could not find 'enc0.0.weight' in checkpoint.")
    base = stem_w.shape[0]
    print(f"[infer] auto-detected base width = {base}")

    # Build the model with the detected base width and load the state dict
    model = UNet(in_ch=4, out_ch=3, base=base, enc_blocks=2, bot_blocks=4).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"[infer] loaded {CKPT_PATH}")

    # Discover three test sprites in fixed order for reproducibility
    candidates = sorted(TEST_DIR.glob("*.png"))
    inputs = [p for p in candidates if not p.name.startswith("Shapovalov")]
    if len(inputs) != 3:
        print(f"[infer] WARNING: expected 3 sprites in {TEST_DIR}, found {len(inputs)}")

    # Process each sprite -> (input_panel, prediction_panel)
    triple_panels: list[Image.Image] = []
    for p in inputs:
        in_disp, pred_disp = process_one(p, model, device)
        # For each sprite we want input AND prediction side-by-side as one chunk
        pair = stitch_horizontal([in_disp, pred_disp], gap=4)
        triple_panels.append(pair)
        print(f"[infer] processed {p.name}")

    # Final figure: three [input | prediction] pairs in a row
    final = stitch_horizontal(triple_panels, gap=24)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_NAME
    final.save(out_path, optimize=True)
    print(f"[infer] saved {out_path}  ({final.size[0]} x {final.size[1]} px)")


if __name__ == "__main__":
    main()
