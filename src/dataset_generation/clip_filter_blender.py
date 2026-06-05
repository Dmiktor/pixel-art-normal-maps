r"""
clip_filter_blender.py
----------------------
Replaces clip_filter.py. Orchestrates batch CLIP-based character filtering
using Blender for rendering (no trimesh/pyrender needed).

Pipeline per model:
  1. Scan INPUT_ROOT for .glb files.
  2. Call Blender as a subprocess per model (blender_render_single.py).
  3. Score the rendered PNG with CLIP  (pos_score - neg_score).
  4. Copy accepted models to ACCEPTED_DIR.
  5. Write clip_results.csv with every score.

Usage:
    python clip_filter_blender.py \
        --blender "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" \
        --limit 100 \
        --threshold 0.05 \
        --workers 2

Requirements (host Python, NOT Blender's Python):
    pip install torch open_clip_torch Pillow tqdm
"""

import os
import csv
import shutil
import argparse
import logging
import subprocess
import threading
import concurrent.futures
from pathlib import Path

import torch
import open_clip
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("clip_filter_blender")


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
INPUT_ROOT   = BASE_DIR / "hf-objaverse-v1" / "glbs"
RENDER_DIR   = BASE_DIR / "blender_renders"
ACCEPTED_DIR = BASE_DIR / "accepted_characters"
RESULTS_CSV  = BASE_DIR / "clip_results.csv"

# Must live next to this script
BLENDER_SCRIPT = BASE_DIR / "blender_render_single.py"

RENDER_SIZE = 224

POSITIVE_PROMPTS = [
    "a 3d model of a human character",
    "a humanoid avatar",
    "a cartoon character",
    "a game character",
    "a person standing",
]

NEGATIVE_PROMPTS = [
    "a building",
    "a rock",
    "a tree",
    "a vehicle",
    "furniture",
    "an animal",
    "a landscape",
    "abstract geometry",
]


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="CLIP character filter — Blender backend")
    p.add_argument(
        "--blender", default="blender",
        help="Blender executable path. "
             r'Example: "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"'
    )
    p.add_argument("--limit",      type=int,   default=0,    help="Max models to process (0=all)")
    p.add_argument("--threshold",  type=float, default=0.05, help="Min CLIP score to accept")
    p.add_argument("--workers",    type=int,   default=1,    help="Parallel Blender processes")
    p.add_argument("--render_size",type=int,   default=RENDER_SIZE)
    p.add_argument("--input_root", default=str(INPUT_ROOT))
    p.add_argument(
        "--keep_renders", action="store_true",
        help="Keep intermediate PNGs after scoring"
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Print full Blender stdout/stderr for every model (very verbose)"
    )
    return p.parse_args()


# ──────────────────────────────────────────────
# STEP 1 — DISCOVER MODELS
# ──────────────────────────────────────────────
def find_models(root, limit):
    files = sorted(Path(root).rglob("*.glb"))
    if limit > 0:
        files = files[:limit]
    log.info("Found %d models under %s", len(files), root)
    return files


# ──────────────────────────────────────────────
# STEP 2 — BLENDER RENDER
# ──────────────────────────────────────────────
def render_with_blender(glb_path, render_dir, blender_exe, render_size, debug=False):
    """
    Spawn Blender headlessly to render one GLB to a preview PNG.
    Returns the PNG Path on success, None on any failure.
    Always prints Blender's output when the render fails.
    """
    stem    = glb_path.stem
    out_png = render_dir / (stem + "_preview.png")

    if out_png.exists():
        log.info("Cache hit: %s", out_png.name)
        return out_png

    cmd = [
        blender_exe,
        "--background",
        "--python", str(BLENDER_SCRIPT),
        "--",
        "--model",        str(glb_path),
        "--out_dir",      str(render_dir),
        "--render_size",  str(render_size),
    ]

    if debug:
        log.info("CMD: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout so nothing is lost
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        log.warning("[TIMEOUT] %s", glb_path.name)
        return None
    except FileNotFoundError:
        log.error(
            "Blender not found at: '%s'\n"
            "Pass the full path with --blender, e.g.:\n"
            r'  --blender "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"',
            blender_exe
        )
        raise   # fatal

    output = result.stdout.decode("utf-8", errors="replace")

    # Success: our script prints "RENDER_OK:<path>" and the file exists
    if "RENDER_OK:" in output and out_png.exists():
        if debug:
            log.info("[OK] %s\n%s", glb_path.name, output)
        return out_png

    # Failure — always show the last 20 lines of Blender output
    tail = "\n".join(output.strip().splitlines()[-20:])
    log.warning(
        "Render FAILED for %s (exit=%d)\n--- Blender output (last 20 lines) ---\n%s\n---",
        glb_path.name, result.returncode, tail
    )
    return None


# ──────────────────────────────────────────────
# STEP 3 — CLIP
# ──────────────────────────────────────────────
def load_clip(device):
    log.info("Loading CLIP (ViT-B/32)...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model     = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    log.info("CLIP ready on %s", device)
    return model, preprocess, tokenizer


def encode_texts(texts, model, tokenizer, device):
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        feats  = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats


def score_image(png_path, model, preprocess, pos_feats, neg_feats, device):
    try:
        img = Image.open(png_path).convert("RGB")
    except Exception as e:
        log.warning("Cannot open image %s: %s", png_path.name, e)
        return float("-inf")

    img_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat  = model.encode_image(img_input)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)

    pos_score = (img_feat @ pos_feats.T).max().item()
    neg_score = (img_feat @ neg_feats.T).max().item()
    return pos_score - neg_score


# ──────────────────────────────────────────────
# STEP 4 — ACCEPT
# ──────────────────────────────────────────────
def accept_model(glb_path, accepted_dir):
    accepted_dir.mkdir(parents=True, exist_ok=True)
    dst = accepted_dir / glb_path.name
    if dst.exists():
        dst = accepted_dir / (glb_path.parent.name + "_" + glb_path.name)
    shutil.copy2(glb_path, dst)
    log.info("  ACCEPTED -> %s", dst.name)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    render_dir   = RENDER_DIR
    accepted_dir = ACCEPTED_DIR
    threshold    = args.threshold

    render_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)

    if not BLENDER_SCRIPT.exists():
        log.error(
            "blender_render_single.py not found at %s\n"
            "Make sure it lives in the same directory as this script.",
            BLENDER_SCRIPT
        )
        return

    models = find_models(args.input_root, args.limit)
    if not models:
        log.error("No .glb files found under %s", args.input_root)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    clip_model, preprocess, tokenizer = load_clip(device)
    pos_feats = encode_texts(POSITIVE_PROMPTS, clip_model, tokenizer, device)
    neg_feats = encode_texts(NEGATIVE_PROMPTS, clip_model, tokenizer, device)

    results        = []
    accepted_count = 0
    csv_lock       = threading.Lock()

    # Open the CSV immediately and write the header so rows flush as they arrive.
    # This means results survive a mid-run crash.
    csv_file   = open(RESULTS_CSV, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["stem", "score", "accepted", "error", "model"])
    csv_writer.writeheader()
    csv_file.flush()

    def write_row(row):
        with csv_lock:
            csv_writer.writerow(row)
            csv_file.flush()

    def process_one(glb_path):
        row = {
            "model":    str(glb_path),
            "stem":     glb_path.stem,
            "score":    None,
            "accepted": False,
            "error":    "",
        }

        png = render_with_blender(
            glb_path, render_dir, args.blender, args.render_size, debug=args.debug
        )
        if png is None:
            row["error"] = "render_failed"
            write_row(row)
            return row

        score        = score_image(png, clip_model, preprocess, pos_feats, neg_feats, device)
        row["score"] = round(score, 6)

        symbol = "OK" if score >= threshold else "--"
        log.info("[%s] %-45s  score=%.4f", symbol, glb_path.name[:45], score)

        if score >= threshold:
            accept_model(glb_path, accepted_dir)
            row["accepted"] = True

        if not args.keep_renders:
            try:
                png.unlink()
            except OSError:
                pass

        write_row(row)
        return row

    if args.workers > 1:
        log.info("Parallel workers: %d", args.workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, p): p for p in models}
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(models), desc="Filtering"
            ):
                row = future.result()
                results.append(row)
                if row["accepted"]:
                    accepted_count += 1
    else:
        for glb_path in tqdm(models, desc="Filtering"):
            row = process_one(glb_path)
            results.append(row)
            if row["accepted"]:
                accepted_count += 1

    csv_file.close()

    log.info("=" * 60)
    log.info("Total    : %d", len(models))
    log.info("Accepted : %d  (threshold=%.3f)", accepted_count, threshold)
    log.info("Failed   : %d", sum(1 for r in results if r["error"]))
    log.info("CSV      : %s", RESULTS_CSV)
    log.info("Output   : %s", accepted_dir)


if __name__ == "__main__":
    main()