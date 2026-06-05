"""
batch_render.py
---------------
Orchestrator that drives blender_pipeline.py across every model in
accepted_characters/, producing a pix2pix training dataset of
pixel-art RGB + stylised normal-map pairs.

Typical workflow
----------------
# 1. Dry-run: see the full job plan, touch nothing
python batch_render.py --blender /path/to/blender --dry_run

# 2. Smoke test: render 3 random models, 2 variants each
python batch_render.py --blender /path/to/blender --sample 3 --variants 2

# 3. Full run
python batch_render.py --blender /path/to/blender --variants 8 --workers 3

# 4. Full run WITH animation sampling
python batch_render.py --blender /path/to/blender --variants 8 \
    --sample_animations --anim_frames 4 --workers 3

Flags
-----
--dry_run          Print the complete job table and exit. Nothing is rendered.
--sample N         Pick N random models and run only those (useful for testing
                   before committing to the full dataset).
--sample_animations  Pass --sample_animations through to blender_pipeline.py.
--anim_frames N    Frames sampled per action (passed through, default 4).

Output layout
-------------
dataset/
    color/<stem>_r000_e15_s064.png
    normal/<stem>_r000_e15_s064.png
    color/<stem>_r090_e25_s048.png
    normal/<stem>_r090_e25_s048.png
    …
    render_log.csv
"""

import os
import csv
import argparse
import logging
import random
import subprocess
import sys
import threading
import concurrent.futures
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("batch_render")


# ─────────────────────────────────────────────────────────────
# PIXEL-ART PARAMETER SPACE
# ─────────────────────────────────────────────────────────────

# Canonical pixel-art character sprite sizes (rendered at these exact pixels)
PIXEL_ART_SIZES = [64]

# Camera elevation angles in degrees
#   0  = pure front view (fighting-game / platformer)
#  15  = slight top-down (most common RPG overworld)
#  25  = classic 2-D RPG (SNES Final Fantasy style)
#  35  = steeper top-down (Zelda-style)
ELEVATION_ANGLES = [0, 0, 15]

# Model Z-rotations — different character facings
#   0   = front
#  45   = front-right diagonal
#  90   = right profile
# 135   = back-right diagonal
# 180   = back
ROTATION_ANGLES = [0, 0, 15, 15, 45, 60, -15, -15, -45, -60]

DEFAULT_VARIANTS = 10


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch pixel-art sprite renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── paths ────────────────────────────────────────────────
    p.add_argument(
        "--blender", default="blender",
        help="Path to the Blender executable.  "
             r'Example (Windows): "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"'
    )
    p.add_argument(
        "--pipeline_script",
        default=str(Path(__file__).resolve().parent / "blender_pipeline.py"),
        help="Path to blender_pipeline.py (default: same directory as this script)."
    )
    p.add_argument(
        "--input_dir",
        default=str(Path(__file__).resolve().parent / "accepted_characters"),
        help="Directory containing .glb models."
    )
    p.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parent / "dataset"),
        help="Root output directory for rendered pairs."
    )

    # ── job control ──────────────────────────────────────────
    p.add_argument(
        "--variants", type=int, default=DEFAULT_VARIANTS,
        help="Number of render variants (size×elevation×rotation combos) per model."
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Parallel Blender processes.  Start with 1; raise if your CPU can handle it."
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Process at most this many models from the input directory (0 = all). "
             "Models are picked in sorted order.  Use --sample for random selection."
    )
    p.add_argument(
        "--sample", type=int, default=0, metavar="N",
        help="Randomly pick N models from the input directory and process only those. "
             "Useful for a quick smoke test before a full run.  "
             "Overrides --limit when both are set."
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible variant sampling and --sample selection."
    )
    p.add_argument(
        "--timeout", type=int, default=300,
        help="Seconds to wait for a single Blender render job before killing it."
    )

    # ── testing / inspection ─────────────────────────────────
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print the full job plan (model, size, elevation, rotation) and exit. "
             "Nothing is rendered.  Use this to verify your configuration before "
             "launching an expensive full run."
    )

    # ── pipeline pass-through flags ───────────────────────────
    p.add_argument(
        "--sample_animations", action="store_true",
        help="Pass --sample_animations to blender_pipeline.py.  "
             "Renders extra pairs from animation frames embedded in each GLB."
    )
    p.add_argument(
        "--anim_frames", type=int, default=4,
        help="Frames sampled per embedded action (passed to blender_pipeline.py)."
    )
    p.add_argument(
        "--model_height", type=float, default=1.8,
        help="Target model height in Blender units (passed to blender_pipeline.py)."
    )
    p.add_argument(
        "--normal_levels", type=int, default=6,
        help="Normal map quantisation levels (passed to blender_pipeline.py)."
    )
    p.add_argument(
        "--min_fill_ratio", type=float, default=0.04,
        help="Minimum opaque-pixel fraction to keep a rendered pair (0-1). "
             "The original default of 0.10 (10%%) discards many valid sprites "
             "whose limbs are spread or which are viewed at steep elevations. "
             "0.04 (4%%) is a safer threshold that still rejects genuinely blank "
             "renders while keeping usable sprites. "
             "(Passed through to blender_pipeline.py --min_fill_ratio.)"
    )
    p.add_argument(
        "--outline_prob", type=float, default=0.20,
        help="Probability (0–1) that a variant has a 1-pixel black pixel-art "
             "outline added to the color image (normal map is never outlined). "
             "Default 0.20 → ~20%% of images have an outline."
    )

    # ── diagnostics ─────────────────────────────────────────
    p.add_argument(
        "--debug", action="store_true",
        help="Print full Blender stdout/stderr for every render job (very verbose)."
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# MODEL DISCOVERY
# ─────────────────────────────────────────────────────────────

def find_models(input_dir: str, limit: int, sample: int, rng: random.Random):
    """
    Scan input_dir for .glb files.
    --sample N  →  random subset of N (ignores --limit)
    --limit N   →  first N in sorted order
    0           →  all models
    """
    all_models = sorted(Path(input_dir).rglob("*.glb"))
    if not all_models:
        log.error("No .glb files found under %s", input_dir)
        sys.exit(1)

    if sample > 0:
        chosen = rng.sample(all_models, min(sample, len(all_models)))
        log.info("--sample %d: picked %d of %d available models",
                 sample, len(chosen), len(all_models))
        return sorted(chosen)

    if limit > 0:
        chosen = all_models[:limit]
        log.info("--limit %d: using first %d of %d models",
                 limit, len(chosen), len(all_models))
        return chosen

    log.info("Found %d model(s) under %s", len(all_models), input_dir)
    return all_models


# ─────────────────────────────────────────────────────────────
# VARIANT SAMPLING
# ─────────────────────────────────────────────────────────────

def sample_variants(n: int, rng: random.Random,
                    outline_prob: float = 0.20):
    """
    Return `n` variant dicts sampled from the full Cartesian product of
    PIXEL_ART_SIZES × ELEVATION_ANGLES × ROTATION_ANGLES, each extended
    with a randomly chosen use_outline flag.

    outline_prob: probability a variant has a 1-px black outline added to
                  the color image (the normal map is never outlined).

    Both decisions are independent, giving four combinations:
    """
    combos = [
        {"size": s, "elevation": e, "rotation": r}
        for s in PIXEL_ART_SIZES
        for e in ELEVATION_ANGLES
        for r in ROTATION_ANGLES
    ]
    rng.shuffle(combos)

    base = []
    while len(base) < n:
        base.extend(combos)
        rng.shuffle(combos)
    base = base[:n]
    return [{**v, "use_outline": rng.random() < outline_prob} for v in base]


# ─────────────────────────────────────────────────────────────
# DRY RUN
# ─────────────────────────────────────────────────────────────

def print_dry_run(jobs, args):
    """
    Print the full job plan as an aligned table, then exit.
    Lets you sanity-check model selection and variant distribution before
    committing to a potentially long render run.
    """
    total = len(jobs)
    print()
    print("=" * 84)
    print(f"  DRY RUN  —  {total} job(s) planned")
    print(f"  Input  : {args.input_dir}")
    print(f"  Output : {args.out_dir}")
    print(f"  Workers: {args.workers}   Timeout: {args.timeout}s")
    print(f"  Outline prob: {args.outline_prob:.0%}")
    if args.sample_animations:
        print(f"  Animation sampling ON  ({args.anim_frames} frames/action)")
    print("=" * 84)
    print(f"  {'#':>4}  {'Model':<32}  {'Size':>4}  {'Elev':>4}  {'Rot':>3}  {'Shade':<5}  {'Outline'}")
    print("  " + "-" * 78)
    for i, (glb, v) in enumerate(jobs, 1):
        ol    = '✓' if v.get('use_outline', False) else ''
        print(f"  {i:>4}  {glb.name:<32}  {v['size']:>4}  "
              f"{v['elevation']:>4}  {v['rotation']:>3}  {shade:<5}  {ol}")
    print("=" * 84)
    print(f"  Run without --dry_run to start rendering.")
    print()
    sys.exit(0)


# ─────────────────────────────────────────────────────────────
# SINGLE VARIANT RENDER
# ─────────────────────────────────────────────────────────────

def _parse_ok_lines(output: str):
    """
    Parse every PIPELINE_OK:<rgb_path>:<normal_path> line emitted by the
    pipeline.  Returns a list of (rgb_path, normal_path) Path pairs.

    We use the paths that the pipeline actually wrote instead of reconstructing
    them here.  This is essential for animated models: the pipeline appends an
    animation-frame slug (e.g. _anim_Walk_f0143) that the orchestrator cannot
    know in advance, so any path reconstructed from suffix alone will always
    point to a non-existent file and the job will be incorrectly marked FAIL.
    """
    pairs = []
    for line in output.splitlines():
        if not line.startswith("PIPELINE_OK:"):
            continue
        # Format: PIPELINE_OK:<rgb_path>:<normal_path>
        # Paths may contain colons on Windows (drive letters like C:\…).
        # We split on the first two colons after the prefix, but on Windows
        # a path like C:\foo contains a colon at index 1 of the path part.
        # Safe split: strip the prefix, then split into exactly 2 pieces from
        # the right, treating ":<drive>:" correctly by searching left-to-right
        # for the boundary between the two paths.  The normal path always starts
        # at the second top-level-path boundary, which we find by locating the
        # normal_dir substring that the pipeline always writes.
        rest = line[len("PIPELINE_OK:"):]
        # Find the split point: the normal path starts right after the first
        # path ends.  On all platforms the pipeline separates the two paths
        # with a literal ":" that is NOT part of a Windows drive letter — i.e.
        # the colon that follows a full file path (after .png or similar).
        # Simplest robust approach: split on PNG extension boundary.
        import re
        m = re.match(r'^(.+\.png):(.+\.png)$', rest)
        if m:
            pairs.append((Path(m.group(1)), Path(m.group(2))))
        else:
            # Fallback: split on first colon that follows a path separator.
            # Works for Unix paths; for Windows this handles drive letters by
            # splitting on the colon that is NOT preceded by a single letter.
            parts = rest.split(":", 1)
            if len(parts) == 2:
                pairs.append((Path(parts[0]), Path(parts[1])))
    return pairs


def render_variant(
    glb_path: Path,
    variant: dict,
    out_dir: Path,
    args,
):
    """
    Spawn one Blender subprocess for a single (model, variant) pair.
    Returns a result dict for the CSV log.

    FIX (Bug #2): Success detection now parses the actual output file paths
    from PIPELINE_OK lines instead of reconstructing them from suffix alone.
    The pipeline appends animation-frame slugs that the orchestrator cannot
    predict, so the old reconstructed path never matched for animated models
    and every animated render was incorrectly recorded as FAIL.

    Cache detection uses the same PIPELINE_OK-based filenames for consistency;
    for static models the behaviour is unchanged.
    """
    size          = variant["size"]
    elevation     = variant["elevation"]
    rotation      = variant["rotation"]
    use_outline   = variant.get("use_outline", False)
    stem          = glb_path.stem

    # Suffix encodes all variant parameters so every combination has a unique
    # filename and the cache check works correctly.
    suffix = f"_r{rotation:03d}_e{elevation:02d}_s{size:03d}"
    if use_outline:
        suffix += "_ol"

    # Cache check for static (non-animated) renders.
    # For animated models the actual filenames include a frame slug we cannot
    # predict here, so we skip the cache check and let the pipeline decide.
    if not args.sample_animations:
        filename        = f"{stem}{suffix}.png"
        rgb_expected    = out_dir / "color"  / filename
        normal_expected = out_dir / "normal" / filename
        if rgb_expected.exists() and normal_expected.exists():
            log.info("Cache hit: %s%s", stem, suffix)
            return _ok_row(glb_path, stem, suffix, size, elevation, rotation,
                           use_outline, rgb_expected, normal_expected, status="cached")

    cmd = [
        args.blender,
        "--background",
        "--python", args.pipeline_script,
        "--",
        "--model",          str(glb_path),
        "--out_dir",        str(out_dir),
        "--render_size",    str(size),
        "--cam_elevation",  str(elevation),
        "--model_rotation", str(rotation),
        "--normal_levels",  str(args.normal_levels),
        "--model_height",   str(args.model_height),
        "--suffix",         suffix,
        "--min_fill_ratio", str(args.min_fill_ratio),
        "--save_stages",
    ]
    if use_outline:
        cmd.append("--use_outline")
    if args.sample_animations:
        cmd += ["--sample_animations", "--anim_frames", str(args.anim_frames)]

    if args.debug:
        log.info("CMD: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        log.warning("[TIMEOUT] %s%s", stem, suffix)
        return _err_row(glb_path, stem, suffix, size, elevation, rotation,
                        False, "timeout", "timeout")
    except FileNotFoundError:
        raise RuntimeError(
            f"Blender not found at {args.blender!r}.\n"
            "Pass the full path with --blender."
        )

    output = result.stdout.decode("utf-8", errors="replace")

    # Parse the actual output file paths from PIPELINE_OK lines.
    # This is the correct approach: the pipeline knows what it wrote;
    # we verify those exact files exist rather than guessing their names.
    ok_pairs = _parse_ok_lines(output)
    # Keep only pairs where both files actually landed on disk.
    valid_pairs = [(r, n) for r, n in ok_pairs if r.exists() and n.exists()]

    if valid_pairs:
        if args.debug:
            log.info("[OK] %s%s\n%s", stem, suffix, output)
        rgb_path, normal_path = valid_pairs[0]
        log.info("[OK] %-40s  size=%d  elev=%d  rot=%d",
                 (stem + suffix)[:40], size, elevation, rotation)
        return _ok_row(glb_path, stem, suffix, size, elevation, rotation,
                       use_outline, rgb_path, normal_path, status="ok")

    # Failure — print last 20 lines of Blender output
    tail = "\n".join(output.strip().splitlines()[-20:])
    log.warning(
        "[FAIL] %s%s (exit=%d)\n--- Blender output (last 20 lines) ---\n%s\n---",
        stem, suffix, result.returncode, tail
    )
    return _err_row(glb_path, stem, suffix, size, elevation, rotation,
                    use_outline, "failed", f"exit={result.returncode}")


def _ok_row(glb, stem, suffix, size, elev, rot, outline, rgb, nrm, status):
    return {"model": str(glb), "stem": stem, "suffix": suffix,
            "size": size, "elevation": elev, "rotation": rot,
            "use_outline": outline,
            "rgb": str(rgb), "normal": str(nrm),
            "status": status, "error": ""}


def _err_row(glb, stem, suffix, size, elev, rot, outline, status, error):
    return {"model": str(glb), "stem": stem, "suffix": suffix,
            "size": size, "elevation": elev, "rotation": rot,
            "use_outline": outline,
            "rgb": "", "normal": "",
            "status": status, "error": error}


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

CSV_FIELDS = ["model", "stem", "suffix", "size", "elevation", "rotation",
              "use_outline", "rgb", "normal", "status", "error"]


def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    # Resolve paths
    args.pipeline_script = str(Path(args.pipeline_script).resolve())
    args.input_dir       = str(Path(args.input_dir).resolve())
    out_dir              = Path(args.out_dir).resolve()

    if not os.path.exists(args.pipeline_script):
        log.error("blender_pipeline.py not found at: %s", args.pipeline_script)
        sys.exit(1)

    # ── discover models ──
    models = find_models(args.input_dir, args.limit, args.sample, rng)

    # ── build job list ──
    jobs = []
    for glb in models:
        for v in sample_variants(args.variants, rng, args.outline_prob):
            jobs.append((glb, v))

    log.info("%d model(s) x %d variant(s) = %d total jobs",
             len(models), args.variants, len(jobs))

    # ── dry run exits here ──
    if args.dry_run:
        print_dry_run(jobs, args)   # calls sys.exit(0)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV log written row-by-row (survives mid-run crashes) ──
    csv_path = out_dir / "render_log.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    csv_file.flush()
    csv_lock = threading.Lock()

    def write_row(row):
        with csv_lock:
            writer.writerow(row)
            csv_file.flush()

    counters = {"ok": 0, "cached": 0, "failed": 0, "timeout": 0}
    c_lock   = threading.Lock()

    def process_job(job):
        glb_path, variant = job
        row = render_variant(glb_path, variant, out_dir, args)
        write_row(row)
        with c_lock:
            counters[row["status"]] = counters.get(row["status"], 0) + 1
        return row

    # ── execute ──────────────────────────────────────────────
    if args.workers > 1:
        log.info("Running with %d parallel workers", args.workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_job, j) for j in jobs]
            done = 0
            for f in concurrent.futures.as_completed(futures):
                f.result()
                done += 1
                if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                    log.info("Progress: %d / %d", done, len(jobs))
    else:
        for i, job in enumerate(jobs, 1):
            process_job(job)
            if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                log.info("Progress: %d / %d", i, len(jobs))

    csv_file.close()

    # ── summary ──────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Total    : %d", len(jobs))
    log.info("  OK     : %d", counters.get("ok", 0))
    log.info("  Cached : %d", counters.get("cached", 0))
    log.info("  Failed : %d", counters.get("failed", 0))
    log.info("  Timeout: %d", counters.get("timeout", 0))
    log.info("CSV log  : %s", csv_path)
    log.info("Output   : %s", out_dir)


if __name__ == "__main__":
    main()