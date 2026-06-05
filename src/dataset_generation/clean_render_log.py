"""
clean_render_log.py
-------------------
Reads render_log.csv and removes every row whose normal-map file
no longer exists on disk.  Writes a cleaned copy.

Usage (run from your dataset_generator folder):
    python clean_render_log.py
    python clean_render_log.py --input render_log.csv --output render_log_clean.csv

The original file is NOT modified unless you pass --inplace.
"""

import argparse
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Remove render_log rows with missing normal maps")
    p.add_argument("--input",   default="render_log.csv",       help="Source CSV")
    p.add_argument("--output",  default="render_log_clean.csv", help="Destination CSV")
    p.add_argument("--inplace", action="store_true",
                   help="Overwrite the input file instead of writing a new one")
    args = p.parse_args()

    src = Path(args.input)
    dst = src if args.inplace else Path(args.output)

    df = pd.read_csv(src)
    total = len(df)
    print(f"[clean] Loaded {total} rows from {src}")

    ok_mask   = df["status"] == "ok"
    ok_rows   = df[ok_mask]
    other_rows = df[~ok_mask]   # failed / skipped rows — keep as-is

    # Check which normal map files actually exist on disk
    missing_mask = ~ok_rows["normal"].apply(lambda p: Path(p).exists())
    missing      = ok_rows[missing_mask]
    kept         = ok_rows[~missing_mask]

    print(f"[clean] OK rows       : {len(ok_rows)}")
    print(f"[clean] Normal missing: {len(missing)}")
    print(f"[clean] OK rows kept  : {len(kept)}")

    if len(missing) == 0:
        print("[clean] Nothing to remove — CSV is already clean.")
        return

    # Show a sample of what's being removed
    print(f"\n[clean] Sample removed paths ({min(5, len(missing))}/{len(missing)}):")
    for path in missing["normal"].head(5):
        print(f"  {path}")

    # Reassemble: kept ok rows + all non-ok rows, preserve original order
    clean_df = pd.concat([kept, other_rows]).sort_index().reset_index(drop=True)

    clean_df.to_csv(dst, index=False)
    print(f"\n[clean] Wrote {len(clean_df)} rows → {dst}")
    print(f"[clean] Removed {total - len(clean_df)} rows total.")


if __name__ == "__main__":
    main()