import os
import json
import shutil

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

import objaverse


# =========================
# CONFIG
# =========================
MAX_MODELS = 2000
OUTPUT_DIR = "./downloaded_models"


# =========================
# LOAD DATASET
# =========================
def load_objaversepp():
    print("Loading Objaverse++ from Hugging Face...")
    ds = load_dataset("cindyxl/ObjaversePlusPlus", split="train")

    df = ds.to_pandas()

    print(f"Total objects: {len(df)}")
    print("Columns:", df.columns.tolist())

    return df


# =========================
# HELPERS
# =========================
def to_bool(series):
    mapped = series.astype(str).str.lower().map({
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0
    })
    return mapped.fillna(0)  # prevent NaN issues


# =========================
# FILTERS
# =========================
def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize booleans
    df["is_figure"] = to_bool(df["is_figure"])
    df["is_scene"] = to_bool(df["is_scene"])
    df["is_multi_object"] = to_bool(df["is_multi_object"])

    # Normalize strings
    df["style"] = df["style"].astype(str).str.lower().str.strip()
    df["density"] = df["density"].astype(str).str.lower().str.strip()

    print("Initial:", len(df))

    # Apply filters SEQUENTIALLY
    df = df[df["score"] >= 2]
    print("After score:", len(df))

    df = df[df["is_figure"] == 1]
    print("After figure:", len(df))

    df = df[df["is_scene"] == 0]
    print("After scene:", len(df))

    df = df[df["is_multi_object"] == 0]
    print("After multi_object:", len(df))

    df = df[df["style"].str.contains("cartoon|anime", na=False)]
    print("After style:", len(df))

    return df


# =========================
# DOWNLOAD
# =========================
def download_models(uids):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Downloading {len(uids)} models...")

    objects = objaverse.load_objects(
        uids=uids,
        download_processes=8
    )

    print("Moving files to OUTPUT_DIR...")

    uid_to_local_path = {}

    for uid, src_path in tqdm(objects.items()):
        if not os.path.exists(src_path):
            continue

        filename = os.path.basename(src_path)
        dst_path = os.path.join(OUTPUT_DIR, filename)

        try:
            shutil.copy2(src_path, dst_path)
            uid_to_local_path[uid] = dst_path
        except Exception as e:
            print(f"Copy failed for {uid}: {e}")

    return uid_to_local_path


# =========================
# MAIN
# =========================
def main():
    df = load_objaversepp()

    df = filter_dataset(df)

    if df.empty:
        print("No models found after filtering.")
        return

    print("Sampling models...")
    df = df.sample(min(MAX_MODELS, len(df)), random_state=42)

    # Direct column (no guessing)
    uids = df["UID"].tolist()

    mapping = download_models(uids)

    mapping_path = os.path.join(OUTPUT_DIR, "uid_to_path.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Saved mapping to {mapping_path}")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    main()