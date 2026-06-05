"""
upload_to_hf.py
===============
Publish the paired pixel-art / normal-map dataset to the Hugging Face Hub as a
**Parquet dataset** (the images are embedded into a few Parquet shards).

Why Parquet and not raw folders?
  The Hub rejects any directory with more than 10,000 files, and this dataset
  has ~14.5k pairs. Parquet packs everything into a handful of shards, uploads
  in seconds instead of thousands of tiny commits, and loads in one call:

      from datasets import load_dataset
      ds = load_dataset("DmytroKhitro/pixel-art-normal-maps", split="train")
      ds[0]["color"]   # PIL.Image  (RGBA pixel-art sprite, model INPUT)
      ds[0]["normal"]  # PIL.Image  (camera-space normal map, TARGET)
      ds[0]["size"], ds[0]["model_id"], ds[0]["filename"]

Prerequisites
-------------
    pip install "datasets>=2.14" "huggingface_hub>=0.23" Pillow
    hf auth login          # or pass --token hf_xxx, or set HF_TOKEN

Examples
--------
    python upload_to_hf.py --repo-id DmytroKhitro/pixel-art-normal-maps --dry-run
    python upload_to_hf.py --repo-id DmytroKhitro/pixel-art-normal-maps --recreate
"""

import argparse
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_COLOR = HERE.parent.parent / "Сире" / "dataset_generator" / "new_dataset" / "color"
DEFAULT_NORMAL = HERE.parent.parent / "Сире" / "dataset_generator" / "new_dataset" / "normal"
CARD = HERE / "DATASET_CARD.md"

SIZE_RE = re.compile(r"_s(\d+)")
PREFIX_RE = re.compile(r"^\d{3}-\d{3}_")
POSE_RE = re.compile(r"_r-?\d+_e\d+_s\d+.*$")


def parse_meta(stem: str):
    """(size:int, model_id:str) from a filename stem."""
    m = SIZE_RE.search(stem)
    size = int(m.group(1)) if m else -1
    mid = PREFIX_RE.sub("", stem)
    mid = POSE_RE.sub("", mid)
    return size, mid


def pair(color_dir: Path, normal_dir: Path):
    color = {p.name: p for p in color_dir.glob("*.png")}
    normal = {p.name: p for p in normal_dir.glob("*.png")}
    names = sorted(set(color) & set(normal))
    if not names:
        sys.exit(f"ERROR: no matching filenames.\n  color : {color_dir} ({len(color)})\n"
                 f"  normal: {normal_dir} ({len(normal)})")
    print(f"color images : {len(color)}")
    print(f"normal images: {len(normal)}")
    print(f"paired (both): {len(names)}")
    print(f"unpaired dropped: {len(color) - len(names)} color / {len(normal) - len(names)} normal")
    return names, color, normal


def build_dataset(names, color, normal, limit=None):
    from datasets import Dataset, Features, Image, Value
    if limit:
        names = names[:limit]
    rows = {"color": [], "normal": [], "size": [], "model_id": [], "filename": []}
    for n in names:
        stem = n[:-4]
        size, mid = parse_meta(stem)
        rows["color"].append(str(color[n]))
        rows["normal"].append(str(normal[n]))
        rows["size"].append(size)
        rows["model_id"].append(mid)
        rows["filename"].append(stem)
    feats = Features({
        "color": Image(), "normal": Image(),
        "size": Value("int32"), "model_id": Value("string"), "filename": Value("string"),
    })
    return Dataset.from_dict(rows, features=feats)


def _split_front_matter(text):
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if m:
        return m.group(1), m.group(2)
    return "", text


def merge_card_into_readme(api, repo_id, token):
    """Merge the auto-generated loader YAML (configs/dataset_info) with our card's
    metadata (license/tags) and human-readable body, then upload as README.md."""
    if not CARD.exists():
        return
    try:
        from huggingface_hub import hf_hub_download
        auto = open(hf_hub_download(repo_id, "README.md", repo_type="dataset", token=token),
                    encoding="utf-8").read()
    except Exception:
        auto = ""
    auto_yaml_s, _ = _split_front_matter(auto)
    my_yaml_s, my_body = _split_front_matter(open(CARD, encoding="utf-8").read())
    try:
        import yaml
        auto_y = yaml.safe_load(auto_yaml_s) or {}
        my_y = yaml.safe_load(my_yaml_s) or {}
        merged = {**my_y, **auto_y}          # auto wins for configs/dataset_info; mine adds license/tags
        front = yaml.dump(merged, sort_keys=False, allow_unicode=True).strip()
    except Exception:
        front = (auto_yaml_s or my_yaml_s).strip()
    readme = f"---\n{front}\n---\n\n{my_body.strip()}\n"
    api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset", token=token,
                    commit_message="Add human-readable dataset card")
    print("dataset card merged into README.md")


def main():
    ap = argparse.ArgumentParser(description="Upload the pixel-art normal-map dataset to the HF Hub (Parquet)")
    ap.add_argument("--repo-id", required=True, help="e.g. your-username/pixel-art-normal-maps")
    ap.add_argument("--color-dir", default=str(DEFAULT_COLOR))
    ap.add_argument("--normal-dir", default=str(DEFAULT_NORMAL))
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--private", action="store_true", help="create a private dataset (default: public)")
    ap.add_argument("--recreate", action="store_true",
                    help="delete the repo first for a clean slate (use after a failed upload)")
    ap.add_argument("--limit", type=int, default=None, help="only N pairs (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="report pairing and exit")
    ap.add_argument("--max-shard-size", default="200MB")
    args = ap.parse_args()

    color_dir, normal_dir = Path(args.color_dir), Path(args.normal_dir)
    if not color_dir.is_dir() or not normal_dir.is_dir():
        sys.exit(f"ERROR: dataset folders not found.\n  --color-dir {color_dir}\n  --normal-dir {normal_dir}")

    names, color, normal = pair(color_dir, normal_dir)
    if args.dry_run:
        print(f"\n[dry-run] would publish {len(names)} pairs as Parquet to '{args.repo_id}'. Nothing pushed.")
        return

    try:
        from datasets import Dataset  # noqa
        from huggingface_hub import HfApi, get_token
    except ImportError:
        sys.exit('ERROR: pip install "datasets>=2.14" "huggingface_hub>=0.23" Pillow')

    token = args.token or os.environ.get("HF_TOKEN") or get_token()
    api = HfApi(token=token)
    try:
        print(f"\nAuthenticated as: {api.whoami().get('name', '?')}")
    except Exception as e:
        sys.exit(f"ERROR: not authenticated ({e}). Run `hf auth login` or pass --token hf_xxx.")

    if args.recreate:
        try:
            api.delete_repo(repo_id=args.repo_id, repo_type="dataset", missing_ok=True)
            print("deleted existing repo for a clean slate")
        except Exception as e:
            print(f"(could not delete repo, continuing: {e})")
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    print(f"Building Parquet dataset ({len(names) if not args.limit else args.limit} pairs)...")
    ds = build_dataset(names, color, normal, limit=args.limit)
    print(f"Pushing to https://huggingface.co/datasets/{args.repo_id} ...")
    ds.push_to_hub(args.repo_id, private=args.private, token=token, max_shard_size=args.max_shard_size)
    merge_card_into_readme(api, args.repo_id, token)
    print(f"\nDone -> https://huggingface.co/datasets/{args.repo_id}")
    print('Load with:  from datasets import load_dataset; '
          f'ds = load_dataset("{args.repo_id}", split="train")')


if __name__ == "__main__":
    main()
