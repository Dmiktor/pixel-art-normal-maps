# Dataset

The full image dataset is **not stored in this repository** — it is published on the
Hugging Face Hub:

> 🤗  https://huggingface.co/datasets/DmytroKhitro/pixel-art-normal-maps

## What it is

Synthetic *(pixel-art sprite → camera-space normal map)* pairs, rendered from low-poly 3D
models. Published as a Parquet dataset with `color` and `normal` image columns (plus `size`, `model_id`, `filename`). The image bytes are embedded, so it loads anywhere.

```
color/   <id>_<hash>_r<rot>_e<elev>_s<size>.png   # RGBA pixel-art sprite  (model INPUT)
normal/  <id>_<hash>_r<rot>_e<elev>_s<size>.png   # camera-space normal map (model TARGET)
```

Filename fields: `r` = model rotation (deg), `e` = camera elevation (deg), `s` = render size (px).

| Property            | Value                                            |
|---------------------|--------------------------------------------------|
| Paired images       | 14,497                                           |
| Unique source models| 648                                            |
| Sprite sizes        | 32, 48, 64, 80, 96 px                            |
| Colour format       | RGBA PNG, hard alpha (transparent background)    |
| Normal encoding     | camera space, `N_rgb = (N + 1) / 2 · 255`        |
| Suggested val split | 10 % (seed 42) → 1,449 images                    |

## How it was generated

See [`../src/dataset_generation/`](../src/dataset_generation). In short: download low-poly
models indexed by Objaverse++, keep clean single-character meshes with CLIP filtering, then
render each model twice in Blender under identical camera/pose — once as a stylised pixel-art
colour sprite, once as the matching normal map.

## Loading example

```python
from datasets import load_dataset

ds = load_dataset("DmytroKhitro/pixel-art-normal-maps", split="train")
ex = ds[0]
color  = ex["color"]    # PIL.Image, RGBA  (model input)
normal = ex["normal"]   # PIL.Image, RGB   (training target)
print(ds)                # 14,497 rows: color, normal, size, model_id, filename
```

## Provenance & license

Rendered from 3D assets indexed by **Objaverse++** (https://github.com/TCXX/ObjaversePlusPlus)
and Objaverse. The rendered dataset is released under **CC BY 4.0**; please also respect the
licenses of the underlying source models. See [`../huggingface/DATASET_CARD.md`](../huggingface/DATASET_CARD.md).
