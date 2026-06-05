---
license: cc-by-4.0
pretty_name: Pixel-Art Normal Maps
task_categories:
  - image-to-image
tags:
  - pixel-art
  - normal-maps
  - normal-map-estimation
  - synthetic
  - computer-graphics
  - game-assets
size_categories:
  - 10K<n<100K
---

# Pixel-Art Normal Maps

Synthetic **(pixel-art sprite → camera-space normal map)** image pairs for training and
evaluating normal-map generation models for 2D pixel-art.

This dataset accompanies a master's thesis on *normal-map generation for pixelated images*.
Code: https://github.com/Dmiktor/pixel-art-normal-maps

## Load

```python
from datasets import load_dataset

ds = load_dataset("DmytroKhitro/pixel-art-normal-maps", split="train")
ex = ds[0]
ex["color"]    # PIL.Image - RGBA pixel-art sprite (model INPUT)
ex["normal"]   # PIL.Image - camera-space normal map (TARGET)
ex["size"], ex["model_id"], ex["filename"]
```

## Columns

| Column     | Type         | Description                                                       |
|------------|--------------|------------------------------------------------------------------|
| `color`    | image (RGBA) | pixel-art sprite, transparent background — model **input**        |
| `normal`   | image (RGB)  | camera-space normal map, `N_rgb = (N + 1) / 2 · 255` — **target** |
| `size`     | int          | render size in px (32, 48, 64, 80 or 96)                          |
| `model_id` | string       | id of the source 3D model the pair was rendered from             |
| `filename` | string       | original render name (`<id>_<model>_r<rot>_e<elev>_s<size>[...]`) |

One `train` split. Stored as Parquet with the image bytes embedded (loads anywhere, no
external files).

| Property             | Value                                          |
|----------------------|------------------------------------------------|
| Paired images        | 14,497                                         |
| Unique source models | 648                                            |
| Sprite sizes         | 32, 48, 64, 80, 96 px                          |
| Colour format        | RGBA, transparent background                   |
| Normal encoding      | camera space, `N_rgb = (N + 1) / 2 · 255`      |
| Suggested val split  | 10 % (seed 42) → 1,449 images                  |

## How it was made

Low-poly 3D models indexed by [Objaverse++](https://github.com/TCXX/ObjaversePlusPlus) are
filtered with CLIP to keep clean single-character meshes, then rendered twice in Blender under
identical camera and pose: a stylised pixel-art colour pass and a matching camera-space normal
pass. Identical framing means the two images align pixel-for-pixel.

## Baseline results (masked angular error, validation = 1,449 images)

| Method            | Mean angular error (°) |
|-------------------|:----------------------:|
| Sobel (luminance) | 49.70 |
| Sobel (alpha)     | 44.26 |
| Beveling (EDT)    | 35.28 |
| U-Net (this work) | **25.62** |

## Citation

```bibtex
@misc{shapovalov2026pixelnormalmaps,
  author = {Shapovalov, Dmytro},
  title  = {Pixel-Art Normal Maps: a synthetic dataset for normal-map generation},
  year   = {2026},
  howpublished = {Hugging Face Hub},
  note   = {Derived from Objaverse++ / Objaverse assets}
}
```

## License & attribution

Released under **CC BY 4.0**. Rendered from Objaverse++ / Objaverse 3D assets, which carry
their own licenses — please attribute Objaverse++ and respect upstream model licenses.
