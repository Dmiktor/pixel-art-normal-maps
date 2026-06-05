# Normal-Map Generation for Pixel-Art Images

Learning to convert stylised **pixel-art** sprites into **screen-space normal maps**,
and comparing the learned model against classical normal-map generation methods.

> Master's thesis project (M.Sc., Computer Science / Machine Learning).
> Supporting code and dataset for the thesis *"Normal-map generation for pixelated images"*.

[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/DmytroKhitro/pixel-art-normal-maps)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/Code-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Анотація (UA)

Робота присвячена **генерації мап нормалі** для пікселізованих зображень (піксель-арту)
за допомогою машинного навчання. Оскільки якісного відкритого набору даних «піксельний
малюнок → мапа нормалі» не існує, ми **синтезуємо пари даних з тривимірних моделей з малою
кількістю полігонів**: для кожної моделі виконується дві візуалізації в одних і тих самих
умовах — стилізований піксельний кольоровий спрайт та відповідна мапа нормалі в просторі
камери. На отриманому наборі даних навчається згорткова модель, результат якої порівнюється
зі стандартними методами генерації мап нормалі (Sobel, Beveling) на однаковій валідаційній
вибірці. Навчена модель досягає **середньої кутової похибки 25.6°** проти 35.3° у найкращого
класичного методу (Beveling), тобто помітно перевершує їх.

---

## 1. Problem

Real-time 2D games light flat pixel-art sprites by attaching a **normal map** — an RGB image
that encodes a surface normal vector per pixel. Authoring these maps by hand is slow, and
classical automatic methods (height-from-luminance + Sobel, distance-transform beveling)
produce only a coarse "pseudo-3D" look because they have no notion of the underlying geometry.

This project asks: **can a model learn to predict screen-space normals directly from the
RGBA pixels of a sprite, and does it beat the classical methods?**

The comparison framing follows Moreira et al., *"Comparing methods to generate normal maps
for pixel-art"* ([arXiv:2212.09692](https://arxiv.org/pdf/2212.09692)).

## 2. Method overview

There is no large public *(pixel-art, normal-map)* dataset, so we build a **synthetic** one
from low-poly 3D models and train on it:

```
 Objaverse++ index ──► download low-poly .glb models      (download_characters.py)
        │
        ▼
 CLIP filtering  ──► keep only clean single-character models   (clip_filter_blender.py
        │                                                        + blender_render_single.py)
        ▼
 Blender 2-pass render (identical camera/pose):           (batch_render.py
        │   • pass 1: stylised pixel-art RGBA sprite        + blender_pipeline.py)
        │   • pass 2: matching camera-space normal map
        ▼
 Paired dataset   color/<name>.png  +  normal/<name>.png
        │
        ├──► train CNN  (RGBA → unit normals)              (train_normalmap.py)
        │
        └──► classical baselines  (Sobel, Beveling)        (sobel_baseline.py, bevel_baseline.py)
                       │
                       ▼
            evaluate + compare on the same val split       (eval_model.py, compare_methods.py)
```

Both renders share identical camera, pose and resolution, so `color/` and `normal/` images
pair by filename with no manifest. Normals are stored in **camera space**, RGB-encoded as
`N_rgb = (N + 1) / 2 · 255`.

## 3. Repository structure

```
pixel-art-normal-maps/
├── src/
│   ├── dataset_generation/      # build the synthetic dataset
│   │   ├── download_characters.py     # pull low-poly models via Objaverse++
│   │   ├── clip_filter_blender.py     # CLIP-score & keep clean characters
│   │   ├── blender_render_single.py   # single preview render (for CLIP)
│   │   ├── blender_pipeline.py        # 2-pass render: pixel color + normal map
│   │   ├── batch_render.py            # orchestrate rendering over all models
│   │   └── clean_render_log.py        # drop dangling rows from render_log.csv
│   ├── training/
│   │   ├── train_normalmap.py         # current model (pad-to-96, RGBA, masked loss)
│   │   └── train_normalmap_old.py     # earlier 4-level U-Net (legacy baseline)
│   ├── baselines/
│   │   ├── sobel_baseline.py          # Sobel: height-from-luminance / from-alpha
│   │   └── bevel_baseline.py          # Beveling via distance transform (EDT)
│   └── evaluation/
│       ├── eval_model.py              # masked angular/L1/MAE for the trained model
│       ├── compare_methods.py         # join all methods into one comparison table
│       ├── inference_test_diploma.py  # qualitative figure for the thesis appendix
│       └── plot_metrics.py            # training-curve plots from metrics.csv
├── results/
│   ├── exp5_unet/        metrics.csv + plots/   (legacy U-Net, 120 epochs)
│   ├── exp_balanced/     metrics.csv + plots/   (current model, 80 epochs)
│   └── comparison/       per-image CSVs, summary tables, comparison charts
├── data/
│   └── README.md         # dataset format + link to Hugging Face
├── huggingface/
│   ├── DATASET_CARD.md   # dataset card (README for the HF dataset repo)
│   └── upload_to_hf.py   # pair color/normal and push the dataset to the Hub
├── requirements.txt              # core ML stack (train / eval / baselines / plots)
├── requirements-datagen.txt      # dataset-generation stack (+ external Blender 4.x)
└── LICENSE                       # CC BY-NC 4.0 (code)
```

> Large artifacts are intentionally **not** in git: virtual environments, the 43 GB of
> training checkpoints (`runs/`), the Blender render caches and the full image dataset.
> The dataset is published on Hugging Face; checkpoints can be regenerated by training.

## 4. Dataset

The published dataset contains **14,497 paired images** rendered from **648 unique**
low-poly 3D models, at five sprite sizes (32, 48, 64, 80, 96 px).

Published on the Hugging Face Hub as a **Parquet dataset** (image bytes embedded), loadable
in one call:

```python
from datasets import load_dataset
ds = load_dataset("DmytroKhitro/pixel-art-normal-maps", split="train")
ds[0]["color"]   # PIL.Image, RGBA pixel-art sprite (input)
ds[0]["normal"]  # PIL.Image, camera-space normal map (target)
```

Columns: `color` (RGBA), `normal` (RGB), `size`, `model_id`, `filename`.
A 10 % held-out split (1,449 images, seed 42) is used for all evaluation below.
See [`data/README.md`](data/README.md) and the
[dataset card](huggingface/DATASET_CARD.md) for full details and a loading snippet.

## 5. Results

All methods are evaluated on the **same 1,449-image validation split**, using a **masked**
metric — angular error is averaged over **opaque (silhouette) pixels only**, so the large
flat-background region cannot inflate the scores. Lower is better.

| Method                 | Mean angular error (°) | Angular `1−cos θ` | L1     |
|------------------------|:----------------------:|:-----------------:|:------:|
| Sobel (luminance)      | 49.70                  | 0.425             | 0.407  |
| Sobel (alpha)          | 44.26                  | 0.344             | 0.372  |
| Beveling (EDT)         | 35.28                  | 0.232             | 0.297  |
| **U-Net (this work)**  | **25.62**              | **0.158**         | **0.212** |

Per sprite size (mean angular error, °):

| Size | Sobel-lum | Sobel-α | Bevel | **U-Net** |
|------|:---------:|:-------:|:-----:|:---------:|
| 32px | 53.74 | 48.90 | 37.00 | **32.16** |
| 48px | 51.03 | 44.85 | 34.21 | **26.72** |
| 64px | 49.61 | 43.93 | 35.11 | **24.85** |
| 80px | 47.59 | 42.48 | 35.14 | **23.14** |
| 96px | 46.90 | 41.47 | 34.98 | **21.75** |

The learned model beats every classical baseline at every render size, and its advantage
grows with resolution. Charts: `results/comparison/methods_mae_by_size.png`,
`results/comparison/methods_mae_overall.png`.

### A note on the two training runs

`results/exp5_unet/metrics.csv` (legacy) reports a much lower `val_mae_deg` (~3.5°) than
`results/exp_balanced/metrics.csv` (~20°). **These two numbers are not comparable**: the
legacy run averages angular error over the *whole frame* (including the easy flat background),
while the current run and `eval_model.py` average over the *silhouette only*. The honest,
silhouette-masked comparison is the table above (from `eval_model.py` / `compare_methods.py`).

## 6. Reproduce

```bash
# 0. Environment
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt                     # core ML stack

# 1. (optional) Rebuild the dataset from scratch  -- needs Blender 4.x + the datagen deps
pip install -r requirements-datagen.txt
python src/dataset_generation/download_characters.py
python src/dataset_generation/clip_filter_blender.py --blender /path/to/blender --limit 2000
python src/dataset_generation/batch_render.py        --blender /path/to/blender --variants 8 --workers 3
#    -> produces  new_dataset/color/*.png  and  new_dataset/normal/*.png
#    (or just download the ready dataset from Hugging Face, see data/README.md)

# 2. Train the model
python src/training/train_normalmap.py --color_dir new_dataset/color \
        --normal_dir new_dataset/normal --out runs/exp_balanced

# 3. Evaluate model + classical baselines on the same split (seed 42, 10 % val)
python src/baselines/sobel_baseline.py --color_dir new_dataset/color --normal_dir new_dataset/normal \
        --output_dir results/comparison/sobel_eval --val_split 0.10 --seed 42
python src/baselines/bevel_baseline.py --color_dir new_dataset/color --normal_dir new_dataset/normal \
        --output_dir results/comparison/bevel_eval --val_split 0.10 --seed 42
python src/evaluation/eval_model.py    --color_dir new_dataset/color --normal_dir new_dataset/normal \
        --checkpoint runs/exp_balanced/checkpoints/best.pth \
        --output_dir results/comparison/model_eval --val_split 0.10 --seed 42
python src/evaluation/compare_methods.py

# 4. Plots
python src/evaluation/plot_metrics.py --csv results/exp_balanced/metrics.csv --out results/exp_balanced/plots
```

## 7. References

- **Source 3D models:** Objaverse++ — https://github.com/TCXX/ObjaversePlusPlus
- **Comparison framing:** R. Moreira et al., *Comparing methods to generate normal maps for
  pixel-art*, arXiv:2212.09692 — https://arxiv.org/pdf/2212.09692

## 8. Authors

- **Author (здобувач):** Шаповалов Дмитро Андрійович — 122 «Комп'ютерні науки»,
  спеціалізація «Машинне навчання».
- **Supervisor (науковий керівник):** Шамрай Максим — Ph.D. in Applied Mathematics.

## License

Source code: **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0) — free to
use, share and adapt for **non-commercial** purposes with attribution; commercial use
requires prior written permission from the author. See [LICENSE](LICENSE).
Dataset: **CC BY 4.0**, distributed on Hugging Face; derived from Objaverse++ / Objaverse
assets, which carry their own licenses (see the dataset card for attribution).
