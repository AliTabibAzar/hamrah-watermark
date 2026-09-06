# Smart Watermark Studio

AI-powered watermark detection, removal, restoration, and creation.
Upload a watermarked image, get a clean one back, optionally stamp your own watermark on it.

## Pipeline

```
Upload Image
    |
Input Validation          jpg / jpeg / png / webp, size and readability checks
    |
Preprocessing             RGB conversion, model-size resize (mask is mapped
                          back to the original resolution afterwards)
    |
Watermark Detection       does a watermark exist, and of which kind
                          (logo, text, semi-transparent, corner, colored)
    |
Watermark Localization    bounding box / region of the watermark
    |
Mask Generation           white = watermark, black = background
    |
Mask Refinement           threshold, open/close, denoise, hole filling,
                          configurable via config/settings.yaml
    |
Watermark Removal         inpainting: Telea baseline, LaMa main model,
                          Stable Diffusion for large masks (server only)
    |
Post-Processing           soft edge blending, light artifact reduction
    |
Result Comparison         original vs mask vs restored, side by side
    |
New Watermark (optional)  text / logo / image, with position, size,
                          rotation, and opacity controls
    |
Download                  final image export
```

## Folder map

Each pipeline stage owns one folder, so two people can work without conflicts:

| Stage                | Folder               |
| -------------------- | -------------------- |
| Input validation     | `src/validation`     |
| Preprocessing        | `src/preprocessing`  |
| Detection            | `src/detection`      |
| Mask generation      | `src/segmentation`   |
| Mask refinement      | `src/mask`           |
| Removal + cleanup    | `src/removal`        |
| New watermark        | `src/watermark`      |
| Metrics + benchmark  | `src/evaluation`     |
| Model loading        | `src/models`         |
| Shared helpers       | `src/utils`          |
| Full run             | `src/pipeline`       |
| App entry point      | `app.py`             |
| Settings             | `config/settings.yaml` |
| Test images          | `data/test/<category>` |
| Outputs (gitignored) | `results/`           |
| Weights (gitignored) | `models/`            |

## Run (server, Python 3.11)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-base.txt -r requirements-ai.txt
python scripts/download_models.py  # weights, server only
streamlit run app.py
```

## Team branches

- `main` — reviewed code only
- `feat/detection` — validation, preprocessing, detection, mask
- `feat/removal` — removal, watermark, UI, evaluation
