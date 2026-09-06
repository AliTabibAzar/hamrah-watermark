# Project tasks — Smart Watermark Studio

Deadline: Friday. Two tracks, one shared contract. Tick boxes as you go:
`- [ ]` open, `- [x]` done and merged to `main`.

Tracks: **Detection** (Ali) owns validation → mask. **Removal** (Hossein)
owns removal → UI. Anything marked **shared** needs a quick agreement first.

## How we work

- Branches: `feat/detection` and `feat/removal`, both cut from `main`.
- Short English commits, e.g. `Add Telea inpainting fallback`.
- Open a PR to `main` when a module works; the other person reviews, then merge.
- Never commit `models/*`, `results/*`, or local reference PDFs.
- Light tests must stay green: `pytest tests/ -k "not ai"`.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-base.txt  # light deps only, no torch
streamlit run app.py        # works once the pipeline lands
pytest tests/ -k "not ai"   # light tests must stay green
```

Heavy deps (`requirements-ai.txt`) and weights install on the server only,
which we do not have yet. All AI code must be mock-safe: import and run
locally without torch, falling back gracefully. No model downloads locally.

## Shared contracts (do not break)

1. Images are BGR `numpy` arrays (uint8). Masks are single-channel uint8,
   values 0/255, exactly the image size.
2. Functions return arrays, never file paths. File I/O only in `app.py`.
3. No magic numbers in code — read them from `config/settings.yaml`.
4. No top-level `import torch`. Lazy-load models inside functions with
   try/except and fall back to Telea so light tests stay green.

## Foundation (shared, Sunday)

- [x] Folder structure on `main`, `feat/detection`, `feat/removal`
- [x] README with pipeline overview
- [x] This task list
- [ ] `config/settings.yaml` — all thresholds, sizes, model paths
- [ ] `requirements-base.txt` — streamlit, opencv, pillow, numpy, pyyaml, pytest
- [ ] `requirements-ai.txt` — torch, diffusers, transformers (server only)
- [ ] `src/utils` — logger, config loader, image helpers
- [ ] `src/models/model_manager.py` — lazy loading, device auto-pick, weight checks

## Detection track (Ali, Sun → Tue)

- [ ] `src/validation/image_validator.py` — format, size, corruption checks
      with UI-ready messages; never crashes on bad input
- [ ] `src/preprocessing/preprocessing.py` — RGB convert, resize to model
      size, map mask back to original resolution
- [ ] `src/detection/detector.py` — corner/text heuristic first, U-Net stub
      (`unet_mask` raises until server weights exist)
- [ ] `src/detection/localization.py` — mask to bounding boxes for the UI
- [ ] `src/segmentation/mask_generator.py` — detect + refine in one call
- [ ] `src/mask/mask_refinement.py` — threshold, open/close, speck removal,
      all values from settings
- [ ] Manual mask box input in the UI (brush editor if time allows)
- [ ] `data/test/*` — sample images per category (corner, text,
      transparent, colored, busy, difficult)
- [ ] Tests: validator, preprocessing roundtrip, mask refinement, metrics

## Removal track (Hossein, Mon → Wed)

- [ ] `src/removal/inpainting.py` — `inpaint_telea()` via `cv2.inpaint`,
      plus mock-safe `inpaint()` wrapper with backend flag
      (signature: `(image_bgr, mask, backend, settings) -> (restored, message)`)
- [ ] `src/removal/postprocessing.py` — soft edge blending, no heavy filters
- [ ] `src/watermark/watermark_adder.py` — text + logo overlay with
      position/size/opacity controls, same-size BGR output
- [ ] `app.py` — full English UI flow: upload → validate → mask → restore
      → compare → new watermark → download, with status + error messages
- [ ] `src/evaluation/metrics.py` — PSNR, SSIM, IoU helpers (numpy only)
- [ ] `scripts/download_models.py` — weight links, server only
- [ ] Tests: watermark size preserved, inpaint on empty mask, metrics sanity

## Integration (both, Wednesday)

- [ ] `src/pipeline/watermark_pipeline.py` — validate → mask → inpaint →
      blend, returning a dict the UI renders; never raises to the UI
- [ ] End-to-end manual-mask run works locally (Telea, no weights)
- [ ] All light tests green on a fresh clone

## Evaluation + docs (both, Thursday buffer)

- [ ] `scripts/benchmark.py` — timing per stage (detection, mask,
      inpaint, watermark, total)
- [ ] Honest limitations section in README (large masks, tiny
      semi-transparent text, approximation not restoration)
- [ ] Demo screenshots for the Friday delivery

## Delivery (Friday)

- [ ] Final review of both tracks, merge everything to `main`
- [ ] Tag `v0.1-mvp` on `main`
- [ ] Short demo: upload → clean → new watermark → download

## Backlog (after Friday, only if time)

- [ ] Real U-Net weights + `unet_mask` wiring (needs server)
- [ ] LaMa inference wiring (needs server + weights)
- [ ] Stable Diffusion inpainting option for large masks
- [ ] Before/after slider, brush mask editor, batch mode
- [ ] Smart backend router (fast / balanced / high quality)
- [ ] Full technical report + failure analysis with real outputs

Questions go to Ali. Keep it small, push early, demo Friday.
