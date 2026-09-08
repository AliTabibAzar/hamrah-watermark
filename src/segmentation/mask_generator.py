"""Final mask generation — detection plus refinement in one call."""

import numpy as np

from src.detection.detector import detect
from src.mask.mask_refinement import refine_mask


def generate_mask(
    image_bgr: np.ndarray, mode: str = "heuristic", settings: dict | None = None
) -> tuple[np.ndarray, str]:
    raw, msg = detect(image_bgr, mode=mode)
    cfg = (settings or {}).get("mask_refinement", {})
    clean = refine_mask(
        raw,
        threshold=cfg.get("threshold", 127),
        open_k=cfg.get("open_kernel", 3),
        close_k=cfg.get("close_kernel", 5),
        min_object_px=cfg.get("min_object_px", 200),
        dilate_iter=cfg.get("dilate_iter", 0),
    )
    if (clean > 0).sum() == 0:
        return clean, msg + " (Mask is empty — draw it manually.)"
    return clean, msg
