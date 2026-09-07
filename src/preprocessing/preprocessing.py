"""Preprocessing — the original image is never modified.

The model sees a 512px copy, but the mask is mapped back to the
original size so output quality is preserved.
"""

import cv2
import numpy as np


def to_model_input(image_bgr: np.ndarray, size: int = 512) -> tuple[np.ndarray, dict]:
    """Returns (model-ready RGB image, metadata for mapping the mask back)."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return small, {"orig_h": h, "orig_w": w, "model_size": size}


def map_mask_to_original(mask_small: np.ndarray, meta: dict) -> np.ndarray:
    """Maps a model-size mask back to the original image size."""
    back = cv2.resize(mask_small, (meta["orig_w"], meta["orig_h"]),
                      interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(back, 127, 255, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)
