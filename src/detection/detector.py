"""Watermark detection — heuristic now, U-Net later.

The corner heuristic catches the common logo/text-in-corner case with no
weights. unet_mask() is a mock-safe stub: it raises a clear error until
server weights exist, and detect() turns that into a manual-mode message
instead of a crash.
"""

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


def heuristic_mask(image_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Strong edges in the corners suggest a watermark. Returns (mask, found)."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)

    mask = np.zeros((h, w), dtype=np.uint8)
    ch, cw = h // 4, w // 4
    corners = [(0, 0, cw, ch), (w - cw, 0, w, ch),
               (0, h - ch, cw, h), (w - cw, h - ch, w, h)]
    found = False
    for x1, y1, x2, y2 in corners:
        if (edges[y1:y2, x1:x2] > 0).mean() > 0.02:
            mask[y1:y2, x1:x2] = 255
            found = True
    return mask, found


def unet_mask(image_bgr: np.ndarray, weight_path: str = "models/watermark-unet.pt") -> np.ndarray:
    """Placeholder for the U-Net model until weights are on the server."""
    from pathlib import Path
    if not Path(weight_path).exists():
        raise FileNotFoundError(
            f"U-Net weights missing: {weight_path} — "
            "run scripts/download_models.py on the server first"
        )
    raise NotImplementedError("U-Net inference wiring comes next")


def detect(image_bgr: np.ndarray, mode: str = "heuristic") -> tuple[np.ndarray, str]:
    """Input BGR, output (mask, message). Never raises."""
    try:
        if mode == "auto":
            return unet_mask(image_bgr), "Model generated the mask."
        mask, found = heuristic_mask(image_bgr)
        if found:
            return mask, "Something found in the corners — please check the mask."
        return mask, "No watermark found. Please mark it manually."
    except (FileNotFoundError, NotImplementedError) as e:
        log.warning("Auto detection not ready: %s", e)
        h, w = image_bgr.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), \
            "Auto model not installed on the server yet — continue manually."
