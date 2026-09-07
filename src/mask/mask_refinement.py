"""Mask cleanup — drops noise and fills small holes.

All numbers come from settings; nothing is hardcoded here.
"""

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


def refine_mask(
    raw: np.ndarray,
    threshold: int = 127,
    open_k: int = 3,
    close_k: int = 5,
    min_object_px: int = 200,
) -> np.ndarray:
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(raw, threshold, 255, cv2.THRESH_BINARY)

    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_object_px:
            out[labels == i] = 255
    return out
