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
    dilate_iter: int = 0,
    max_object_ratio: float = 1.0,
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
    # watermark pieces are compact; giant blobs are sky/water, specks are noise.
    # size filter runs BEFORE dilation (dilation merges strokes into big blobs)
    max_px = int(m.shape[0] * m.shape[1] * max_object_ratio)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_object_px <= area <= max_px:
            out[labels == i] = 255
    if dilate_iter > 0:
        # thin strokes must be fully covered — even 1px of residue stays
        # visible after inpainting, so we grow the mask slightly
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        out = cv2.dilate(out, k, iterations=dilate_iter)
    return out
