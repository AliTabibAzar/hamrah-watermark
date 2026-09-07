"""Localization — turns a mask into boxes for display in the UI."""

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


def mask_to_boxes(mask: np.ndarray, min_area: int = 200) -> list[tuple[int, int, int, int]]:
    """Returns a list of (x1, y1, x2, y2), smallest boxes dropped."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x + w, y + h))
    return boxes
