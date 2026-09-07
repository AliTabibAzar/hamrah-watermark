"""Small image helpers shared by all modules."""

import cv2
import numpy as np


def read_upload(file_bytes: bytes) -> np.ndarray | None:
    """Decodes uploaded bytes to BGR; None for corrupt data."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def ensure_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Forces a mask to binary (0/255) and to the image size."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)


def mask_area_ratio(mask: np.ndarray) -> float:
    white = int((mask > 0).sum())
    return white / max(mask.shape[0] * mask.shape[1], 1)
