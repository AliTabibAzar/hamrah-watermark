"""Input validation — runs before the image reaches any model.

Contract: bytes + filename in, (ok, message, image) out.
Messages are UI-ready English. Never raises on bad input.
"""

import numpy as np

from src.utils.config_loader import load_settings
from src.utils.image_utils import read_upload


def validate_upload(
    file_bytes: bytes, filename: str, settings: dict | None = None
) -> tuple[bool, str, np.ndarray | None]:
    cfg = (settings or load_settings()).get("validation", {})
    app_cfg = (settings or load_settings()).get("app", {})

    if not file_bytes:
        return False, "No file selected.", None

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in app_cfg.get("supported_formats", ["jpg", "jpeg", "png", "webp"]):
        return False, "Unsupported format. Please upload jpg, png, or webp.", None

    max_mb = app_cfg.get("max_upload_mb", 15)
    if len(file_bytes) > max_mb * 1024 * 1024:
        return False, f"File exceeds {max_mb} MB. Please upload a smaller image.", None

    img = read_upload(file_bytes)
    if img is None:
        return False, "This file is corrupt or not an image.", None

    h, w = img.shape[:2]
    if w < cfg.get("min_width", 128) or h < cfg.get("min_height", 128):
        return False, "Image is too small — clean restoration is not possible.", None
    if w > cfg.get("max_width", 6000) or h > cfg.get("max_height", 6000):
        return False, "Image is too large. Please downscale it first.", None

    return True, "OK.", img
