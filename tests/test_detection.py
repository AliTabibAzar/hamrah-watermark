"""Light tests for the detection track — no torch needed."""

import numpy as np

from src.detection.detector import detect, heuristic_mask
from src.detection.localization import mask_to_boxes
from src.mask.mask_refinement import refine_mask
from src.preprocessing.preprocessing import map_mask_to_original, to_model_input
from src.utils.image_utils import ensure_mask, mask_area_ratio
from src.validation.image_validator import validate_upload


def _fake_bgr(h=200, w=300):
    rng = np.random.default_rng(7)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def test_validator_rejects_bad_format():
    ok, _, img = validate_upload(b"xxx", "a.txt")
    assert not ok and img is None


def test_validator_rejects_tiny():
    import cv2
    _, buf = cv2.imencode(".jpg", np.zeros((10, 10, 3), dtype=np.uint8))
    ok, _, _ = validate_upload(buf.tobytes(), "tiny.jpg")
    assert not ok


def test_validator_accepts_good_image():
    import cv2
    _, buf = cv2.imencode(".png", _fake_bgr())
    ok, _, img = validate_upload(buf.tobytes(), "good.png")
    assert ok and img is not None


def test_preprocess_roundtrip():
    img = _fake_bgr()
    small, meta = to_model_input(img)
    assert small.shape[:2] == (512, 512)
    fake = np.zeros((512, 512), dtype=np.uint8)
    fake[10:50, 10:50] = 255
    back = map_mask_to_original(fake, meta)
    assert back.shape == img.shape[:2]
    assert back[10, 15] == 255  # inside the scaled-down box
    assert back[150, 250] == 0


def test_heuristic_never_crashes_and_auto_falls_back():
    img = _fake_bgr()
    mask, _ = heuristic_mask(img)
    assert mask.shape == img.shape[:2]
    auto_mask, msg = detect(img, mode="auto")
    assert auto_mask.shape == img.shape[:2]
    assert "manually" in msg


def test_refine_removes_specks():
    raw = np.zeros((100, 100), dtype=np.uint8)
    raw[5, 5] = 255
    raw[40:80, 40:80] = 255
    out = refine_mask(raw, min_object_px=200)
    assert out[5, 5] == 0 and out[60, 60] == 255


def test_boxes_and_ratio():
    m = np.zeros((100, 100), dtype=np.uint8)
    m[10:30, 10:30] = 255
    assert len(mask_to_boxes(m)) == 1
    assert len(mask_to_boxes(np.zeros((100, 100), dtype=np.uint8))) == 0
    assert 0 < mask_area_ratio(m) < 1
    assert ensure_mask(m, 100, 100).shape == (100, 100)
