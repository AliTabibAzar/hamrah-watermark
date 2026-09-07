"""Tests for the VLM agent — mock backend only, no weights, no internet."""

import numpy as np

from src.detection.florence_backend import MockBackend
from src.detection.vlm_agent import CONTRACT_KEYS, detect_with_agent


def _fake_bgr(h=200, w=300):
    rng = np.random.default_rng(3)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def test_agent_contract_keys():
    _, out = detect_with_agent(_fake_bgr(), backend=MockBackend())
    assert set(out) == set(CONTRACT_KEYS)
    assert isinstance(out["boxes"], list) and out["boxes"]
    assert 0.0 <= out["confidence"] <= 1.0


def test_agent_mask_matches_image():
    img = _fake_bgr()
    mask, _ = detect_with_agent(img, backend=MockBackend())
    assert mask.shape == img.shape[:2]
    assert set(np.unique(mask)).issubset({0, 255})


def test_agent_falls_back_on_broken_backend():
    class Broken(MockBackend):
        def propose(self, image_rgb, prompts):
            raise RuntimeError("no weights here")

    mask, out = detect_with_agent(_fake_bgr(), backend=Broken())
    assert out["type"] == "none" and out["confidence"] == 0.0
    assert mask.shape == (200, 300)


def test_agent_rejects_giant_mask():
    class Greedy(MockBackend):
        def propose(self, image_rgb, prompts):
            h, w = image_rgb.shape[:2]
            return [{"box": (0, 0, w, h), "label": "watermark"}]

    _, out = detect_with_agent(_fake_bgr(), backend=Greedy())
    assert out["type"] == "none"  # area guard -> heuristic fallback
