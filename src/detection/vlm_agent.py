"""Agentic VLM detector — proposes boxes, tightens, validates, falls back.

The agent never crashes the pipeline: bad output, missing weights, or an
area-ratio violation all degrade gracefully to the heuristic detector.
Output contract (stable for the rest of the pipeline):
{"boxes": [{"x1","y1","x2","y2"}], "type": str, "confidence": float, "note": str}
"""

import numpy as np

from src.detection.detector import detect as heuristic_detect
from src.detection.florence_backend import BaseVLMBackend, MockBackend
from src.mask.mask_refinement import refine_mask
from src.utils.image_utils import mask_area_ratio
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_PROMPTS = [
    "watermark",
    "logo",
    "semi-transparent text overlay",
    "copyright text",
]

CONTRACT_KEYS = ("boxes", "type", "confidence", "note")


def _boxes_to_mask(boxes: list[dict], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def _classify(prompts_hit: list[str]) -> str:
    joined = " ".join(prompts_hit).lower()
    if "logo" in joined:
        return "logo"
    if "text" in joined or "copyright" in joined:
        return "text"
    return "mixed"


def detect_with_agent(
    image_bgr: np.ndarray,
    backend: BaseVLMBackend | None = None,
    prompts: list[str] | None = None,
    settings: dict | None = None,
    max_refine_rounds: int = 2,
) -> tuple[np.ndarray, dict]:
    """Runs the agentic loop. Returns (mask, contract dict). Never raises."""
    import cv2

    h, w = image_bgr.shape[:2]
    prompts = prompts or list(DEFAULT_PROMPTS)
    cfg = (settings or {}).get("detection", {})
    max_ratio = cfg.get("max_mask_area_ratio", 0.45)
    backend = backend or MockBackend()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def fallback(note: str) -> tuple[np.ndarray, dict]:
        mask, msg = heuristic_detect(image_bgr, mode="heuristic")
        return mask, {"boxes": [], "type": "none",
                      "confidence": 0.0, "note": f"{note} | {msg}"}

    try:
        proposals = backend.propose(rgb, prompts)
    except Exception as e:
        log.warning("VLM propose failed: %s", e)
        return fallback("VLM unavailable")
    if not proposals:
        return fallback("VLM found nothing")

    # Refine rounds: tighten each box (max 2, cheap by design).
    boxes = proposals
    for _ in range(max(0, max_refine_rounds)):
        try:
            boxes = [{"box": backend.refine_box(rgb, b["box"]), "label": b["label"]}
                     for b in boxes]
        except Exception as e:
            log.warning("VLM refine failed, keeping coarse boxes: %s", e)
            break

    mask = _boxes_to_mask(boxes, h, w)
    mask = refine_mask(mask)
    ratio = mask_area_ratio(mask)
    if ratio > max_ratio or (mask > 0).sum() == 0:
        return fallback(f"VLM mask rejected (ratio={ratio:.3f})")

    out = {
        "boxes": [{"x1": int(b["box"][0]), "y1": int(b["box"][1]),
                   "x2": int(b["box"][2]), "y2": int(b["box"][3])} for b in boxes],
        "type": _classify([b["label"] for b in boxes]),
        "confidence": round(min(0.9, 0.4 + 0.1 * len(boxes)), 2),
        "note": f"Florence agent, {len(boxes)} region(s), ratio={ratio:.3f}",
    }
    assert set(out) == set(CONTRACT_KEYS)
    return mask, out
