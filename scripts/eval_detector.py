"""Local detector evaluation — IoU per sample against hand-made GT.

Usage:
    python scripts/eval_detector.py
For tiled watermarks there is no single box: we check that the mask spreads
over at least 3 of 4 image quadrants instead of an IoU number.
GT boxes use my own visual estimates (0-1000 normalized), good enough to
catch regressions while tuning thresholds — not a published benchmark.
"""

import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detection.detector import heuristic_mask  # noqa: E402
from src.mask.mask_refinement import refine_mask  # noqa: E402

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def to_pixels(boxes, w, h):
    return [(int(x1 * w / 1000), int(y1 * h / 1000),
             int(x2 * w / 1000), int(y2 * h / 1000)) for x1, y1, x2, y2 in boxes]


def gt_mask(shape, boxes):
    m = np.zeros(shape, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        m[y1:y2, x1:x2] = 255
    return m


def iou(a, b):
    a, b = a > 0, b > 0
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / max(union, 1)


def recall(pred, gt):
    gt = gt > 0
    return float(((pred > 0) & gt).sum()) / max(int(gt.sum()), 1)


def quadrant_hits(mask):
    h, w = mask.shape
    quads = [mask[0:h // 2, 0:w // 2], mask[0:h // 2, w // 2:w],
             mask[h // 2:h, 0:w // 2], mask[h // 2:h, w // 2:w]]
    return sum(int((q > 0).mean() > 0.02) for q in quads)


def main():
    with open(os.path.join(BASE, "scripts", "gt_watermarks.json"), encoding="utf-8") as f:
        gt = json.load(f)
    print(f"{'sample':<18}{'type':<14}{'IoU':<8}{'recall':<8}{'ratio':<8}verdict")
    ok = 0
    for path in sorted(glob.glob(os.path.join(BASE, "watermarked_images", "*.jpg"))):
        name = os.path.basename(path)
        img = cv2.imread(path)
        h, w = img.shape[:2]
        raw, _ = heuristic_mask(img)
        mask = refine_mask(raw, dilate_iter=2, max_object_ratio=0.5)
        ratio = float((mask > 0).mean())
        spec = gt[name]
        if spec["type"] == "tiled":
            q = quadrant_hits(mask)
            verdict = "PASS" if q >= 3 else "FAIL"
            print(f"{name:<18}{spec['type']:<14}{'quads=' + str(q):<8}{'':<8}{ratio:<8.3f}{verdict}")
        else:
            g = gt_mask((h, w), to_pixels(spec["boxes_norm_1000"], w, h))
            v, r = iou(mask, g), recall(mask, g)
            verdict = "PASS" if (r >= 0.5 and ratio <= 0.4) else "FAIL"
            print(f"{name:<18}{spec['type']:<14}{v:<8.3f}{r:<8.3f}{ratio:<8.3f}{verdict}")
        ok += verdict == "PASS"
    print(f"\n{ok}/{len(gt)} samples passing (recall>=0.5, ratio<=0.4, or 3+ quadrants)")


if __name__ == "__main__":
    main()
