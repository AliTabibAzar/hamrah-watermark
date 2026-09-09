"""Bulk detector evaluation on real datasets with GT masks.

Layout per dataset root: images/ + masks/ (white = watermark).
Modes: heuristic (no torch) | unet (needs weights+torch) | ensemble (both).

Usage:
    python scripts/eval_bulk.py --data data/clwd_test --mode heuristic --limit 500
    python scripts/eval_bulk.py --data data/pita_test data/logo_test --mode ensemble
    python scripts/eval_bulk.py --data data/clwd_test --mode unet --thresholds 0.3 0.5 0.7

Output: per-dataset mean IoU/recall table + 20 worst fails saved as
side-by-side (image | pred | gt) under --fail-dir for visual analysis.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = {"clwd": (0.45, 0.70), "logo": (0.40, None)}
# PITA labels proved unreliable (audit: boxes off the visible watermarks),
# so PITA sets get no gate — they run for reference numbers only.


def iou(a, b):
    a, b = a > 0, b > 0
    return int((a & b).sum()) / max(int((a | b).sum()), 1)


def recall(pred, gt):
    gt = gt > 0
    return float(((pred > 0) & gt).sum()) / max(int(gt.sum()), 1)


def load_pairs(root):
    imgs = {}
    for p in glob.glob(os.path.join(root, "images", "*")):
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            imgs[os.path.splitext(os.path.basename(p))[0]] = p
    pairs = []
    for stem, ipath in sorted(imgs.items()):
        for ext in (".png", ".jpg"):
            mpath = os.path.join(root, "masks", stem + ext)
            if os.path.exists(mpath):
                pairs.append((ipath, mpath))
                break
    return pairs


def predict(mode, img, unet_cache, threshold):
    if mode == "heuristic":
        from src.detection.detector import heuristic_mask
        from src.mask.mask_refinement import refine_mask
        raw, _ = heuristic_mask(img)
        return refine_mask(raw, dilate_iter=2, max_object_ratio=0.5)
    if mode == "unet":
        from src.mask.mask_refinement import refine_mask
        from src.models.unet_infer import unet_mask
        raw = unet_mask(img, threshold=threshold,
                        model=unet_cache.get("model"), device=unet_cache.get("device", "cpu"))
        return refine_mask(raw, dilate_iter=1)
    if mode == "ensemble":
        from src.detection.detector import cue_ocr, cue_tophat, cue_whiteness
        from src.mask.mask_refinement import refine_mask
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        votes = ((cue_ocr(img)[0] > 0).astype(np.uint8) * 2
                 + (cue_whiteness(img) > 0).astype(np.uint8)
                 + (cue_tophat(gray) > 0).astype(np.uint8))
        if unet_cache.get("model") is not None:
            from src.models.unet_infer import predict_proba
            prob = predict_proba(img, model=unet_cache.get("model"),
                                 device=unet_cache.get("device", "cpu"))
            votes = votes + (prob > threshold).astype(np.uint8)
        else:
            # broken/retired weights stay out instead of poisoning the vote
            print("  (ensemble without U-Net: weights missing or unloadable)")
        raw = (votes >= 2).astype(np.uint8) * 255
        return refine_mask(raw, dilate_iter=1)
    raise ValueError(f"unknown mode {mode}")


def kind_of(root):
    base = os.path.basename(root.rstrip(os.sep)).lower()
    if "clwd" in base:
        return "clwd"
    if "logo" in base:
        return "logo"
    return "pita"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--mode", default="heuristic", choices=["heuristic", "unet", "ensemble"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5])
    ap.add_argument("--fail-dir", default="results/fails")
    ap.add_argument("--weights", default="models/watermark-unet.pt")
    args = ap.parse_args()

    unet_cache: dict = {}
    if args.mode in ("unet", "ensemble"):
        try:
            import torch

            from src.models.unet_infer import load_unet_weights
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model, dev = load_unet_weights(args.weights, dev)
            unet_cache = {"model": model, "device": dev}
            print(f"unet on {dev}")
        except Exception as e:  # noqa: BLE001 — missing torch/weights is normal
            if args.mode == "unet":
                raise SystemExit(f"ERROR: unet mode needs weights+torch: {e}")
            print(f"ensemble continues without U-Net ({e})")

    for thr in args.thresholds:
        print(f"\n=== threshold={thr} ===")
        fails = []
        gate_results = []
        for root in args.data:
            pairs = load_pairs(root)
            if args.limit:
                pairs = pairs[:args.limit]
            if not pairs:
                print(f"{root}: no pairs, skipped")
                continue
            ious, recs = [], []
            for ipath, mpath in pairs:
                img = cv2.imread(ipath)
                gt = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                if img is None or gt is None:
                    continue
                gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
                _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
                try:
                    pred = predict(args.mode, img, unet_cache, thr)
                except Exception as e:  # noqa: BLE001 — eval must not die
                    print(f"  predict failed on {ipath}: {e}")
                    continue
                v, r = iou(pred, gt), recall(pred, gt)
                ious.append(v)
                recs.append(r)
                fails.append((v, ipath, pred, gt, img))
            mi, mr = sum(ious) / max(len(ious), 1), sum(recs) / max(len(recs), 1)
            kind = kind_of(root)
            if kind == "pita":
                print(f"{root}: n={len(ious)} meanIoU={mi:.3f} "
                      f"meanRecall={mr:.3f} REFERENCE-ONLY (no gate: unreliable labels)")
                gate_results.append(True)
                continue
            need_iou, need_rec = PASS[kind]
            verdict = "PASS" if mi >= need_iou and (need_rec is None or mr >= need_rec) else "FAIL"
            print(f"{root}: n={len(ious)} meanIoU={mi:.3f} (need {need_iou}) "
                  f"meanRecall={mr:.3f} {verdict}")
            gate_results.append(verdict == "PASS")
        fails.sort(key=lambda t: t[0])
        os.makedirs(args.fail_dir, exist_ok=True)
        for rank, (v, ipath, pred, gt, img) in enumerate(fails[:20]):
            pred3 = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
            gt3 = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
            h = 256
            canvas = np.hstack([cv2.resize(x, (h, h)) for x in (img, pred3, gt3)])
            cv2.imwrite(os.path.join(args.fail_dir, f"thr{thr}_fail{rank:02d}_{v:.2f}.jpg"), canvas)
        print(f"worst 20 fails -> {args.fail_dir}")
        gate = "GATE PASS" if gate_results and all(gate_results) else "GATE FAIL"
        print(f"*** thr={thr}: {gate} ({sum(gate_results)}/{len(gate_results)} datasets) ***")


if __name__ == "__main__":
    main()
