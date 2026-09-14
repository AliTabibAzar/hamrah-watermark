"""SLBR/SplitNet baseline eval — Kaggle/server only (needs torch + extern repo).

Compares three detector modes on the SAME dataset roots (images/ + masks/):
  heuristic : current classical pipeline, no torch, runs anywhere
  unet      : our WatermarkUNet weights via src/models/unet_infer.py
  slbr      : external SLBR repo (bcmi) + CLWD-trained weight, mask head only

The SLBR weight is research-licensed and stays OUTSIDE src/ — this script is
only a ruler to tell us what IoU our UNet must reach. Paper numbers (CLWD):
SplitNet IoU 71.96 / SLBR IoU 74.63. Our gate: CLWD IoU>=0.45 + recall>=0.70.

Setup on Kaggle (once):
    git clone https://github.com/bcmi/SLBR-Visible-Watermark-Removal.git extern/slbr
    # gdown the slbr-clwd weight, see: python scripts/download_models.py --only slbr-clwd-gdrive

Usage:
    python scripts/eval_slbr_baseline.py --data data/clwd_test --mode heuristic --limit 200
    python scripts/eval_slbr_baseline.py --data data/clwd_test --mode unet --weights models/watermark-unet.pt --limit 200
    python scripts/eval_slbr_baseline.py --data data/clwd_test --mode slbr --slbr-weights models/slbr_clwd.pth --slbr-repo extern/slbr --limit 200
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)


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


_SLBR_CACHE = {}


def _slbr_model(weights, repo):
    """Loads the external SLBR machine once. Raises with a clear message."""
    import torch

    key = (os.path.abspath(weights), os.path.abspath(repo))
    if key in _SLBR_CACHE:
        return _SLBR_CACHE[key]
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"SLBR weights missing: {weights} — "
            "see `python scripts/download_models.py --only slbr-clwd-gdrive`")
    if not os.path.isdir(repo):
        raise FileNotFoundError(
            f"SLBR repo missing: {repo} — "
            "git clone https://github.com/bcmi/SLBR-Visible-Watermark-Removal.git " + repo)
    sys.path.insert(0, os.path.abspath(repo))
    try:
        import src.models as slbr_models  # noqa: E402
        from options import Options  # noqa: E402
    except ImportError as e:
        raise ImportError(f"cannot import SLBR repo at {repo}: {e}") from e
    import argparse as _ap
    args = Options().init(_ap.ArgumentParser()).parse_args(args=[])
    args.models = "slbr"
    args.crop_size = getattr(args, "crop_size", 256)
    machine = slbr_models.__dict__[args.models](datasets=(None, None), args=args)
    state = torch.load(weights, map_location=machine.device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    machine.model.load_state_dict(state, strict=False)
    machine.model.eval()
    _SLBR_CACHE[key] = machine
    return machine


def predict_slbr_mask(img_bgr, weights, repo, threshold=0.5):
    """Returns a uint8 0/255 mask in the ORIGINAL image size."""
    import torch
    import torch.nn.functional as F

    machine = _slbr_model(weights, repo)
    h, w = img_bgr.shape[:2]
    crop = int(getattr(machine.args, "crop_size", 256))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb.transpose(2, 0, 1)[None])
    t = F.interpolate(t, size=(crop, crop), mode="bilinear", align_corners=False)
    with torch.no_grad():
        out = machine.model(t.to(machine.device).float())
    mask = out[1][0]  # (imoutput, immask_all, imwatermark) -> mask batch
    if mask.dim() == 4:
        mask = mask[0]
    prob = torch.sigmoid(mask).float().cpu().numpy()
    if prob.ndim == 3:
        prob = prob[0]
    back = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return (back > threshold).astype(np.uint8) * 255


def predict(mode, img, unet_cache, args, threshold):
    if mode == "heuristic":
        from src.detection.detector import heuristic_mask
        from src.mask.mask_refinement import refine_mask
        raw, _ = heuristic_mask(img)
        return refine_mask(raw, dilate_iter=2, max_object_ratio=0.5)
    if mode == "unet":
        from src.mask.mask_refinement import refine_mask
        from src.models.unet_infer import unet_mask
        raw = unet_mask(img, threshold=threshold,
                        model=unet_cache.get("model"),
                        device=unet_cache.get("device", "cpu"))
        return refine_mask(raw, dilate_iter=1)
    if mode == "slbr":
        from src.mask.mask_refinement import refine_mask
        raw = predict_slbr_mask(img, args.slbr_weights, args.slbr_repo, threshold)
        return refine_mask(raw, dilate_iter=1)
    raise ValueError(f"unknown mode {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--mode", default="heuristic", choices=["heuristic", "unet", "slbr"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5])
    ap.add_argument("--weights", default="models/watermark-unet.pt")
    ap.add_argument("--slbr-weights", default="models/slbr_clwd.pth")
    ap.add_argument("--slbr-repo", default="extern/slbr")
    ap.add_argument("--fail-dir", default="results/baseline_fails")
    args = ap.parse_args()

    unet_cache = {}
    if args.mode == "unet":
        try:
            import torch

            from src.models.unet_infer import load_unet_weights
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model, dev = load_unet_weights(args.weights, dev)
            unet_cache = {"model": model, "device": dev}
            print(f"unet on {dev}")
        except Exception as e:  # noqa: BLE001 — missing torch/weights is normal locally
            raise SystemExit(f"ERROR: unet mode needs weights+torch: {e}")

    for thr in args.thresholds:
        print(f"\n=== mode={args.mode} threshold={thr} ===")
        for root in args.data:
            pairs = load_pairs(root)
            if args.limit:
                pairs = pairs[:args.limit]
            if not pairs:
                print(f"{root}: no pairs, skipped")
                continue
            ious, recs, fails = [], [], []
            for ipath, mpath in pairs:
                img = cv2.imread(ipath)
                gt = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                if img is None or gt is None:
                    continue
                gt = cv2.resize(gt, (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
                _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
                try:
                    pred = predict(args.mode, img, unet_cache, args, thr)
                except Exception as e:  # noqa: BLE001 — eval must not die per file
                    print(f"  predict failed on {ipath}: {e}")
                    break
                v, r = iou(pred, gt), recall(pred, gt)
                ious.append(v)
                recs.append(r)
                fails.append((v, ipath, pred, gt, img))
            if not ious:
                continue
            mi = sum(ious) / len(ious)
            mr = sum(recs) / len(recs)
            gate = "PASS" if (mi >= 0.45 and mr >= 0.70) else "FAIL"
            print(f"{root}: n={len(ious)} meanIoU={mi:.3f} (paper SLBR 0.746) "
                  f"meanRecall={mr:.3f} GATE-{gate}")
            fails.sort(key=lambda t: t[0])
            os.makedirs(args.fail_dir, exist_ok=True)
            for rank, (v, ipath, pred, gt, img) in enumerate(fails[:10]):
                pred3 = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
                gt3 = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
                canvas = np.hstack([cv2.resize(x, (256, 256)) for x in (img, pred3, gt3)])
                cv2.imwrite(os.path.join(args.fail_dir,
                                         f"{args.mode}_thr{thr}_fail{rank:02d}_{v:.2f}.jpg"),
                            canvas)
            print(f"worst 10 fails -> {args.fail_dir}")


if __name__ == "__main__":
    main()
