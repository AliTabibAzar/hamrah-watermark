"""Label audit — which YOLO interpretation actually matches the watermarks?

Draws every label file three ways on its image (if present):
  green  standard YOLO  (cx cy w h normalized, top-left origin)
  yellow y-flipped      (cx 1-cy w h — bottom-left origin suspicion)
  magenta corner-format (x1 y1 x2 y2 normalized)
Writes a contact sheet + per-sample overlays. A human decides in one glance
whether the offset is systematic (fixable) or random (drop the dataset).

Usage:
    python scripts/audit_labels.py --labels data/pita/_unzipped --images data/pita/images --out results/audit --n 12
Any <labels> tree with *.txt + any <images> tree works; matched by stem.
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np


def read_labels(txt_path):
    boxes = []
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                boxes.append([float(v) for v in parts[1:5]])
            except ValueError:
                continue
    return boxes


def interpret_standard(b, w, h):
    cx, cy, bw, bh = b
    return int((cx - bw / 2) * w), int((cy - bh / 2) * h), \
        int((cx + bw / 2) * w), int((cy + bh / 2) * h)


def interpret_yflip(b, w, h):
    cx, cy, bw, bh = b
    cy = 1.0 - cy
    return int((cx - bw / 2) * w), int((cy - bh / 2) * h), \
        int((cx + bw / 2) * w), int((cy + bh / 2) * h)


def interpret_corner(b, w, h):
    x1, y1, x2, y2 = b
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


INTERPS = [
    ("standard", (0, 255, 0), interpret_standard),
    ("yflip", (0, 255, 255), interpret_yflip),
    ("corner", (255, 0, 255), interpret_corner),
]


def overlay(img, boxes_list):
    vis = img.copy()
    for color, boxes in boxes_list:
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="tree with *.txt label files")
    ap.add_argument("--images", required=True, help="tree with images, matched by stem")
    ap.add_argument("--out", default="results/audit")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    txts = glob.glob(os.path.join(args.labels, "**", "*.txt"), recursive=True)
    imgs = {}
    for p in glob.glob(os.path.join(args.images, "**", "*"), recursive=True):
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            imgs[os.path.splitext(os.path.basename(p))[0]] = p
    rng = random.Random(args.seed)
    rng.shuffle(txts)
    os.makedirs(args.out, exist_ok=True)

    done, thumbs = 0, []
    for t in txts:
        stem = os.path.splitext(os.path.basename(t))[0]
        if stem not in imgs:
            continue
        img = cv2.imread(imgs[stem])
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = read_labels(t)
        drawn = []
        for name, color, fn in INTERPS:
            try:
                drawn.append((color, [fn(b, w, h) for b in boxes]))
            except Exception:  # noqa: BLE001 — one bad interp must not kill audit
                pass
        vis = overlay(img, drawn)
        cv2.putText(vis, stem, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out = os.path.join(args.out, f"audit_{done:02d}.jpg")
        cv2.imwrite(out, vis)
        thumbs.append(cv2.resize(vis, (320, 240)))
        done += 1
        if done >= args.n:
            break
    if thumbs:
        cols = 4
        rows = (len(thumbs) + cols - 1) // cols
        pad = [np.zeros_like(thumbs[0])] * (rows * cols - len(thumbs))
        sheet_rows = []
        for i in range(rows):
            row = thumbs[i * cols:(i + 1) * cols] + pad[:max(0, cols - len(thumbs[i * cols:(i + 1) * cols]))]
            sheet_rows.append(np.hstack(row))
        cv2.imwrite(os.path.join(args.out, "contact_sheet.jpg"), np.vstack(sheet_rows))
    print(f"audited={done} -> {args.out} (green=standard yellow=yflip magenta=corner)")
    print("legend: whichever color sits ON the watermark across samples wins")
    if done == 0:
        raise SystemExit("ERROR: no label/image stem matches found")


if __name__ == "__main__":
    main()
