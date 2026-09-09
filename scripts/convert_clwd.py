"""CLWD split -> images/+masks layout by matching filename stems.

Expected input (after unrar of the needed split only):
    <root>/Mask/<id>.png
    <root>/Watermarked_image/<id>.jpg
    <root>/Watermark_free_image/<id>.jpg   (optional, copied alongside)

Output:
    <dst>/images/<id>.jpg + <dst>/masks/<id>.png (+ clean/ if present)
Plus overlay previews for visual verification before any eval run.

Usage:
    python scripts/convert_clwd.py --src data/clwd_raw/CLWD/test --dst data/clwd_test
"""

import argparse
import glob
import os
import shutil

import cv2
import numpy as np


def _by_stem(folder: str, exts: tuple[str, ...]) -> dict[str, str]:
    out = {}
    for p in glob.glob(os.path.join(folder, "*")):
        if p.lower().endswith(exts) and os.path.isfile(p):
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="CLWD split dir (Mask/ + Watermarked_image/)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--preview", type=int, default=4)
    args = ap.parse_args()

    img_dir = os.path.join(args.dst, "images")
    mask_dir = os.path.join(args.dst, "masks")
    prev_dir = os.path.join("results", "clwd_check")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    masks = _by_stem(os.path.join(args.src, "Mask"), (".png", ".jpg"))
    marked = _by_stem(os.path.join(args.src, "Watermarked_image"), (".png", ".jpg", ".jpeg"))
    clean_dir = os.path.join(args.src, "Watermark_free_image")
    cleans = _by_stem(clean_dir, (".png", ".jpg", ".jpeg")) if os.path.exists(clean_dir) else {}
    clean_out = os.path.join(args.dst, "clean") if cleans else None
    if clean_out:
        os.makedirs(clean_out, exist_ok=True)

    common = sorted(set(masks) & set(marked))
    print(f"masks={len(masks)} watermarked={len(marked)} matched={len(common)}")
    done = 0
    for i, stem in enumerate(common):
        img = cv2.imread(marked[stem])
        mask = cv2.imread(masks[stem], cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(mask_dir, stem + ".png"), mask)
        if stem in cleans:
            shutil.copyfile(cleans[stem], os.path.join(clean_out, stem + ".jpg"))
        if i < args.preview:
            vis = img.copy()
            vis[mask > 0] = (0, 0, 255)
            cv2.imwrite(os.path.join(prev_dir, f"check_{i:02d}.jpg"), vis)
        done += 1
    print(f"pairs={done} -> {args.dst} (+{min(args.preview, done)} previews in {prev_dir})")
    if done == 0:
        raise SystemExit(f"ERROR: no stem matches under {args.src}")


if __name__ == "__main__":
    main()
