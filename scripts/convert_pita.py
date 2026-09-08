"""PITA snapshot -> images/+masks training layout.

The snapshot holds split zips (train/val/test, plain + yolo variants).
This script unzips them and produces one flat dataset:
  - real mask files (*mask*.png) are used as-is when present
  - otherwise YOLO label files (*.txt, class cx cy w h normalized) are
    rasterized into filled-box masks (coarse but trainable)
Splits become filename prefixes (train__ / val__ / test__), so nothing leaks
across splits silently.

Usage:
    python scripts/convert_pita.py --src data/pita_raw --dst data/pita
    # -> data/pita/images/*.jpg + data/pita/masks/*.png
"""

import argparse
import glob
import os
import shutil
import zipfile

from PIL import Image, ImageDraw

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _unzip_all(src: str, work: str) -> list[str]:    zips = sorted(glob.glob(os.path.join(src, "**", "*.zip"), recursive=True))
    out_dirs = []
    for z in zips:
        name = os.path.splitext(os.path.basename(z))[0]  # train / val / test-yolo ...
        dst = os.path.join(work, name)
        if not os.path.exists(dst):
            with zipfile.ZipFile(z) as f:
                f.extractall(dst)
        out_dirs.append((name, dst))
    return out_dirs


def _find_images(root: str) -> dict[str, str]:
    found = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS) and "mask" not in f.lower():
                found[os.path.splitext(f)[0]] = os.path.join(dirpath, f)
    return found


def _find_masks(root: str) -> dict[str, str]:
    found = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS) and "mask" in f.lower():
                key = os.path.splitext(f)[0].replace("_mask", "").replace("mask_", "")
                found[key] = os.path.join(dirpath, f)
    return found


def _find_yolo(root: str) -> dict[str, str]:
    found = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".txt"):
                found[os.path.splitext(f)[0]] = os.path.join(dirpath, f)
    return found


def _rasterize_yolo(txt_path: str, w: int, h: int) -> Image.Image:
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = parts[:5]
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            d.rectangle([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], fill=255)
    return mask


def _inspect(src: str, work: str):
    """Dumps the unzipped layout: dir tree + sample filenames per folder."""
    for name, root in _unzip_all(src, work):
        print(f"== split: {name} ==")
        for dirpath, _, files in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            if rel.count(os.sep) > 2:
                continue
            sample = ", ".join(sorted(files)[:4])
            print(f"  [{rel}] {len(files)} files e.g. {sample}")
    print("inspect done — paste this output to the developer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="snapshot dir (with data/*.zip)")
    ap.add_argument("--dst", required=True, help="output dataset root")
    ap.add_argument("--inspect", action="store_true",
                    help="only dump the unzipped layout, convert nothing")
    args = ap.parse_args()

    img_dir = os.path.join(args.dst, "images")
    mask_dir = os.path.join(args.dst, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    work = os.path.join(args.dst, "_unzipped")
    os.makedirs(work, exist_ok=True)
    if args.inspect:
        _inspect(args.src, work)
        return

    total, from_mask, from_yolo, skipped = 0, 0, 0, 0
    # group plain + yolo variants per split, prefer plain (real masks win)
    groups: dict[str, list] = {}
    for name, root in _unzip_all(args.src, work):
        groups.setdefault(name.replace("-yolo", ""), []).append((name, root))
    splits = [(base, sorted(v, key=lambda t: ("yolo" in t[0]))[0][1])
              for base, v in sorted(groups.items())]
    for split, root in splits:
        images = _find_images(root)
        masks = _find_masks(root)
        yolos = _find_yolo(root)
        for key, ipath in sorted(images.items()):
            out_name = f"{split}__{os.path.basename(key)}"
            try:
                img = Image.open(ipath).convert("RGB")
            except OSError:
                skipped += 1
                continue
            w, h = img.size
            mask = None
            if key in masks:
                try:
                    mask = Image.open(masks[key]).convert("L").resize((w, h))
                    from_mask += 1
                except OSError:
                    mask = None
            if mask is None and key in yolos:
                mask = _rasterize_yolo(yolos[key], w, h)
                from_yolo += 1
            if mask is None:
                skipped += 1
                continue
            img.save(os.path.join(img_dir, out_name + ".jpg"), quality=92)
            mask.save(os.path.join(mask_dir, out_name + ".png"))
            total += 1
    print(f"pairs={total} from_mask={from_mask} from_yolo={from_yolo} skipped={skipped} -> {args.dst}")
    if total == 0:
        raise SystemExit(f"ERROR: no usable pairs under {args.src}")


if __name__ == "__main__":
    main()
