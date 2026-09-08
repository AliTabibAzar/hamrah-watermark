"""Flatten a dataset snapshot into a backgrounds folder for gen_synthetic.

Walks <src> recursively, copies every image to <dst> with unique flat names.
Layout-agnostic: works for COCO-like trees, HF snapshots, zips-extracted
folders — anything with image files in it.

Usage:
    python scripts/convert_pita.py --src data/pita_raw --dst data/pita_bg
    python scripts/convert_pita.py --src data/pita_raw --dst data/pita_bg --limit 3000 --ext .jpg .png
"""

import argparse
import os
import shutil

DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--ext", nargs="+", default=list(DEFAULT_EXTS))
    args = ap.parse_args()

    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext}
    os.makedirs(args.dst, exist_ok=True)
    found, copied, n = 0, 0, 0
    for dirpath, _, files in os.walk(args.src):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in exts:
                continue
            found += 1
            if args.limit and copied >= args.limit:
                break
            src = os.path.join(dirpath, f)
            # unique flat name: parent dir + file, avoids collisions
            parent = os.path.basename(dirpath.rstrip(os.sep)) or "root"
            stem, ext = os.path.splitext(f)
            dst_name = f"{parent}_{stem}{ext}"
            dst = os.path.join(args.dst, dst_name)
            k = 1
            while os.path.exists(dst):
                dst_name = f"{parent}_{stem}_{k}{ext}"
                dst = os.path.join(args.dst, dst_name)
                k += 1
            shutil.copyfile(src, dst)
            copied += 1
    print(f"found={found} copied={copied} -> {args.dst}")
    if copied == 0:
        raise SystemExit(f"ERROR: no images under {args.src} (exts={sorted(exts)})")


if __name__ == "__main__":
    main()
