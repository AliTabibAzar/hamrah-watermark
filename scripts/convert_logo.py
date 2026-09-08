"""LOGO zip -> images/+masks eval set via input/target differencing.

Extracts ONLY val_input_256/* (watermarked) + val_target_256/* (clean) from
the 15GB zip — never the whole archive. GT masks come from the difference:
  mask = |input - target| > --diff-thresh, despeckled and hole-filled.
Also writes overlay previews for visual verification before any eval run.

Usage:
    python scripts/convert_logo.py --zip data/logo_dl/10kmid.zip --dst data/logo_eval
    python scripts/convert_logo.py --zip data/logo_dl/10kmid.zip --dst data/logo_eval --diff-thresh 30 --preview 6
"""

import argparse
import os
import zipfile

import cv2
import numpy as np


def _selective_extract(zip_path: str, work: str) -> tuple[str, str]:
    inp_dir = os.path.join(work, "input")
    tgt_dir = os.path.join(work, "target")
    os.makedirs(inp_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)
    n_inp = n_tgt = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename
            if name.endswith("/"):
                continue
            low = name.lower()
            if "val_input_256/" in low and low.endswith((".png", ".jpg", ".jpeg")):
                z.extract(info, inp_dir)
                n_inp += 1
            elif "val_target_256/" in low and low.endswith((".png", ".jpg", ".jpeg")):
                z.extract(info, tgt_dir)
                n_tgt += 1
    # flatten one level if the zip kept its folder prefix inside
    return _flatten(inp_dir), _flatten(tgt_dir), n_inp, n_tgt


def _flatten(d: str) -> str:
    import glob
    import shutil
    subs = [p for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p)]
    while len(subs) == 1 and not glob.glob(os.path.join(d, "*.png")) \
            and not glob.glob(os.path.join(d, "*.jpg")):
        inner = subs[0]
        for f in glob.glob(os.path.join(inner, "*")):
            shutil.move(f, d)
        os.rmdir(inner)
        subs = [p for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p)]
    return d


def _by_stem(d: str) -> dict[str, str]:
    import glob
    out = {}
    for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
        if p.lower().endswith((".png", ".jpg", ".jpeg")) and os.path.isfile(p):
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def diff_mask(watermarked: np.ndarray, clean: np.ndarray, thresh: int) -> np.ndarray:
    g_w = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY).astype(np.int16)
    g_c = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff = np.abs(g_w - g_c).astype(np.uint8)
    _, m = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--diff-thresh", type=int, default=20)
    ap.add_argument("--preview", type=int, default=6)
    args = ap.parse_args()

    img_dir = os.path.join(args.dst, "images")
    mask_dir = os.path.join(args.dst, "masks")
    prev_dir = os.path.join("results", "logo_check")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)
    work = os.path.join(args.dst, "_unzipped")
    os.makedirs(work, exist_ok=True)

    inp_dir, tgt_dir, n_inp, n_tgt = _selective_extract(args.zip, work)
    print(f"extracted input={n_inp} target={n_tgt}")
    inputs, targets = _by_stem(inp_dir), _by_stem(tgt_dir)
    pairs = sorted(set(inputs) & set(targets))
    print(f"matched pairs={len(pairs)} (input-only={len(set(inputs) - set(targets))}, "
          f"target-only={len(set(targets) - set(inputs))})")

    done = 0
    for i, stem in enumerate(pairs):
        wimg = cv2.imread(inputs[stem])
        cimg = cv2.imread(targets[stem])
        if wimg is None or cimg is None:
            continue
        if wimg.shape != cimg.shape:
            cimg = cv2.resize(cimg, (wimg.shape[1], wimg.shape[0]))
        mask = diff_mask(wimg, cimg, args.diff_thresh)
        if (mask > 0).sum() == 0:
            continue  # identical pair, no watermark to learn
        cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), wimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(mask_dir, stem + ".png"), mask)
        if i < args.preview:
            vis = wimg.copy()
            vis[mask > 0] = (0, 0, 255)
            cv2.imwrite(os.path.join(prev_dir, f"check_{i:02d}.jpg"), vis)
        done += 1
    print(f"pairs={done} -> {args.dst} (+{min(args.preview, done)} previews in {prev_dir})")
    if done == 0:
        raise SystemExit("ERROR: no usable pairs")


if __name__ == "__main__":
    main()
