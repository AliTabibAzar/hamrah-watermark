"""Synthetic watermark generator — unlimited training pairs with free GT.

Takes clean background photos, stamps random watermarks (text / logo shapes /
bands / tiles) with random position, opacity, size, and rotation, and saves
the image plus the exact pixel mask. Only Pillow + numpy, runs anywhere.

Usage:
    python scripts/gen_synthetic.py --bg watermarked_images --out data/synth --n 2000
NOTE: use only CLEAN backgrounds ideally; the script still works otherwise.
"""

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEXTS = ["PROOF", "Copyright", "SAMPLE", "DRAFT", "Photo Studio",
         "DO NOT COPY", "PREVIEW", "Studio X", "© 2026", "Watermark"]


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _stamp_text(base: Image.Image, mask_layer: Image.Image, rng: random.Random):
    w, h = base.size
    txt = rng.choice(TEXTS)
    fs = rng.randint(max(12, w // 24), max(24, w // 6))
    font = _font(fs)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    bbox = d.textbbox((0, 0), txt, font=font)
    tw, th = bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8
    tw = min(tw, w - 4)
    x, y = rng.randint(0, max(0, w - tw)), rng.randint(0, max(0, h - th))
    op = rng.randint(int(255 * 0.2), int(255 * 0.7))
    d.text((x, y), txt, font=font, fill=(255, 255, 255, op))
    if rng.random() < 0.4:
        layer = layer.rotate(rng.uniform(-25, 25), resample=Image.BILINEAR,
                             center=(x + tw / 2, y + th / 2))
    alpha = layer.split()[-1].point(lambda v: 255 if v > 10 else 0)
    mask_layer = Image.composite(Image.new("L", base.size, 255), mask_layer, alpha)
    return Image.alpha_composite(base, layer), mask_layer


def _stamp_shape(base: Image.Image, mask_layer: Image.Image, rng: random.Random):
    w, h = base.size
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rw, rh = rng.randint(w // 12, w // 4), rng.randint(h // 12, h // 4)
    x, y = rng.randint(0, max(0, w - rw)), rng.randint(0, max(0, h - rh))
    op = rng.randint(int(255 * 0.2), int(255 * 0.6))
    thick = max(4, rw // 12)
    if rng.random() < 0.5:
        # solid badge (like app logo fondos)
        d.ellipse([x, y, x + rw, y + rh], fill=(255, 255, 255, op // 2))
    if rng.random() < 0.5:
        d.ellipse([x, y, x + rw, y + rh], outline=(255, 255, 255, op),
                  width=thick)
    else:
        d.rectangle([x, y, x + rw, y + rh], outline=(255, 255, 255, op),
                    width=thick)
    alpha = layer.split()[-1].point(lambda v: 255 if v > 10 else 0)
    mask_layer = Image.composite(Image.new("L", base.size, 255), mask_layer, alpha)
    return Image.alpha_composite(base, layer), mask_layer


def _stamp_band(base: Image.Image, mask_layer: Image.Image, rng: random.Random):
    w, h = base.size
    txt = (rng.choice(TEXTS) + " ") * rng.randint(3, 6)
    fs = rng.randint(max(12, h // 20), max(20, h // 8))
    font = _font(fs)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    bbox = d.textbbox((0, 0), txt, font=font)
    th = bbox[3] - bbox[1] + 8
    y = rng.choice([0, max(0, h - th - rng.randint(0, h // 10))])
    op = rng.randint(int(255 * 0.25), int(255 * 0.6))
    d.text((0, y), txt, font=font, fill=(255, 255, 255, op))
    alpha = layer.split()[-1].point(lambda v: 255 if v > 10 else 0)
    mask_layer = Image.composite(Image.new("L", base.size, 255), mask_layer, alpha)
    return Image.alpha_composite(base, layer), mask_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", required=True, help="folder of background photos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    bgs = [os.path.join(args.bg, f) for f in os.listdir(args.bg)
           if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    assert bgs, f"no backgrounds in {args.bg}"
    img_dir = os.path.join(args.out, "images")
    mask_dir = os.path.join(args.out, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    stampers = [_stamp_text, _stamp_shape, _stamp_band]
    for i in range(args.n):
        bg = Image.open(rng.choice(bgs)).convert("RGBA")
        mask = Image.new("L", bg.size, 0)
        for _ in range(rng.randint(1, 3)):
            bg, mask = rng.choice(stampers)(bg, mask, rng)
        bg.convert("RGB").save(os.path.join(img_dir, f"synth_{i:05d}.jpg"), quality=92)
        mask.save(os.path.join(mask_dir, f"synth_{i:05d}.png"))
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{args.n}", flush=True)
    print(f"done: {args.n} pairs in {args.out}")


if __name__ == "__main__":
    main()
