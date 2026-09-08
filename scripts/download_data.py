"""Dataset fetcher — runs on Kaggle (needs internet), not locally.

Sources (all public, research-standard):
  CLWD      60K colored watermarks  — https://github.com/MRUIL/WDNet
  LOGO-*    12K each (L/H/Gray)     — https://github.com/vinthony/deep-blind-watermark-removal
  PITA      ~19.5K with masks       — https://huggingface.co/datasets/bastienp/visible-watermark-pita

Usage on Kaggle:
    python scripts/download_data.py --out data --sets clwd logo pita

Everything lands in the images/+masks layout that src/data understands.
"""

import argparse
import os

SETS = ("clwd", "logo-l", "logo-h", "logo-gray", "pita")


def fetch_pita(out: str):
    from huggingface_hub import snapshot_download
    dst = os.path.join(out, "pita_raw")
    snapshot_download("bastienp/visible-watermark-pita", repo_type="dataset",
                      local_dir=dst)
    print(f"PITA downloaded to {dst} — convert to images/+masks with scripts/convert_pita.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--sets", nargs="+", default=["clwd"], choices=SETS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if "pita" in args.sets:
        fetch_pita(args.out)
    for s in ("clwd", "logo-l", "logo-h", "logo-gray"):
        if s in args.sets:
            print(f"[{s}] download from the repo linked above into {args.out}/{s}/ "
                  f"(images/ + masks/ layout), then re-run training.")


if __name__ == "__main__":
    main()
