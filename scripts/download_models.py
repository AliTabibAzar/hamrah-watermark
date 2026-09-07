"""Weight downloader — run on the server only, never locally.

Usage:
    python scripts/download_models.py
"""

from pathlib import Path

MODELS = {
    # Florence-2 base detector (~0.5GB) — MIT license
    "florence2-base": "https://huggingface.co/microsoft/Florence-2-base",
    # LaMa inpainting (~200MB)
    "big-lama": "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
}

if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    print("This script runs on the server only. Weights:")
    for name, url in MODELS.items():
        print(f" - {name}: {url}")
    print("On the server: pip install -r requirements-ai.txt, then")
    print("huggingface-cli download microsoft/Florence-2-base --local-dir models/florence2-base")
