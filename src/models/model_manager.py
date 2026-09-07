"""Model management — everything loads lazily.

No module imports torch directly; everyone asks this manager instead,
so the code still imports on machines without a GPU.
"""

from pathlib import Path

from src.utils.logger import get_logger

log = get_logger(__name__)


def pick_device(want: str = "auto") -> str:
    if want != "auto":
        return want
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def weight_exists(path: str) -> bool:
    return Path(path).exists()


def load_unet(weight_path: str = "models/watermark-unet.pt", device: str = "auto"):
    """Returns the watermark U-Net, or None when weights are missing."""
    if not weight_exists(weight_path):
        log.warning("U-Net weights not found (%s)", weight_path)
        return None
    try:
        import torch
        model = torch.load(weight_path, map_location=pick_device(device))
        model.eval()
        return model
    except Exception as e:
        log.error("Failed to load U-Net: %s", e)
        return None


def load_lama(path: str = "models/big-lama.pt", device: str = "auto"):
    """Returns LaMa weights, or None so callers fall back to Telea."""
    if not weight_exists(path):
        log.warning("LaMa weights not found (%s) — falling back to Telea", path)
        return None
    try:
        import torch
        model = torch.load(path, map_location=pick_device(device))
        model.eval()
        return model
    except Exception as e:
        log.error("Failed to load LaMa: %s", e)
        return None
