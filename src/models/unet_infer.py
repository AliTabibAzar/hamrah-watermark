"""U-Net inference — loads train_unet.py weights, returns a probability map.

torch is imported lazily: without it (or without weights) every function
raises a clear error that callers turn into a fallback, never a crash.
"""

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def load_unet_weights(weight_path: str = "models/watermark-unet.pt", device: str = "auto"):
    import torch

    from src.models.model_manager import pick_device
    from src.models.unet import WatermarkUNet

    dev = pick_device(device)
    model = WatermarkUNet(pretrained_encoder=False)
    state = torch.load(weight_path, map_location=dev)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(dev).eval()
    log.info("U-Net weights loaded from %s on %s", weight_path, dev)
    return model, dev


def predict_proba(image_bgr: np.ndarray, model=None, device: str = "cpu",
                  size: int = 256,
                  weight_path: str = "models/watermark-unet.pt") -> np.ndarray:
    """Returns a float32 HxW probability map in the ORIGINAL image size."""
    import cv2
    import torch

    if model is None:
        model, device = load_unet_weights(weight_path, device)
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = np.asarray(small, dtype=np.float32).transpose(2, 0, 1) / 255.0
    x = torch.from_numpy((x - _MEAN) / _STD)[None].to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    back = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return back.astype(np.float32)


def unet_mask(image_bgr: np.ndarray, threshold: float = 0.5, **kwargs) -> np.ndarray:
    return (predict_proba(image_bgr, **kwargs) > threshold).astype(np.uint8) * 255
