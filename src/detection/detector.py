"""Watermark detection — multi-cue classical detector, no weights needed.

Design: watermarks are overlays, so each cue hunts one overlay family and
votes. A pixel survives with >= 2 votes (whiteness counts double because
semi-transparent white overlays are the most common case):
  - whiteness: bright + desaturated vs the LOCAL background (catches all
    semi-transparent white text/logos on any background, dark or light)
  - top-hat: bright structures larger than the image texture
  - mser: stable text-stroke regions (PROOF, Copyright, names)
  - edges: tiles whose edge density is an outlier vs the image median
    (relative threshold — absolute ones fire on rocky textures)
Full-image grid coverage: center and tiled watermarks are not missed.
"""

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


def cue_whiteness(bgr: np.ndarray, v_margin: int = 5, s_max: int = 200,
                  v_min: int = 60, bg_kernel: int = 15) -> np.ndarray:
    """Bright-overlay pixels: slightly brighter than the LOCAL background.

    Semi-transparent overlays inherit background saturation, so the
    saturation gate stays wide open — brightness vs a tight local median
    does the work. Voting with other cues suppresses the noise this lets in.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    k = bg_kernel if bg_kernel % 2 == 1 else bg_kernel + 1
    bg = cv2.medianBlur(v, k)
    boost = cv2.subtract(v, bg)  # saturating, no wrap-around
    white = ((boost > v_margin) & (s < s_max) & (v > v_min)).astype(np.uint8) * 255
    return white


def cue_tophat(gray: np.ndarray, kernel: int = 25, thresh: int = 15) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    _, out = cv2.threshold(th, thresh, 255, cv2.THRESH_BINARY)
    return out


def cue_strokes(bgr: np.ndarray, v_margin: int = 5, v_min: int = 60,
                grad_pct: float = 93.0) -> np.ndarray:
    """Thin bright structures: locally brighter AND high local contrast.

    Watermark strokes are bright edges; skies and calm water are bright but
    smooth. Requiring both rejects the smooth-bright false positives that a
    whiteness cue alone produces.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    bg = cv2.medianBlur(v, 15)
    bright = (cv2.subtract(v, bg) > v_margin) & (v > v_min)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge = grad > np.percentile(grad, grad_pct)
    return (bright & edge).astype(np.uint8) * 255


def cue_mser(gray: np.ndarray, min_area: int = 30, max_area_ratio: float = 0.05) -> np.ndarray:
    h, w = gray.shape
    mser = cv2.MSER_create(delta=3, min_area=min_area,
                           max_area=int(h * w * max_area_ratio))
    regions, _ = mser.detectRegions(gray)
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in regions:
        x, y, rw, rh = cv2.boundingRect(pts.reshape(-1, 1, 2))
        # text strokes are compact, not full-frame blobs
        if rw < w * 0.9 and rh < h * 0.9:
            mask[y:y + rh, x:x + rw] = 255
    return mask


def cue_edge_tiles(gray: np.ndarray, grid: int = 6, overlap: float = 0.25,
                   k_std: float = 1.0) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    h, w = gray.shape
    densities = []
    tiles = []
    step_y, step_x = h / grid, w / grid
    win_y, win_x = step_y * (1 + overlap), step_x * (1 + overlap)
    for gy in range(grid):
        for gx in range(grid):
            y1 = max(0, int(gy * step_y - (win_y - step_y) / 2))
            x1 = max(0, int(gx * step_x - (win_x - step_x) / 2))
            y2 = min(h, int(y1 + win_y))
            x2 = min(w, int(x1 + win_x))
            d = float((edges[y1:y2, x1:x2] > 0).mean())
            densities.append(d)
            tiles.append((x1, y1, x2, y2))
    med = float(np.median(densities))
    std = float(np.std(densities)) + 1e-6
    mask = np.zeros((h, w), dtype=np.uint8)
    found = False
    for d, (x1, y1, x2, y2) in zip(densities, tiles):
        if d > med + k_std * std and d > 0.003:
            mask[y1:y2, x1:x2] = 255
            found = True
    return mask if found else np.zeros((h, w), dtype=np.uint8)


def combine_cues(white: np.ndarray, tophat: np.ndarray,
                 mser: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # kept for API compatibility; the main path uses strokes + support
    score = ((white > 0).astype(np.uint8) + (tophat > 0).astype(np.uint8)
             + (mser > 0).astype(np.uint8) + (edges > 0).astype(np.uint8))
    return (score >= 2).astype(np.uint8) * 255


_OCR_ENGINE = None


def cue_ocr(image_bgr: np.ndarray, min_conf: float = 0.3) -> tuple[np.ndarray, list[str]]:
    """Text boxes from on-device OCR. Returns (mask, texts found).

    OCR reads watermark text directly (PROOF, names, logos) in under a
    second on CPU. Missing library -> empty mask, never raises.
    """
    global _OCR_ENGINE
    h, w = image_bgr.shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
        except ImportError:
            log.warning("rapidocr not installed — OCR cue disabled")
            _OCR_ENGINE = False
    if _OCR_ENGINE is False:
        return empty, []
    try:
        res, _ = _OCR_ENGINE(image_bgr)
    except Exception as e:
        log.warning("OCR failed: %s", e)
        return empty, []
    mask = np.zeros((h, w), dtype=np.uint8)
    texts = []
    for quad, text, conf in (res or []):
        if float(conf) < min_conf:
            continue
        pts = np.array(quad, dtype=np.int32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [pts], 255)
        texts.append(str(text))
    return mask, texts


def heuristic_mask(image_bgr: np.ndarray, settings: dict | None = None) -> tuple[np.ndarray, bool]:
    """Multi-cue mask. Returns (mask, found). Same signature as before."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    white = cue_whiteness(image_bgr)
    top = cue_tophat(gray)
    ms = cue_mser(gray)
    ed = cue_edge_tiles(gray)
    ocr, _ = cue_ocr(image_bgr)
    # two-tier vote: OCR boxes always count (OCR fires on text, not clouds);
    # elsewhere 3+ agreeing cues are needed; near OCR text 2 suffice since
    # logos/icons sit next to their wordmark
    s = ((white > 0).astype(np.uint8) + (top > 0).astype(np.uint8)
         + (ms > 0).astype(np.uint8) + (ed > 0).astype(np.uint8))
    near_text = cv2.dilate((ocr > 0).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
    keep = (ocr > 0) | (s >= 3) | ((s >= 2) & near_text)
    mask = keep.astype(np.uint8) * 255
    found = bool((mask > 0).sum() > 100)
    return (mask if found else np.zeros_like(mask)), found


def unet_mask(image_bgr: np.ndarray, weight_path: str = "models/watermark-unet.pt") -> np.ndarray:
    """Placeholder for the U-Net model until weights are on the server."""
    from pathlib import Path
    if not Path(weight_path).exists():
        raise FileNotFoundError(
            f"U-Net weights missing: {weight_path} — "
            "run scripts/download_models.py on the server first"
        )
    raise NotImplementedError("U-Net inference wiring comes next")


def detect(image_bgr: np.ndarray, mode: str = "heuristic",
           settings: dict | None = None) -> tuple[np.ndarray, str]:
    """Input BGR, output (mask, message). Never raises."""
    try:
        if mode == "auto":
            return unet_mask(image_bgr), "Model generated the mask."
        mask, found = heuristic_mask(image_bgr, settings)
        if found:
            return mask, "Watermark regions found — please check the mask."
        return mask, "No watermark found. Please mark it manually."
    except (FileNotFoundError, NotImplementedError) as e:
        log.warning("Auto detection not ready: %s", e)
        h, w = image_bgr.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), \
            "Auto model not installed on the server yet — continue manually."
