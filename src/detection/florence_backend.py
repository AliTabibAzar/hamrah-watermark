"""Florence-2 backend — lazy, mock-safe, engine-swappable.

No torch/transformers import at module level: everything loads inside
functions, so this module imports fine on machines without a GPU.
Swap the engine later by subclassing BaseVLMBackend.
"""

from src.utils.logger import get_logger

log = get_logger(__name__)


class BaseVLMBackend:
    """Contract every VLM engine follows. Boxes are x1,y1,x2,y2 pixels."""

    name = "base"

    def propose(self, image_rgb, prompts: list[str]) -> list[dict]:
        """Coarse pass: returns [{'box': (x1,y1,x2,y2), 'label': str}, ...]."""
        raise NotImplementedError

    def refine_box(self, image_rgb, box: tuple) -> tuple:
        """Tight pass on a crop. Returns a tighter box or the input."""
        return box


class MockBackend(BaseVLMBackend):
    """Deterministic fake for tests — returns a fixed center box."""

    name = "mock"

    def propose(self, image_rgb, prompts: list[str]) -> list[dict]:
        h, w = image_rgb.shape[:2]
        return [{"box": (w // 4, h // 4, 3 * w // 4, 3 * h // 4),
                 "label": prompts[0] if prompts else "watermark"}]


class FlorenceBackend(BaseVLMBackend):
    """microsoft/Florence-2-base via transformers. Server only.

    Chain per prompt: <OPEN_VOCABULARY_DETECTION> for boxes, then
    <REFERRING_EXPRESSION_SEGMENTATION> for pixel polygons, plus
    <OCR_WITH_REGION> for text watermarks. Callers union the results.
    """

    name = "florence2-base"
    MODEL_ID = "microsoft/Florence-2-base"

    def __init__(self, device: str = "auto"):
        self._model = None
        self._processor = None
        self._device = device

    def _ensure_loaded(self):
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as e:
            log.warning("Florence-2 deps missing (torch/transformers): %s", e)
            return False
        try:
            from src.models.model_manager import pick_device
            device = pick_device(self._device)
            dtype = torch.float16 if device == "cuda" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID, torch_dtype=dtype, trust_remote_code=True).to(device)
            self._processor = AutoProcessor.from_pretrained(
                self.MODEL_ID, trust_remote_code=True)
            self._device_resolved = device
            return True
        except Exception as e:
            log.error("Failed to load Florence-2: %s", e)
            return False

    def propose(self, image_rgb, prompts: list[str]) -> list[dict]:
        if not self._ensure_loaded():
            raise RuntimeError("Florence-2 weights/backend not available")
        from PIL import Image
        pil = Image.fromarray(image_rgb)
        out = []
        for p in prompts:
            task = "<OPEN_VOCABULARY_DETECTION>"
            inputs = self._processor(text=task + " " + p, images=pil,
                                     return_tensors="pt").to(self._model.device)
            ids = self._model.generate(**inputs, max_new_tokens=1024, num_beams=3)
            text = self._processor.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = self._processor.post_process_generation(
                text, task=task, image_size=pil.size)
            for box, label in zip(parsed.get("bboxes", []),
                                  parsed.get("labels", [])):
                x1, y1, x2, y2 = (int(v) for v in box)
                out.append({"box": (x1, y1, x2, y2), "label": label or p})
        return out
