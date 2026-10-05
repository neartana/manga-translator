"""Mask building (text cleaning) and background inpainting."""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from .config import ProcessingConfig
from .utils import TextRegion, sample_ink_color

log = logging.getLogger("manga_translator")


def build_text_mask(img: np.ndarray, regions: list[TextRegion], cfg: ProcessingConfig) -> np.ndarray:
    """Build a binary mask (255 = remove) covering exactly the text pixels."""
    import cv2

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    for region in regions:
        if not region.text:
            continue
        x1, y1 = max(0, region.x1), max(0, region.y1)
        x2, y2 = min(w, region.x2), min(h, region.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = gray[y1:y2, x1:x2]

        ink = sample_ink_color(img, region)
        region.fg_color = ink
        dark_text = sum(ink) / 3 < 128

        # binarise the crop, polarity chosen by detected ink colour
        block = cv2.adaptiveThreshold(
            crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if dark_text else cv2.THRESH_BINARY, 25, 10)
        if region.kind == "bubble-text":
            # Separate lettering from the balloon outline: components that
            # touch the balloon's border ring are outline/tail — keep them;
            # everything fully interior is text and gets masked.
            ch, cw = block.shape[:2]
            ring = np.zeros((ch, cw), np.uint8)
            band = max(4, min(cw, ch) // 20)
            cv2.rectangle(ring, (0, 0), (cw - 1, ch - 1), 255, band)
            n, labels = cv2.connectedComponents(block)
            ring_labels = set(np.unique(labels[ring > 0])) - {0}
            for lab in range(1, n):
                if lab in ring_labels:
                    block[labels == lab] = 0
        # only keep the central area — adaptive threshold edges can halo
        block[:1, :] = 0; block[-1:, :] = 0; block[:, :1] = 0; block[:, -1:] = 0
        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], block)

    if cfg.mask_dilation > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (cfg.mask_dilation, cfg.mask_dilation))
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


class LamaInpainter:
    name = "lama"

    def __init__(self, cfg: ProcessingConfig):
        from simple_lama_inpainting import SimpleLama
        self.lama = SimpleLama(device=cfg.resolved_device())

    def inpaint(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = self.lama(Image.fromarray(img), Image.fromarray(mask))
        return np.asarray(result.convert("RGB"))


class OpenCVInpainter:
    name = "opencv"

    def inpaint(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import cv2
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


_inpainter_cache: dict = {}


def get_inpainter(cfg: ProcessingConfig):
    choice = cfg.inpainter
    if choice in ("auto", "lama"):
        try:
            if "lama" not in _inpainter_cache:
                _inpainter_cache["lama"] = LamaInpainter(cfg)
            return _inpainter_cache["lama"]
        except Exception as e:
            if choice == "lama":
                raise
            log.warning("LaMa inpainter unavailable (%s); using OpenCV.", e)
    if "opencv" not in _inpainter_cache:
        _inpainter_cache["opencv"] = OpenCVInpainter()
    return _inpainter_cache["opencv"]


def clean_image(img: np.ndarray, regions: list[TextRegion], cfg: ProcessingConfig):
    """Return (cleaned_image, mask). Bubble interiors get flat-filled first,
    which keeps balloon interiors perfectly clean even without LaMa."""
    import cv2
    mask = build_text_mask(img, regions, cfg)

    out = img.copy()
    # cheap win: inside white-ish bubbles, replace text pixels with the
    # bubble's dominant background colour before model inpainting
    for region in regions:
        if not region.text or region.kind != "bubble-text":
            continue
        x1, y1, x2, y2 = region.box
        sub_mask = mask[y1:y2, x1:x2] > 0
        if sub_mask.sum() == 0:
            continue
        crop = out[y1:y2, x1:x2]
        bg_pixels = crop[~sub_mask]
        if len(bg_pixels) > 10 and bg_pixels.mean() > 180:  # light bubble
            bg = np.median(bg_pixels, axis=0)
            crop[sub_mask] = bg
            mask[y1:y2, x1:x2][sub_mask] = 0  # nothing left for the model here

    if mask.sum() > 0:
        out = get_inpainter(cfg).inpaint(out, mask)
    return out, mask
