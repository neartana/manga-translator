"""OCR stage.

- manga-ocr (kha-white) for Japanese — purpose-built for manga, handles
  vertical text and furigana.
- EasyOCR as the multi-language fallback (~80 languages).

Both are lazy-loaded so a lightweight install never pays the model cost.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from PIL import Image

from .config import ProcessingConfig
from .utils import TextRegion

log = logging.getLogger("manga_translator")

# map our language codes to EasyOCR codes
EASYOCR_LANGS = {
    "ja": "ja", "en": "en", "zh-CN": "ch_sim", "zh-TW": "ch_tra",
    "ko": "ko", "es": "es", "fr": "fr", "de": "de", "it": "it",
    "pt": "pt", "ru": "ru", "ar": "ar", "th": "th", "vi": "vi",
    "id": "id", "nl": "nl", "pl": "pl", "tr": "tr", "uk": "uk",
}


class MangaOcrEngine:
    name = "manga-ocr"

    def __init__(self, cfg: ProcessingConfig):
        from manga_ocr import MangaOcr
        import torch  # noqa: F401  (ensure torch present for device check)
        self.mocr = MangaOcr()

    def read(self, crop: Image.Image) -> str:
        return (self.mocr(crop) or "").strip()


class EasyOcrEngine:
    name = "easyocr"

    def __init__(self, cfg: ProcessingConfig, langs: List[str]):
        import easyocr
        gpu = cfg.resolved_device() in ("cuda", "mps")
        self.reader = easyocr.Reader(langs, gpu=gpu, model_storage_directory=cfg.model_dir,
                                     verbose=False)

    def read(self, crop: Image.Image) -> str:
        arr = np.asarray(crop)
        results = self.reader.readtext(arr, detail=0, paragraph=True)
        return " ".join(t.strip() for t in results if t.strip())


class RapidOcrEngine:
    """ONNX-runtime OCR (no torch needed) — covers Latin + Chinese well."""

    name = "rapidocr"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()

    def read(self, crop: Image.Image) -> str:
        result, _ = self.engine(np.asarray(crop))
        if not result:
            return ""
        return " ".join(line[1].strip() for line in result if line[1].strip())


_engines: dict = {}


def _get_manga_ocr(cfg) -> MangaOcrEngine:
    if "manga-ocr" not in _engines:
        _engines["manga-ocr"] = MangaOcrEngine(cfg)
    return _engines["manga-ocr"]


def _get_easyocr(cfg, langs) -> EasyOcrEngine:
    key = ("easyocr", tuple(langs))
    if key not in _engines:
        _engines[key] = EasyOcrEngine(cfg, langs)
    return _engines[key]


def _get_rapidocr() -> RapidOcrEngine:
    if "rapidocr" not in _engines:
        _engines["rapidocr"] = RapidOcrEngine()
    return _engines["rapidocr"]


def _prep_crop(img: np.ndarray, region: TextRegion, pad: int = 6) -> Image.Image:
    h, w = img.shape[:2]
    x1, y1 = max(0, region.x1 - pad), max(0, region.y1 - pad)
    x2, y2 = min(w, region.x2 + pad), min(h, region.y2 + pad)
    crop = Image.fromarray(img[y1:y2, x1:x2])
    # upscale tiny crops — OCR engines like >= ~48px line height
    cw, ch = crop.size
    if max(cw, ch) < 128:
        scale = 128 / max(cw, ch)
        crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
    return crop


def run_ocr(img: np.ndarray, regions: List[TextRegion], cfg: ProcessingConfig) -> None:
    """Fill region.text in place."""
    src = cfg.source_lang
    for region in regions:
        crop = _prep_crop(img, region)
        text = ""
        use_mocr = (
            cfg.ocr == "manga-ocr"
            or (cfg.ocr == "auto" and (src == "ja" or (src == "auto" and region.direction == "vertical")))
        )
        if use_mocr:
            try:
                text = _get_manga_ocr(cfg).read(crop)
            except Exception as e:
                if cfg.ocr == "manga-ocr":
                    raise
                log.warning("manga-ocr failed (%s); trying EasyOCR.", e)
        if not text:
            langs = [EASYOCR_LANGS.get(src, "en")] if src in EASYOCR_LANGS else ["en"]
            try:
                text = _get_easyocr(cfg, langs).read(crop)
            except Exception as e:
                log.warning("EasyOCR unavailable (%s); trying RapidOCR.", e)
                try:
                    text = _get_rapidocr().read(crop)
                except Exception as e2:
                    log.error("OCR failed for region %s: %s", region.box, e2)
                    text = ""
        region.text = " ".join(text.split())
