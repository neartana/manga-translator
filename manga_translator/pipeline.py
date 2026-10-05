"""The translation pipeline — one page in, translated page out.

Stages: detect -> OCR -> clean/inpaint -> translate -> render.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from .config import ProcessingConfig
from .detection import get_detector
from .inpainting import clean_image
from .ocr import run_ocr
from .rendering import TextRenderer
from .translation import translate_regions
from .utils import TextRegion, load_image, pil_to_np

log = logging.getLogger("manga_translator")

ProgressFn = Optional[Callable[[str, float], None]]  # (stage, fraction)


class TranslationPipeline:
    def __init__(self, cfg: ProcessingConfig):
        self.cfg = cfg
        self._renderer: Optional[TextRenderer] = None

    @property
    def renderer(self) -> TextRenderer:
        if self._renderer is None:
            self._renderer = TextRenderer(self.cfg)
        return self._renderer

    def _report(self, cb: ProgressFn, stage: str, frac: float):
        if cb:
            cb(stage, frac)
        log.info("[%-12s] %.0f%%", stage, frac * 100)

    # ------------------------------------------------------------------
    def process_image(self, img: Image.Image, progress: ProgressFn = None,
                      debug_dir: Optional[Path] = None) -> Tuple[np.ndarray, List[TextRegion]]:
        cfg = self.cfg
        arr = pil_to_np(img)

        self._report(progress, "detecting", 0.05)
        detector = get_detector(cfg)
        regions, _bubbles = detector.detect(arr)
        regions = [r for r in regions if r.area >= cfg.min_region_area]

        if not regions:
            self._report(progress, "done", 1.0)
            return arr, []

        self._report(progress, "reading text", 0.25)
        run_ocr(arr, regions, cfg)
        regions = [r for r in regions if r.text]

        if not regions:
            self._report(progress, "done", 1.0)
            return arr, []

        self._report(progress, "cleaning", 0.45)
        cleaned, mask = clean_image(arr, regions, cfg)

        self._report(progress, "translating", 0.65)
        translate_regions(regions, cfg)

        self._report(progress, "rendering", 0.85)
        out = self.renderer.render(cleaned, regions)

        if cfg.save_intermediates and debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(debug_dir / "mask.png")
            Image.fromarray(cleaned).save(debug_dir / "cleaned.png")

        self._report(progress, "done", 1.0)
        return out, regions

    # ------------------------------------------------------------------
    def process_file(self, src: Path, dst: Path, progress: ProgressFn = None,
                     debug_dir: Optional[Path] = None) -> List[TextRegion]:
        img = load_image(src)
        out, regions = self.process_image(img, progress=progress, debug_dir=debug_dir)

        dst.parent.mkdir(parents=True, exist_ok=True)
        fmt = self.cfg.output_format.lower()
        suffix = dst.suffix.lower().lstrip(".")
        if fmt == "auto":
            fmt = suffix if suffix in ("png", "jpg", "jpeg", "webp") else "png"
        if fmt in ("jpg", "jpeg"):
            dst = dst.with_suffix(".jpg")
            Image.fromarray(out).save(dst, quality=95)
        else:
            dst = dst.with_suffix(f".{fmt}")
            Image.fromarray(out).save(dst)
        return regions
