"""Shared data structures and image helpers."""
from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger("manga_translator")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class TextRegion:
    """A detected block of text (inside a bubble or free-floating)."""
    x1: int
    y1: int
    x2: int
    y2: int
    text: str = ""
    translation: str = ""
    kind: str = "text"          # 'bubble-text' | 'text' (free / outside bubbles)
    direction: str = "horizontal"  # or 'vertical'
    confidence: float = 1.0
    bubble: Optional[Tuple[int, int, int, int]] = None  # enclosing bubble box, if any
    fg_color: Tuple[int, int, int] = (0, 0, 0)          # detected ink colour

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class PageResult:
    source_path: str
    output_path: str = ""
    regions: List[TextRegion] = field(default_factory=list)
    error: Optional[str] = None


def load_image(path_or_bytes) -> Image.Image:
    if isinstance(path_or_bytes, (str, Path)):
        img = Image.open(path_or_bytes)
    else:
        img = Image.open(io.BytesIO(path_or_bytes))
    img = img.convert("RGB")
    return img


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def collect_images(root: Path) -> List[Path]:
    """Recursively collect image files, sorted for deterministic order."""
    return sorted(p for p in root.rglob("*") if p.is_file() and is_image(p))


def extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract a zip into dest/<zip-stem> and return that folder."""
    target = dest / zip_path.stem
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # basic zip-slip protection
            if member.startswith("/") or ".." in Path(member).parts:
                continue
            zf.extract(member, target)
    return target


def resize_longer_side(img: np.ndarray, size: int) -> Tuple[np.ndarray, float]:
    """Resize so the longer side equals `size`. Returns (image, scale)."""
    h, w = img.shape[:2]
    longer = max(h, w)
    if longer <= size:
        return img, 1.0
    scale = size / longer
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    import cv2
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def guess_direction(region: TextRegion) -> str:
    """Vertical if the box is notably taller than wide (CJK columns)."""
    return "vertical" if region.height > region.width * 1.35 else "horizontal"


def sample_ink_color(img: np.ndarray, region: TextRegion) -> Tuple[int, int, int]:
    """Estimate the colour of the original text inside a region.

    Looks at the extreme tails of the luminance histogram: dark ink on a
    light background (typical) or light ink on dark art (night scenes,
    SFX). Returns an (r, g, b) tuple.
    """
    import cv2
    x1, y1, x2, y2 = region.box
    crop = img[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)]
    if crop.size == 0:
        return (0, 0, 0)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    p_dark = np.percentile(gray, 2)
    if p_dark < 128:                       # dark text on light ground
        mask = gray <= max(p_dark, 1)
    else:                                  # light text on dark ground
        mask = gray >= np.percentile(gray, 98)
    if mask.sum() < 4:
        return (0, 0, 0)
    colour = crop[mask].mean(axis=0)
    return tuple(int(c) for c in colour)
