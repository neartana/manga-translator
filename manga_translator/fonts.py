"""Font-pack management.

Layout of the fonts directory (drop your own packs in — that's it)::

    fonts/
      en/            <- used when target language is English
        ComicNeue-Bold.ttf
      zh-CN/
        NotoSansSC-Regular.otf
      ja/
      default/       <- fallback for any language
      <anything>.ttf <- files at the top level also count as 'default'

The folder name must match the target language code (en, zh-CN, ko, ...).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .config import ProcessingConfig

log = logging.getLogger("manga_translator")

FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}

_SYSTEM_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arialbd.ttf",
]


def list_fonts(font_dir: str) -> dict:
    """Return {pack_name: [font files]} for the UI."""
    root = Path(font_dir)
    packs: dict[str, List[str]] = {}
    if not root.exists():
        return packs
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() in FONT_EXTENSIONS:
            pack = "default" if f.parent == root else f.parent.name
            packs.setdefault(pack, []).append(f.name)
    return packs


class FontManager:
    def __init__(self, cfg: ProcessingConfig):
        self.font_dir = Path(cfg.font_dir)
        self.explicit = Path(cfg.font_path) if cfg.font_path else None
        self._cache: dict = {}

    def _files_for(self, target_lang: str) -> List[Path]:
        files: List[Path] = []
        lang_dir = self.font_dir / target_lang
        if lang_dir.is_dir():
            files += sorted(f for f in lang_dir.iterdir()
                            if f.suffix.lower() in FONT_EXTENSIONS)
        default_dir = self.font_dir / "default"
        if default_dir.is_dir():
            files += sorted(f for f in default_dir.iterdir()
                            if f.suffix.lower() in FONT_EXTENSIONS)
        if self.font_dir.is_dir():
            files += sorted(f for f in self.font_dir.iterdir()
                            if f.is_file() and f.suffix.lower() in FONT_EXTENSIONS)
        return files

    def get(self, target_lang: str, size: int):
        """Return a PIL FreeTypeFont at the requested size."""
        from PIL import ImageFont
        key = (target_lang, size)
        if key in self._cache:
            return self._cache[key]

        candidates: List[Path] = []
        if self.explicit and self.explicit.exists():
            candidates.append(self.explicit)
        candidates += self._files_for(target_lang)
        candidates += [Path(p) for p in _SYSTEM_CANDIDATES]

        font = None
        for path in candidates:
            try:
                if path.exists():
                    font = ImageFont.truetype(str(path), size)
                    break
            except Exception:
                continue
        if font is None:
            log.warning("No usable font found; falling back to PIL default bitmap font.")
            font = ImageFont.load_default(size)
        self._cache[key] = font
        return font
