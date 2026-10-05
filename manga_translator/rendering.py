"""Text rendering — draws translations back onto the cleaned page.

Features: automatic font-size fitting, word wrap (with optional
hyphenation) and character wrap for CJK, horizontal and vertical
typesetting, automatic ink colour + contrasting outline, custom font
packs per target language.
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .config import ProcessingConfig
from .fonts import FontManager
from .utils import TextRegion

log = logging.getLogger("manga_translator")

_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]")


def _is_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore


class TextRenderer:
    def __init__(self, cfg: ProcessingConfig):
        self.cfg = cfg
        self.fonts = FontManager(cfg)

    # ---- measuring ----------------------------------------------------
    def _wrap(self, text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> List[str]:
        def w(s: str) -> float:
            return draw.textlength(s, font=font)

        if _is_cjk(text):  # character-level wrap
            lines, cur = [], ""
            for ch in text:
                if ch == "\n" or (cur and w(cur + ch) > max_width):
                    lines.append(cur)
                    cur = "" if ch == "\n" else ch
                else:
                    cur += ch
            if cur:
                lines.append(cur)
            return lines

        def split_long_word(word: str) -> List[str]:
            """Break an over-wide word into hyphenated pieces that fit."""
            if not (self.cfg.hyphenate and len(word) > 6):
                return [word]
            pieces, cur = [], ""
            for ch in word:
                if cur and w(cur + ch + "-") > max_width:
                    pieces.append(cur + "-")
                    cur = ch
                else:
                    cur += ch
            pieces.append(cur)
            return [p for p in pieces if p != "-"]

        lines: List[str] = []
        for raw_line in text.split("\n"):
            cur = ""
            for word in raw_line.split():
                for piece in split_long_word(word):
                    if not cur:
                        cur = piece
                    elif w(cur + " " + piece) <= max_width:
                        cur = cur + " " + piece
                    else:
                        lines.append(cur)
                        cur = piece
            if cur:
                lines.append(cur)
        return lines or [""]

    def _fit_font_size(self, text: str, box_w: int, box_h: int, vertical: bool,
                       draw: ImageDraw.ImageDraw, lang: str) -> Tuple[int, List[str]]:
        lo, hi = self.cfg.font_size_minimum, max(self.cfg.font_size_minimum, int(box_h * 1.2))
        best_size, best_lines = lo, [text]
        while lo <= hi:
            mid = (lo + hi) // 2
            font = self.fonts.get(lang, mid)
            if vertical:
                cols = self._wrap_vertical(text, font, box_h, draw)
                fits = len(cols) * (mid * self.cfg.line_spacing * 1.1) <= box_w + mid * 0.4
                lines = cols
            else:
                lines = self._wrap(text, font, box_w, draw)
                line_h = mid * 1.15 * self.cfg.line_spacing
                fits = len(lines) * line_h <= box_h
            if fits:
                best_size, best_lines = mid, lines
                lo = mid + 1
            else:
                hi = mid - 1
        return best_size + self.cfg.font_size_offset, best_lines

    def _wrap_vertical(self, text: str, font, max_height: int,
                       draw: ImageDraw.ImageDraw) -> List[str]:
        size = getattr(font, "size", 16)
        per_col = max(1, int(max_height // (size * 1.05 * self.cfg.line_spacing)))
        chars = [c for c in text if c != "\n"]
        return ["".join(chars[i:i + per_col]) for i in range(0, len(chars), per_col)] or [""]

    # ---- colours --------------------------------------------------------
    def _colors(self, region: TextRegion, img: np.ndarray):
        if self.cfg.text_color != "auto" and self.cfg.text_color.startswith("#"):
            fg = _hex_to_rgb(self.cfg.text_color)
        else:
            fg = region.fg_color
        lum = 0.299 * fg[0] + 0.587 * fg[1] + 0.114 * fg[2]
        # snap to near-black / near-white for readability, keep coloured ink
        if lum < 80:
            fg = (20, 20, 20)
        elif lum > 200:
            fg = (250, 250, 250)
        outline = (250, 250, 250) if lum < 128 else (20, 20, 20)
        return fg, (outline if self.cfg.outline else None)

    # ---- main ----------------------------------------------------------
    def render(self, img: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
        page = Image.fromarray(img).convert("RGB")
        draw = ImageDraw.Draw(page)
        lang = self.cfg.target_lang

        for region in regions:
            text = region.translation or ""
            if not text:
                continue
            if self.cfg.uppercase:
                text = text.upper()

            vertical = (
                self.cfg.direction == "vertical"
                or (self.cfg.direction == "auto" and region.direction == "vertical" and _is_cjk(text))
            )
            if vertical and not _is_cjk(text):
                vertical = False  # latin text stacked vertically is unreadable

            # render into the bubble when we know it, else the text box (padded)
            if region.bubble:
                bx1, by1, bx2, by2 = region.bubble
                # inscribe a rectangle in the balloon's ellipse (~1/√2) so
                # lines never spill over the curved outline
                pad_x, pad_y = int((bx2 - bx1) * 0.17), int((by2 - by1) * 0.17)
                box = (bx1 + pad_x, by1 + pad_y, bx2 - pad_x, by2 - pad_y)
            else:
                pad = 6
                box = (region.x1 - pad, region.y1 - pad, region.x2 + pad, region.y2 + pad)
            bw, bh = max(10, box[2] - box[0]), max(10, box[3] - box[1])

            size, lines = self._fit_font_size(text, bw, bh, vertical, draw, lang)
            size = max(self.cfg.font_size_minimum, size)
            font = self.fonts.get(lang, size)
            fg, outline = self._colors(region, img)
            stroke = max(0, size // 18) if outline else 0

            if vertical:
                self._draw_vertical(draw, lines, font, box, fg, outline, stroke)
            else:
                self._draw_horizontal(draw, lines, font, box, size, fg, outline, stroke)
        return np.asarray(page)

    def _draw_horizontal(self, draw, lines, font, box, size, fg, outline, stroke):
        line_h = size * 1.15 * self.cfg.line_spacing
        total_h = line_h * len(lines)
        # nudge above centre — balloon tails hang below the visual middle
        y = box[1] + (box[3] - box[1] - total_h) / 2 - (box[3] - box[1]) * 0.04
        for line in lines:
            lw = draw.textlength(line, font=font)
            x = box[0] + (box[2] - box[0] - lw) / 2
            draw.text((x, y), line, font=font, fill=fg,
                      stroke_width=stroke, stroke_fill=outline)
            y += line_h

    def _draw_vertical(self, draw, columns, font, box, fg, outline, stroke):
        size = getattr(font, "size", 16)
        col_w = size * 1.05 * self.cfg.line_spacing
        total_w = col_w * len(columns)
        x = box[2] - (box[2] - box[0] - total_w) / 2 - col_w  # right-to-left
        for col in columns:
            col_h = len(col) * size * 1.05
            y = box[1] + (box[3] - box[1] - col_h) / 2
            for ch in col:
                draw.text((x, y), ch, font=font, fill=fg,
                          stroke_width=stroke, stroke_fill=outline)
                y += size * 1.05
            x -= col_w
