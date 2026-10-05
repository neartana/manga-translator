"""Configuration for the manga/comic translation pipeline.

All settings live in a single dataclass so the Web UI, CLI and pipeline
share exactly the same options. Settings can be serialised to / from JSON
so the Web UI can persist user preferences.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_FONT_DIR = PROJECT_ROOT / "fonts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
CONFIG_PATH = PROJECT_ROOT / "config.json"

# Languages shown in the UI / accepted by the CLI.
# Keys are ISO-ish codes, values are (display name, google-translate code).
SUPPORTED_LANGUAGES = {
    "auto": ("Auto detect", "auto"),
    "ja": ("Japanese", "ja"),
    "en": ("English", "en"),
    "zh-CN": ("Chinese (Simplified)", "zh-CN"),
    "zh-TW": ("Chinese (Traditional)", "zh-TW"),
    "ko": ("Korean", "ko"),
    "es": ("Spanish", "es"),
    "fr": ("French", "fr"),
    "de": ("German", "de"),
    "it": ("Italian", "it"),
    "pt": ("Portuguese", "pt"),
    "ru": ("Russian", "ru"),
    "ar": ("Arabic", "ar"),
    "th": ("Thai", "th"),
    "vi": ("Vietnamese", "vi"),
    "id": ("Indonesian", "id"),
    "nl": ("Dutch", "nl"),
    "pl": ("Polish", "pl"),
    "tr": ("Turkish", "tr"),
    "uk": ("Ukrainian", "uk"),
}

TARGET_LANGUAGES = {k: v for k, v in SUPPORTED_LANGUAGES.items() if k != "auto"}


@dataclass
class TranslatorConfig:
    """LLM endpoint settings (OpenAI-compatible)."""
    base_url: str = field(default_factory=lambda: os.environ.get(
        "MT_LLM_BASE_URL", "https://api.openai.com/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("MT_LLM_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("MT_LLM_MODEL", "gpt-4o-mini"))
    temperature: float = 0.3
    max_retries: int = 2


@dataclass
class ProcessingConfig:
    # language
    source_lang: str = "auto"          # 'auto' or a code from SUPPORTED_LANGUAGES
    target_lang: str = "en"

    # pipeline stages
    translator: str = "auto"           # auto | llm | google
    detector: str = "auto"             # auto | yolo | opencv
    ocr: str = "auto"                  # auto | manga-ocr | easyocr
    inpainter: str = "auto"            # auto | lama | opencv

    # hardware
    device: str = "auto"               # auto | cpu | cuda | mps

    # detection tuning
    detection_size: int = 1280         # longer-side resize for the detector
    box_threshold: float = 0.35        # detector confidence
    mask_dilation: int = 12            # px to grow the text mask before inpainting

    # OCR / text handling
    min_text_length: int = 1
    min_region_area: int = 80          # px^2, drop specks

    # rendering
    direction: str = "auto"            # auto | horizontal | vertical
    font_path: str = ""                # explicit font file overrides font packs
    font_dir: str = str(DEFAULT_FONT_DIR)
    font_size_offset: int = 0
    font_size_minimum: int = 12
    line_spacing: float = 1.0
    uppercase: bool = False
    text_color: str = "auto"           # auto | #rrggbb
    outline: bool = True               # dark outline for light text & vice versa
    hyphenate: bool = True

    # batch behaviour
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    output_format: str = "auto"        # auto | png | jpg | webp
    skip_no_text: bool = False
    overwrite: bool = True
    save_intermediates: bool = False   # save masks / cleaned pages for debugging
    use_cache: bool = True

    # LLM settings
    llm: TranslatorConfig = field(default_factory=TranslatorConfig)

    # paths
    model_dir: str = str(DEFAULT_MODEL_DIR)

    def resolved_device(self) -> str:
        """Return 'cuda', 'mps' or 'cpu'."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    # ---- persistence -------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "ProcessingConfig":
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                llm_data = data.pop("llm", {})
                cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
                for k, v in llm_data.items():
                    if hasattr(cfg.llm, k):
                        setattr(cfg.llm, k, v)
                return cfg
            except Exception:
                pass
        return cls()
