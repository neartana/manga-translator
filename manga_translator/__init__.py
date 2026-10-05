"""Manga/comic page translator.

Detects text in speech bubbles and free-floating text, removes the
original lettering, translates it, and re-letters the page.
"""
__version__ = "0.1.0"

from .config import ProcessingConfig          # noqa: F401
from .pipeline import TranslationPipeline     # noqa: F401
