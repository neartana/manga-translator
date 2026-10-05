"""Translation stage.

Backends:
- ``llm``    : any OpenAI-compatible chat endpoint (OpenAI, Kimi/Moonshot,
               DeepSeek, Anthropic proxies, llama.cpp / LM Studio local
               servers...). The whole page's text goes in one request so the
               model sees dialogue context.
- ``google`` : free Google Translate via deep-translator (no key needed).
- ``auto``   : use the LLM when an API key is configured, else Google.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from .config import ProcessingConfig, SUPPORTED_LANGUAGES

log = logging.getLogger("manga_translator")

CACHE_PATH = Path(__file__).resolve().parent.parent / "outputs" / ".translation_cache.json"


class TranslationCache:
    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self._data = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def key(self, text: str, src: str, tgt: str, backend: str) -> str:
        raw = f"{backend}|{src}|{tgt}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def put(self, key: str, value: str):
        self._data[key] = value

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=1),
                             encoding="utf-8")


class GoogleBackend:
    name = "google"

    def __init__(self, cfg: ProcessingConfig):
        from deep_translator import GoogleTranslator
        self.cfg = cfg

    def translate(self, texts: List[str], src: str, tgt: str) -> List[str]:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTTimeout
        from deep_translator import GoogleTranslator
        src_code = SUPPORTED_LANGUAGES.get(src, ("", "auto"))[1]
        tgt_code = SUPPORTED_LANGUAGES.get(tgt, ("", "en"))[1]
        tr = GoogleTranslator(source=src_code, target=tgt_code)
        out = []
        pool = ThreadPoolExecutor(max_workers=4)
        futures = [pool.submit(tr.translate, t) for t in texts]
        for t, fut in zip(texts, futures):
            try:
                out.append(fut.result(timeout=45) or t)
            except FTTimeout:
                log.error("Google translate timed out — check network/proxy")
                out.append(t)
            except Exception as e:
                log.error("Google translate failed: %s", e)
                out.append(t)
        pool.shutdown(wait=False)
        return out


class LLMBackend:
    name = "llm"

    def __init__(self, cfg: ProcessingConfig):
        self.cfg = cfg.llm
        if not self.cfg.api_key:
            raise RuntimeError("No LLM API key configured")

    def translate(self, texts: List[str], src: str, tgt: str) -> List[str]:
        import requests
        src_name = SUPPORTED_LANGUAGES.get(src, ("the source language",))[0]
        tgt_name = SUPPORTED_LANGUAGES.get(tgt, ("English",))[0]
        numbered = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))
        prompt = (
            f"Translate the following comic/manga dialogue lines from {src_name} to {tgt_name}. "
            "Keep the numbering exactly as given, one translated line per source line. "
            "Translate naturally for comics: casual speech stays casual, keep sound-effect "
            "lines punchy and short, keep honorifics only if the target language has an "
            "equivalent tone. Do not add explanations.\n\n" + numbered
        )
        body = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "messages": [
                {"role": "system", "content": "You are a professional manga letterer and translator."},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.cfg.api_key}",
                   "Content-Type": "application/json"}
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=120)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return self._parse(content, texts)
            except Exception as e:
                last_err = e
                log.warning("LLM translate attempt %d failed: %s", attempt + 1, e)
        raise RuntimeError(f"LLM translation failed: {last_err}")

    @staticmethod
    def _parse(content: str, originals: List[str]) -> List[str]:
        results: dict[int, str] = {}
        for line in content.strip().splitlines():
            m = re.match(r"\s*\[?(\d+)\]?\s*[.:\)]?\s*(.+)", line.strip())
            if m:
                idx = int(m.group(1))
                results.setdefault(idx, m.group(2).strip())
        out = []
        for i in range(1, len(originals) + 1):
            out.append(results.get(i) or originals[i - 1])
        return out


_backend_cache: dict = {}


def get_backend(cfg: ProcessingConfig):
    choice = cfg.translator
    if choice in ("auto", "llm"):
        try:
            if "llm" not in _backend_cache:
                _backend_cache["llm"] = LLMBackend(cfg)
            return _backend_cache["llm"]
        except Exception as e:
            if choice == "llm":
                raise
            log.info("LLM backend not configured (%s); using Google Translate.", e)
    if "google" not in _backend_cache:
        _backend_cache["google"] = GoogleBackend(cfg)
    return _backend_cache["google"]


_cache = TranslationCache()


def translate_regions(regions, cfg: ProcessingConfig) -> None:
    texts = [r.text for r in regions if r.text and len(r.text) >= cfg.min_text_length]
    if not texts:
        return
    backend = get_backend(cfg)
    src, tgt = cfg.source_lang, cfg.target_lang

    missing, missing_keys = [], []
    resolved: dict[str, str] = {}
    for t in texts:
        key = _cache.key(t, src, tgt, backend.name)
        if cfg.use_cache and (hit := _cache.get(key)):
            resolved[t] = hit
        else:
            missing.append(t)
            missing_keys.append(key)

    if missing:
        # LLM context window is fine with a full page; Google gets one call per line
        translated = backend.translate(missing, src, tgt)
        for t, key, tr in zip(missing, missing_keys, translated):
            resolved[t] = tr
            _cache.put(key, tr)
        _cache.save()

    for r in regions:
        if r.text in resolved:
            r.translation = resolved[r.text]
