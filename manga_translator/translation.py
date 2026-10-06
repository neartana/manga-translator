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
        # include the exact source text (repr) so lookups can never collide
        # with a different string and silently return the wrong translation
        raw = repr([backend, src, tgt, text])
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
        # NOTE: previously the translator instance was created but never used,
        # and every line silently fell back to the *original* text on any
        # error — which made the output look completely untranslated.
        src_code = SUPPORTED_LANGUAGES.get(src, ("", "auto"))[1]
        tgt_code = SUPPORTED_LANGUAGES.get(tgt, ("", "en"))[1]
        if tgt_code == "auto" or not tgt_code:
            raise RuntimeError(f"Unsupported target language: {tgt!r}")

        def _new_translator():
            # one instance per thread — deep-translator objects are not reentrant
            return GoogleTranslator(source=src_code, target=tgt_code)

        out: List[str] = []
        failures = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for t in texts:
                tr = _new_translator()
                futures.append(pool.submit(tr.translate, t))
            for t, fut in zip(texts, futures):
                translated = None
                try:
                    translated = fut.result(timeout=45)
                except FTTimeout:
                    log.error("Google translate timed out — check network/proxy")
                except Exception as e:
                    log.error("Google translate failed: %s", e)
                if translated and str(translated).strip():
                    out.append(str(translated).strip())
                else:
                    # keep original so the bubble is never left empty
                    failures += 1
                    out.append(t)
        if failures == len(texts) and texts:
            log.error("Google Translate returned no translations for any of the "
                      "%d line(s) — output will show the original text. "
                      "Check network access / proxy.", len(texts))
        elif failures:
            log.warning("Google Translate failed for %d/%d line(s); those lines "
                        "keep their original text.", failures, len(texts))
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
            # never fall back to the *original* text — that makes the page
            # look untranslated; instead flag the miss loudly and keep the
            # model's best-effort output (empty here means "line missing")
            tr = results.get(i) or ""
            if not tr.strip():
                log.warning("LLM response did not include a translation for "
                            "line [%d]; it will be left as-is on the page.", i)
            out.append(tr)
        return out


_backend_cache: dict = {}


def get_backend(cfg: ProcessingConfig):
    """Instantiate the requested backend.

    Backends are cached keyed by their effective settings — previously a
    single global cache entry meant that changing the API key / model /
    target language in the UI had no effect: jobs kept running against the
    first (often unconfigured or stale) endpoint, so nothing was ever
    actually translated.
    """
    choice = cfg.translator
    if choice in ("auto", "llm"):
        try:
            key = ("llm", cfg.llm.base_url, cfg.llm.api_key, cfg.llm.model)
            if key not in _backend_cache:
                _backend_cache[key] = LLMBackend(cfg)
                log.info("Using LLM backend: %s (%s)", cfg.llm.model, cfg.llm.base_url)
            return _backend_cache[key]
        except Exception as e:
            if choice == "llm":
                raise
            log.info("LLM backend not configured (%s); using Google Translate.", e)
    gkey = ("google",)
    if gkey not in _backend_cache:
        _backend_cache[gkey] = GoogleBackend(cfg)
    return _backend_cache[gkey]


_cache = TranslationCache()


def translate_regions(regions, cfg: ProcessingConfig) -> None:
    """Translate every region **individually** and assign the result back to
    ``region.translation`` keyed by the region object itself.

    Two bugs used to break this stage:
      1. Regions were matched back to translations via ``r.text in resolved``,
         so two bubbles with identical OCR text shared one entry — and any
         region whose lookup failed kept an empty translation and was silently
         skipped by the renderer (the output looked untranslated).
      2. The cache key did not include the exact source text, so duplicated
         lines could collide.

    Now each region gets its own translation slot, cached per unique text,
    and any failure leaves that region's original text rendered in place
    instead of vanishing.
    """
    candidates = [r for r in regions
                  if r.text and r.text.strip() and len(r.text) >= cfg.min_text_length]
    if not candidates:
        log.warning("translate_regions: no OCR text found on this page "
                    "(nothing to translate / render)")
        return

    backend = get_backend(cfg)
    src, tgt = cfg.source_lang, cfg.target_lang
    log.info("Translating %d region(s) with '%s' backend: %s -> %s",
             len(candidates), backend.name, src, tgt)

    # One batch call per *unique* missing text, preserving order.
    missing: List[str] = []
    missing_keys: List[str] = []
    seen_missing = set()
    resolved: dict[str, str] = {}
    for r in candidates:
        t = r.text
        if t in resolved or t in seen_missing:
            continue
        key = _cache.key(t, src, tgt, backend.name)
        hit = _cache.get(key) if cfg.use_cache else None
        if hit:
            resolved[t] = hit
        else:
            seen_missing.add(t)
            missing.append(t)
            missing_keys.append(key)

    if missing:
        try:
            # LLM context window is fine with a full page; Google batches internally
            translated = backend.translate(missing, src, tgt)
        except Exception as e:
            log.error("Translation backend '%s' failed: %s — keeping original text",
                      backend.name, e)
            translated = list(missing)
        if len(translated) != len(missing):
            log.error("Backend returned %d lines for %d inputs; padding/truncating",
                      len(translated), len(missing))
            translated = (translated + list(missing))[:len(missing)]
        for t, key, tr in zip(missing, missing_keys, translated):
            tr = (tr or "").strip() or t   # never leave a bubble empty
            resolved[t] = tr
            _cache.put(key, tr)
        try:
            _cache.save()
        except Exception as e:
            log.warning("Could not persist translation cache: %s", e)

    assigned = 0
    for r in candidates:
        tr = resolved.get(r.text, "").strip()
        if tr:
            r.translation = tr
            assigned += 1
        else:
            # last-resort fallback: render the original text so the page
            # still shows something where the lettering used to be
            r.translation = r.text
    log.info("Translation complete: %d/%d region(s) translated (%s -> %s)",
             assigned, len(candidates), src, tgt)
