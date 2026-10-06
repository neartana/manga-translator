"""Translation stage.

Backends:
- ``llm``    : **bring your own AI key** — any OpenAI-compatible chat endpoint
               (OpenAI, Kimi/Moonshot, DeepSeek, OpenRouter, Groq, Azure
               OpenAI, Anthropic proxies, llama.cpp / LM Studio / Ollama local
               servers...). The whole page's text goes in one request so the
               model sees dialogue context.  Set the key/base URL/model in the
               web UI (saved to config.json), with ``--api-key/--base-url/
               --model`` on the CLI, or via MT_LLM_API_KEY & friends.
- ``google`` : free Google Translate.  Uses a direct client that batches all
               lines into a *single* request and automatically rotates the
               public endpoints when one rate-limits us (HTTP 429) — the old
               per-line deep-translator calls tripped Google's 5 req/s limit
               constantly ("Google Translate returned no translations").
- ``auto``   : use the LLM when an API key is configured, else Google; if the
               LLM errors out mid-job we fall back to Google and say so loudly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import List, Optional

from .config import ProcessingConfig, SUPPORTED_LANGUAGES

log = logging.getLogger("manga_translator")


class TranslationError(RuntimeError):
    """Raised when every translation backend failed for a batch."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail

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


def _google_lang(code: str) -> str:
    """Map our language codes onto Google's (e.g. 'zh-CN' -> 'zh-CN')."""
    return SUPPORTED_LANGUAGES.get(code, ("", code))[1]


class GoogleBackend:
    """Free Google Translate via a direct client.

    Deliberately does **not** use deep-translator's per-line API any more:
    firing one HTTP request per bubble blew past Google's ~5 req/s public
    limit and the endpoint answered 429 Too Many Requests, so every line
    silently kept its original text ("Google Translate returned no
    translations").  Now all lines go into ONE request (the free web endpoint
    accepts repeated ``q=`` params), chunks stay small, requests are spaced
    politely, and we rotate between the public endpoints when one rate-limits
    us.  Optional: set ``llm.google_api_key`` (or MT_GOOGLE_API_KEY) to use
    the paid Cloud Translation API, which never rate-limits like this.
    """
    name = "google"
    CHUNK = 40          # lines per request — well under URL length limits
    MIN_INTERVAL = 0.35  # seconds between requests (stay below ~3 req/s)

    def __init__(self, cfg: ProcessingConfig):
        self.cfg = cfg
        self._last_request = 0.0
        self._url_idx = 0

    # -- low-level request -------------------------------------------------
    def _endpoints(self) -> List[str]:
        urls = [u for u in (self.cfg.llm.google_fallback_urls or [])
                if isinstance(u, str) and u.startswith("https://")]
        return urls or ["https://translate.googleapis.com/translate_a/single"]

    def _throttle(self):
        wait = self.MIN_INTERVAL - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _request(self, texts: List[str], src: str, tgt: str) -> List[str]:
        import requests

        params = {"client": "gtx", "sl": src, "tl": tgt, "dt": "t"}
        if self.cfg.llm.google_api_key:
            params["key"] = self.cfg.llm.google_api_key
        for t in texts:
            params.setdefault("q", []).append(t)

        urls = self._endpoints()
        order = list(range(len(urls)))
        order.sort(key=lambda i: (i != self._url_idx, random.random()))
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.llm.max_retries + 1):
            for idx in order:
                url = urls[idx]
                try:
                    self._throttle()
                    resp = requests.get(url, params=params, timeout=(6.05, 30))
                    if resp.status_code == 429:
                        last_err = TranslationError(
                            "Google Translate is rate-limiting this server "
                            "(too many requests)")
                        continue                      # try the next endpoint
                    resp.raise_for_status()
                    data = json.loads(resp.text, strict=False)
                    self._url_idx = idx
                    return self._parse_payload(data, len(texts))
                except Exception as e:
                    last_err = e
                    log.debug("google endpoint %s failed: %s", url, e)
            if attempt < self.cfg.llm.max_retries:
                delay = min(8.0, 1.2 * (2 ** attempt)) + random.uniform(0, 0.7)
                log.warning("Google Translate failed (%s); retrying in %.1fs "
                            "(attempt %d/%d)", last_err, delay, attempt + 1,
                            self.cfg.llm.max_retries)
                time.sleep(delay)
        raise TranslationError(
            f"Google Translate request failed: {last_err}", detail=str(last_err))

    @staticmethod
    def _parse_payload(data, expected: int) -> List[str]:
        """Parse the gtx JSON array-of-arrays response into one string per q."""
        segs = [seg for seg in (data[0] or []) if isinstance(seg, list) and seg[0]]
        by_q: dict[int, List[str]] = {}
        plain: List[str] = []
        for seg in segs:
            txt = seg[0]
            src_orig = seg[2] if len(seg) > 2 else None
            if isinstance(src_orig, int):     # segment belongs to the q-th input
                by_q.setdefault(src_orig, []).append(txt)
            else:
                plain.append(txt)
        if by_q:
            out = ["".join(by_q.get(i, [])).strip() for i in range(expected)]
            return out
        joined = "".join(plain).strip()       # single implicit query
        return [joined] + [""] * (expected - 1)

    # -- public API ---------------------------------------------------------
    def translate(self, texts: List[str], src: str, tgt: str) -> List[str]:
        src_code = _google_lang(src)
        tgt_code = _google_lang(tgt)
        if not tgt_code or tgt_code == "auto":
            raise TranslationError(f"Unsupported target language: {tgt!r}")

        out: List[str] = []
        for i in range(0, len(texts), self.CHUNK):
            chunk = texts[i:i + self.CHUNK]
            got = self._request(chunk, src_code, tgt_code)
            if len(got) < len(chunk):
                got += [""] * (len(chunk) - len(got))
            out.extend(g[:len(chunk)])
        missing = sum(1 for t, tr in zip(texts, out) if not (tr or "").strip())
        if missing == len(texts) and texts:
            raise TranslationError(
                "Google Translate returned no translations (it may be "
                "rate-limiting this server)")
        if missing:
            log.warning("Google Translate returned nothing for %d/%d line(s); "
                        "those keep their original text.", missing, len(texts))
        return out


class LLMBackend:
    name = "llm"

    def __init__(self, cfg: ProcessingConfig):
        self.cfg = cfg.llm
        if not self.cfg.api_key:
            raise RuntimeError("No LLM API key configured")

    def _endpoint(self) -> str:
        """Normalise the user-supplied base URL into a chat-completions URL.

        Accepts anything people actually paste in — ``.../v1``, ``.../v1/``,
        the full ``.../chat/completions`` URL, or an Azure deployment URL —
        instead of blindly appending and producing broken endpoints.
        """
        url = (self.cfg.base_url or "").strip()
        if not url:
            url = "https://api.openai.com/v1"
        if not url.startswith(("http://", "https://")):
            url = "https://" + url          # users often paste "api.foo.com/v1"
        low = url.lower().rstrip("/")
        if low.endswith("/chat/completions"):
            return url.rstrip("/")
        if "deployments/" in low:            # Azure OpenAI style
            return f"{url.rstrip('/')}/chat/completions"
        return url.rstrip("/") + "/chat/completions"

    @staticmethod
    def _friendly_http_error(resp) -> str:
        """Turn raw provider errors into actionable messages."""
        status = resp.status_code
        try:
            payload = resp.json()
            detail = (payload.get("error") or {}).get("message") \
                or payload.get("message") or json.dumps(payload)[:300]
        except Exception:
            detail = (resp.text or "").strip()[:300]
        hints = {
            401: ("the API key was rejected — check it is valid for this "
                  "provider and has no stray spaces"),
            403: ("access forbidden — your key may not be allowed for this "
                  "model/endpoint, or the gateway needs different auth; "
                  "verify the key, model name and base URL in Settings → AI"),
            404: ("model or endpoint not found — double-check the model name "
                  "and base URL"),
            429: ("rate limited or out of quota on the provider side — wait a "
                  "moment or check your billing/quota"),
        }
        msg = f"HTTP {status} from {resp.url}"
        if detail:
            msg += f": {detail}"
        hint = hints.get(status)
        if hint:
            msg += f". Hint: {hint}"
        return msg

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
        url = self._endpoint()
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=120)
                if resp.status_code in (401, 403, 404):
                    # deterministic auth/config errors — retrying is pointless
                    raise TranslationError(self._friendly_http_error(resp))
                resp.raise_for_status()
                payload = resp.json()
                try:
                    content = payload["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    raise TranslationError(
                        f"Unexpected response shape from {url}: "
                        f"{json.dumps(payload)[:300]}")
                return self._parse(content, texts)
            except TranslationError:
                raise
            except Exception as e:
                last_err = e
                log.warning("LLM translate attempt %d failed: %s", attempt + 1, e)
                if attempt < self.cfg.max_retries:
                    time.sleep(min(6.0, 1.5 * (attempt + 1)))
        raise TranslationError(f"LLM translation failed: {last_err}",
                               detail=str(last_err))

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
            tr = results.get(i) or ""
            if not tr.strip():
                log.warning("LLM response did not include a translation for "
                            "line [%d]; it will be retried on its own.", i)
            out.append(tr)
        return out


_backend_cache: dict = {}


def get_backend(cfg: ProcessingConfig):
    """Instantiate the requested backend.

    Backends are cached keyed by their effective settings so changing the
    API key / model / base URL in the UI takes effect immediately instead of
    keeping a stale first-built endpoint alive for the whole server process.
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
    gkey = ("google", cfg.llm.google_api_key, tuple(cfg.llm.google_fallback_urls))
    if gkey not in _backend_cache:
        _backend_cache[gkey] = GoogleBackend(cfg)
    return _backend_cache[gkey]


def test_llm_connection(llm) -> str:
    """Minimal round-trip against a user's AI endpoint; returns the reply text.

    Used by the web UI's "Test connection" button so a bad key/URL/model is
    caught immediately with a clear message instead of after a full page run.
    """
    import requests
    if not llm.api_key:
        raise TranslationError("No API key entered")
    if not llm.model:
        raise TranslationError("No model name entered")
    backend = LLMBackend.__new__(LLMBackend)   # skip __init__ key check here
    backend.cfg = llm
    url = backend._endpoint()
    resp = requests.post(url,
                         headers={"Authorization": f"Bearer {llm.api_key}",
                                  "Content-Type": "application/json"},
                         json={"model": llm.model, "max_tokens": 16,
                               "messages": [{"role": "user",
                                             "content": "Reply with exactly: OK"}]},
                         timeout=30)
    if resp.status_code != 200:
        raise TranslationError(backend._friendly_http_error(resp))
    try:
        return (resp.json()["choices"][0]["message"]["content"] or "").strip() \
            or "(empty reply)"
    except Exception:
        return "(unexpected response shape)"


_cache = TranslationCache()


def translate_regions(regions, cfg: ProcessingConfig) -> None:
    """Translate every region and assign the result back to
    ``region.translation`` — one slot per region, never empty, never silently
    skipped.

    Resilience rules (all of these used to fail silently and leave pages
    looking untranslated):
      * one batch call per **unique** missing text, assigned per region so
        duplicate bubbles can't collide;
      * if the LLM endpoint errors out mid-job, individual lines are retried
        once, then the whole remaining batch falls back to Google Translate;
      * only if *every* backend fails do we keep the original text — and that
        raises a clear TranslationError so the UI shows exactly why.
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

    def _store(texts: List[str], keys: List[str], translated: List[str]) -> int:
        ok = 0
        for t, key, tr in zip(texts, keys, translated):
            tr = (tr or "").strip() or t   # never leave a bubble empty
            if tr != t:
                ok += 1
            resolved[t] = tr
            _cache.put(key, tr)
        return ok

    if missing:
        primary_err: Optional[Exception] = None
        try:
            translated = backend.translate(missing, src, tgt)
            if len(translated) != len(missing):
                log.error("Backend returned %d lines for %d inputs; padding/truncating",
                          len(translated), len(missing))
                translated = (translated + [""] * len(missing))[:len(missing)]
            # retry the lines the model simply didn't answer, individually
            gaps = [i for i, (t, tr) in enumerate(zip(missing, translated))
                    if not (tr or "").strip()]
            if gaps:
                log.warning("Retrying %d missed line(s) individually via '%s'",
                            len(gaps), backend.name)
                for i in gaps:
                    try:
                        one = backend.translate([missing[i]], src, tgt)[0]
                    except Exception as e:
                        log.warning("Single-line retry failed: %s", e)
                        one = ""
                    if (one or "").strip():
                        translated[i] = one
            _store(missing, missing_keys, translated)
        except Exception as e:
            primary_err = e
            log.error("Translation backend '%s' failed: %s", backend.name, e)

        if primary_err is not None:
            # fall back to Google (or vice-versa) instead of giving up
            alt: Optional[object] = None
            try:
                if backend.name == "google":
                    alt_key = ("llm", cfg.llm.base_url, cfg.llm.api_key, cfg.llm.model)
                    if cfg.llm.api_key:
                        alt = _backend_cache.get(alt_key) or LLMBackend(cfg)
                else:
                    gk = ("google", cfg.llm.google_api_key,
                          tuple(cfg.llm.google_fallback_urls))
                    if gk not in _backend_cache:
                        _backend_cache[gk] = GoogleBackend(cfg)
                    alt = _backend_cache[gk]
            except Exception as e2:
                log.error("Fallback backend could not start: %s", e2)
            got = 0
            if alt is not None:
                try:
                    log.info("Falling back to '%s' backend after '%s' failed",
                             alt.name, backend.name)
                    translated = alt.translate(missing, src, tgt)
                    if len(translated) != len(missing):
                        translated = (translated + [""] * len(missing))[:len(missing)]
                    got = _store(missing, missing_keys, translated)
                except Exception as e2:
                    log.error("Fallback backend '%s' also failed: %s", alt.name, e2)
            if got == 0:
                raise TranslationError(
                    f"Translation failed: {primary_err}"
                    + (f" | Fallback ({getattr(alt, 'name', '?')}): "
                       f"{_short(alt)} — add or fix your AI API key in Settings,"
                       " or try again in a minute." if alt is not None else ""),
                    detail=str(primary_err))

    try:
        _cache.save()
    except Exception as e:
        log.warning("Could not persist translation cache: %s", e)

    assigned = 0
    for r in candidates:
        tr = resolved.get(r.text, "").strip()
        if tr and tr != r.text:
            r.translation = tr
            assigned += 1
        else:
            # last-resort fallback: render the original text so the page
            # still shows something where the lettering used to be
            r.translation = r.text
    log.info("Translation complete: %d/%d region(s) translated (%s -> %s)",
             assigned, len(candidates), src, tgt)


def _short(obj) -> str:
    try:
        return str(obj)[:200]
    except Exception:
        return "?"
