# Manga Translator — 漫画翻訳

One-click manga & comic page translation. Drop in a page, a folder, or a
zip — get back cleaned, re-lettered pages with the translation typeset
where the original text used to be.

**Pipeline:** detect text & speech bubbles → OCR → remove original
lettering (clean + neural inpaint) → translate → render with auto-fitted
fonts.

- 🖥 **Web UI** and ⌨ **CLI**, same engine
- 📦 Single image, folder, or `.zip` batches — **folder structure preserved**
- 🗣 19 target languages; LLM (OpenAI-compatible) or free Google Translate
- 🔤 Custom **font packs** per language — just drop `.ttf` files into `fonts/<lang>/`
- ⚡ GPU (CUDA / Apple MPS) when available, pure **CPU mode** otherwise
- 🧩 Graceful degradation: every deep-learning stage has a classical
  fallback, so even a minimal install works end to end

---

## Quick start

```bash
pip install -r requirements.txt          # core (lightweight, CPU)
pip install -r requirements-ml.txt       # full quality: YOLO, manga-ocr, LaMa, EasyOCR
python scripts/download_fonts.py         # starter font packs (OFL fonts)

# Web UI
python -m manga_translator --serve       # → http://localhost:7860

# CLI
python -m manga_translator page.png -l en
python -m manga_translator ./chapter1 -o ./out -s ja -l en --zip
python -m manga_translator chapter.zip -l fr
```

Models download automatically on first use (into `./models`).

### GPU

Install the CUDA build of PyTorch **before** `requirements-ml.txt`
(see https://pytorch.org/get-started/locally/), e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-ml.txt
python -m manga_translator ./chapter1 -l en --device cuda
```

`--device auto` (default) picks CUDA → MPS → CPU in that order.

### Docker

```bash
docker build -t manga-translator .                              # light CPU image
docker build --build-arg INSTALL_ML=1 -t manga-translator:ml .  # full ML image
docker run -p 7860:7860 -v $PWD/outputs:/app/outputs manga-translator
# GPU: docker run --gpus all -p 7860:7860 manga-translator:ml
```

---

## Translation engines

| Engine | Quality | Needs |
|---|---|---|
| `llm` | Best — whole-page context, natural dialogue | Any OpenAI-compatible endpoint + key |
| `google` | Good, free | nothing |

`auto` (default) uses the LLM when a key is configured, else Google.

Configure the LLM via the UI (Advanced section), CLI flags, config file,
or environment variables:

```bash
export MT_LLM_BASE_URL="https://api.moonshot.ai/v1"   # or OpenAI, DeepSeek, LM Studio…
export MT_LLM_API_KEY="sk-…"
export MT_LLM_MODEL="kimi-k2"                          # any chat model
```

Local servers work too: `--llm-base-url http://localhost:1234/v1`
(LM Studio, llama.cpp, Ollama's OpenAI shim…).

## Models used

| Stage | Full install | Lightweight fallback |
|---|---|---|
| Detection | `ogkalu/comic-text-and-bubble-detector` (YOLO) | OpenCV threshold + contours |
| OCR (Japanese) | `manga-ocr` (vertical text, furigana) | — |
| OCR (other) | EasyOCR (~80 languages) | RapidOCR (ONNX, no torch) |
| Cleaning / inpaint | LaMa (`simple-lama-inpainting`) | OpenCV Telea + bubble flat-fill |
| Translation | your LLM | Google Translate |

Text **inside bubbles** and **free-floating text** (SFX, narration boxes)
are both picked up; bubble interiors are flat-filled before inpainting so
balloons come out perfectly clean.

## CLI reference

```
positional:
  input                  image file, folder, or .zip

options:
  -o, --output DIR       output directory
  -s, --source-lang XX   auto | ja | en | zh-CN | …   (default: auto)
  -l, --target-lang XX   language to translate into   (default: en)
  --translator           auto | llm | google
  --detector             auto | yolo | opencv
  --ocr                  auto | manga-ocr | easyocr
  --inpainter            auto | lama | opencv
  --device               auto | cpu | cuda | mps
  --direction            auto | horizontal | vertical
  --font-path FILE       explicit font (overrides packs)
  --font-dir DIR         font-pack root (default: ./fonts)
  --font-size-offset N   grow/shrink rendered text
  --uppercase            ALL CAPS lettering
  --llm-base-url / --llm-api-key / --llm-model
  --mask-dilation N      px to expand the text mask before cleaning (default 12)
  --output-format        auto | png | jpg | webp
  --zip                  also write <output>.zip
  --skip-no-text         don't save pages without detected text
  --no-overwrite         skip pages that already exist in the output
  --no-cache             disable the translation cache
  --save-intermediates   keep masks & cleaned pages (debugging)
  --serve [--port 7860]  start the web UI
```

## Custom font packs

See [fonts/README.md](fonts/README.md). Short version: put `.ttf` files in
`fonts/en/`, `fonts/zh-CN/`, `fonts/ko/`, … and the renderer uses them for
that target language automatically.

## Output layout

```
outputs/
  chapter1-translated/          # mirrors the input folder tree
    chapter1/
      page03.png
  runs/<job-id>/                # web UI jobs
    uploads/  translated/  translated.zip
  .translation_cache.json       # repeat pages are free
```

## Configuration file

The UI persists settings to `config.json` in the project root; the CLI
reads the same file and its flags override it.

## Troubleshooting

- **Missing text** — pages at very low resolution: try `--detector yolo`
  and/or upscale the page; very stylised SFX sometimes escapes detection.
- **Leftover ink** — raise `--mask-dilation` (e.g. 18–24).
- **Text too small/large** — `--font-size-offset ±N`.
- **LLM errors** — check base URL/key; the job falls back per line only for
  Google, so a dead LLM endpoint fails loudly (by design).
