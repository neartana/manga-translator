"""Command-line interface.

Examples
--------
Translate one image:
    python -m manga_translator page.png -l en

Translate a whole folder (structure preserved), Japanese -> English on GPU:
    python -m manga_translator ./manga/chapter1 -o ./out -s ja -l en --device cuda

Translate a zip and zip the results back:
    python -m manga_translator chapter.zip -o ./out --zip

Launch the web UI instead:
    python -m manga_translator --serve
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import zipfile
from pathlib import Path

from .config import ProcessingConfig, SUPPORTED_LANGUAGES, TARGET_LANGUAGES
from .batch import process_input, zip_results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manga_translator",
        description="Translate manga/comic pages: detect text, clean, translate, re-letter.")
    p.add_argument("input", nargs="?", help="Image file, folder, or .zip archive")
    p.add_argument("-o", "--output", help="Output directory (default: ./outputs/<input>-translated)")
    p.add_argument("-s", "--source-lang", default=None,
                   choices=list(SUPPORTED_LANGUAGES), help="Source language (default: auto)")
    p.add_argument("-l", "--target-lang", default=None,
                   choices=list(TARGET_LANGUAGES), help="Target language (default: en)")
    p.add_argument("--translator", choices=["auto", "llm", "google"], default=None,
                   help="Translation backend")
    p.add_argument("--detector", choices=["auto", "yolo", "opencv"], default=None)
    p.add_argument("--ocr", choices=["auto", "manga-ocr", "easyocr"], default=None)
    p.add_argument("--inpainter", choices=["auto", "lama", "opencv"], default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    p.add_argument("--direction", choices=["auto", "horizontal", "vertical"], default=None,
                   help="Rendered text direction")
    p.add_argument("--font-path", default=None, help="Explicit font file (overrides packs)")
    p.add_argument("--font-dir", default=None, help="Font-pack directory")
    p.add_argument("--font-size-offset", type=int, default=None)
    p.add_argument("--uppercase", action="store_true", help="Render text in ALL CAPS")
    p.add_argument("--llm-base-url", default=None, help="OpenAI-compatible endpoint URL")
    p.add_argument("--llm-api-key", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--mask-dilation", type=int, default=None)
    p.add_argument("--output-format", choices=["auto", "png", "jpg", "webp"], default=None)
    p.add_argument("--zip", dest="make_zip", action="store_true",
                   help="Also write a <output>.zip with all results")
    p.add_argument("--skip-no-text", action="store_true",
                   help="Don't save pages where no text was found")
    p.add_argument("--no-overwrite", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--save-intermediates", action="store_true",
                   help="Save masks and cleaned pages next to results (debugging)")
    p.add_argument("--serve", action="store_true", help="Start the web UI and exit")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def apply_args(cfg: ProcessingConfig, args: argparse.Namespace) -> ProcessingConfig:
    for key in ("source_lang", "target_lang", "translator", "detector", "ocr",
                "inpainter", "device", "direction", "font_path", "font_dir",
                "font_size_offset", "mask_dilation", "output_format"):
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg, key, val)
    if args.uppercase:
        cfg.uppercase = True
    if args.skip_no_text:
        cfg.skip_no_text = True
    if args.no_overwrite:
        cfg.overwrite = False
    if args.no_cache:
        cfg.use_cache = False
    if args.save_intermediates:
        cfg.save_intermediates = True
    if args.llm_base_url:
        cfg.llm.base_url = args.llm_base_url
    if args.llm_api_key:
        cfg.llm.api_key = args.llm_api_key
    if args.llm_model:
        cfg.llm.model = args.llm_model
    return cfg


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    if args.serve:
        import uvicorn
        from server.app import create_app
        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0

    if not args.input:
        build_parser().print_help()
        return 1

    cfg = apply_args(ProcessingConfig.load(), args)
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 1

    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        output_dir = Path(cfg.output_dir) / f"{input_path.stem}-translated"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {cfg.resolved_device()}  |  {cfg.source_lang} -> {cfg.target_lang}  "
          f"|  translator: {cfg.translator}")

    t0 = time.time()

    def progress(done: int, total: int, current: str):
        print(f"\r[{done}/{total}] {current[:60]:<60}", end="", flush=True)

    results = process_input(input_path, output_dir, cfg, progress=progress)
    print()

    ok = [r for r in results if not r.error]
    failed = [r for r in results if r.error]
    for r in ok:
        n = len(r.regions)
        print(f"  ✓ {Path(r.source_path).name} -> {r.output_path}  ({n} text regions)")
    for r in failed:
        print(f"  ✗ {Path(r.source_path).name}: {r.error}", file=sys.stderr)

    if args.make_zip and ok:
        zp = zip_results(results, output_dir, output_dir.with_suffix(".zip"))
        print(f"Archive: {zp}")

    print(f"Done in {time.time() - t0:.1f}s — {len(ok)} ok, {len(failed)} failed. "
          f"Output: {output_dir}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
