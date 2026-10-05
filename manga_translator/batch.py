"""Batch coordination — files, folders and zip archives.

The input folder structure is mirrored under the output directory, so
``chapter01/page03.png`` lands at ``<out>/chapter01/page03.png``.
Results can also be zipped back up for convenient download.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, List, Optional

from .config import ProcessingConfig
from .pipeline import TranslationPipeline
from .utils import PageResult, collect_images, extract_zip, is_image

log = logging.getLogger("manga_translator")

BatchProgress = Optional[Callable[[int, int, str], None]]  # (done, total, current file)


def _output_name(src: Path, cfg: ProcessingConfig) -> Path:
    if cfg.output_format == "auto":
        return src.name
    return src.stem + "." + ("jpg" if cfg.output_format == "jpeg" else cfg.output_format)


def process_input(input_path: Path, output_dir: Path, cfg: ProcessingConfig,
                  progress: BatchProgress = None) -> List[PageResult]:
    """Translate a single image, a folder, or a zip of images/folders."""
    pipeline = TranslationPipeline(cfg)
    results: List[PageResult] = []

    with tempfile.TemporaryDirectory(prefix="mt_") as tmp:
        tmp_path = Path(tmp)
        root: Path
        cleanup_root = False

        if input_path.suffix.lower() == ".zip":
            root = extract_zip(input_path, tmp_path)
            cleanup_root = True
        elif input_path.is_dir():
            root = input_path
        elif is_image(input_path):
            root = tmp_path / "_single"
            root.mkdir(exist_ok=True)
            shutil.copy2(input_path, root / input_path.name)
            cleanup_root = True
        else:
            raise ValueError(f"Unsupported input: {input_path}")

        images = collect_images(root)
        total = len(images)
        log.info("Found %d image(s) under %s", total, root)

        for i, src in enumerate(images, 1):
            rel = src.relative_to(root)
            # flatten the temp wrapper folder for single-file / zip-stem levels
            parts = rel.parts
            if parts and parts[0] in ("_single",):
                rel = Path(*parts[1:])
            dst = output_dir / rel.parent / _output_name(src, cfg)

            result = PageResult(source_path=str(src))
            if progress:
                progress(i - 1, total, str(rel))
            try:
                if dst.exists() and not cfg.overwrite:
                    result.output_path = str(dst)
                else:
                    debug_dir = (dst.parent / f".debug_{src.stem}") if cfg.save_intermediates else None
                    regions = pipeline.process_file(
                        src, dst,
                        progress=lambda stage, frac: None,
                        debug_dir=debug_dir)
                    if not regions and cfg.skip_no_text:
                        if dst.exists():
                            dst.unlink()
                        result.output_path = ""
                    else:
                        result.output_path = str(dst)
                    result.regions = regions
            except Exception as e:
                log.exception("Failed on %s", src)
                result.error = str(e)
            results.append(result)
            if progress:
                progress(i, total, str(rel))

    return results


def zip_results(results: List[PageResult], output_dir: Path, zip_path: Path) -> Path:
    """Pack all produced files (structure preserved) into one archive."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            if r.output_path:
                p = Path(r.output_path)
                zf.write(p, arcname=str(p.relative_to(output_dir)))
    return zip_path
