"""FastAPI web server: upload pages, watch progress, download results.

Run:  python -m manga_translator --serve   (or: uvicorn server.app:create_app)
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from manga_translator.batch import process_input, zip_results
from manga_translator.config import (ProcessingConfig, SUPPORTED_LANGUAGES,
                                     TARGET_LANGUAGES, PROJECT_ROOT)
from manga_translator.fonts import list_fonts

log = logging.getLogger("manga_translator.web")

STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"


class Job:
    def __init__(self, job_id: str, cfg: ProcessingConfig):
        self.id = job_id
        self.cfg = cfg
        self.status = "queued"          # queued | running | done | error
        self.stage = ""                 # current per-page stage label
        self.stage_frac = 0.0           # 0..1 within the current page
        self.total = 0
        self.done = 0
        self.current = ""
        self.error = ""
        self.results: List[dict] = []
        self.output_dir = RUNS_DIR / job_id / "translated"
        self.upload_dir = RUNS_DIR / job_id / "uploads"
        self.created = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id, "status": self.status, "stage": self.stage,
            "stage_frac": self.stage_frac,
            "total": self.total, "done": self.done, "current": self.current,
            "error": self.error, "results": self.results,
        }


def _run_job(job: Job):
    job.status = "running"
    try:
        input_path = _consolidate_uploads(job)

        def progress(done, total, current):
            job.done, job.total, job.current = done, total, current
            if done >= total and total:
                job.stage, job.stage_frac = "finished", 1.0

        def page_progress(stage, frac):
            # called from the pipeline for each page — this is what makes the
            # UI actually show "translating…" instead of a frozen bar
            job.stage, job.stage_frac = stage, frac

        results = process_input(input_path, job.output_dir, job.cfg,
                                progress=progress, page_progress=page_progress)
        job.done = job.total = len(results)
        job.stage, job.stage_frac = "finished", 1.0
        for r in results:
            if r.error:
                job.results.append({"src": Path(r.source_path).name, "error": r.error})
            elif r.output_path:
                rel = str(Path(r.output_path).relative_to(job.output_dir))
                translated_n = sum(1 for x in r.regions
                                   if x.text and x.translation and x.translation != x.text)
                job.results.append({
                    "src": Path(r.source_path).name,
                    "out": rel,
                    "url": f"/api/jobs/{job.id}/files/{rel}",
                    "regions": len(r.regions),
                    "translated": translated_n,
                    "texts": [{"src": x.text, "tgt": x.translation} for x in r.regions if x.text],
                })
        failures = [r for r in job.results if "error" in r]
        if failures and len(failures) == len(job.results):
            job.status, job.error = "error", "All pages failed: " + failures[0]["error"]
        else:
            job.status = "done"
            ok = [r for r in results if r.output_path]
            if ok:
                zip_results(results, job.output_dir, RUNS_DIR / job.id / "translated.zip")
    except Exception as e:
        log.exception("job %s failed", job.id)
        job.status, job.error = "error", str(e)


def _consolidate_uploads(job: Job) -> Path:
    """Return a single input path for the batch coordinator.

    One file -> that file; several -> the upload folder. If the single
    upload is a zip it is passed straight through.
    """
    files = [p for p in job.upload_dir.rglob("*") if p.is_file()]
    if not files:
        raise RuntimeError("No files uploaded")
    if len(files) == 1:
        return files[0]
    return job.upload_dir


def create_app() -> FastAPI:
    app = FastAPI(title="Manga Translator", version="0.1.0")
    jobs: Dict[str, Job] = {}
    base_cfg = ProcessingConfig.load()

    @app.get("/api/meta")
    def meta():
        cfg = ProcessingConfig.load()
        gpu = cfg.resolved_device()
        return {
            "device": gpu,
            "gpu_available": gpu != "cpu",
            "llm_configured": bool(cfg.llm.api_key),
            "llm_model": cfg.llm.model,
            "source_languages": [{"code": k, "name": v[0]} for k, v in SUPPORTED_LANGUAGES.items()],
            "target_languages": [{"code": k, "name": v[0]} for k, v in TARGET_LANGUAGES.items()],
            "font_packs": list_fonts(cfg.font_dir),
        }

    @app.post("/api/jobs")
    async def create_job(
        files: List[UploadFile] = File(...),
        source_lang: str = Form("auto"),
        target_lang: str = Form("en"),
        translator: str = Form("auto"),
        direction: str = Form("auto"),
        uppercase: bool = Form(False),
        device: str = Form("auto"),
        font_path: str = Form(""),
        llm_api_key: str = Form(""),
        llm_base_url: str = Form(""),
        llm_model: str = Form(""),
    ):
        job_id = uuid.uuid4().hex[:12]
        cfg = ProcessingConfig.load()
        cfg.source_lang, cfg.target_lang = source_lang, target_lang
        cfg.translator, cfg.direction = translator, direction
        cfg.uppercase, cfg.device = uppercase, device
        if font_path:
            cfg.font_path = font_path
        if llm_api_key:
            cfg.llm.api_key = llm_api_key
        if llm_base_url:
            cfg.llm.base_url = llm_base_url
        if llm_model:
            cfg.llm.model = llm_model

        job = Job(job_id, cfg)
        job.upload_dir.mkdir(parents=True, exist_ok=True)
        for uf in files:
            # browsers send folder uploads with relative paths as the filename
            rel = Path(uf.filename or uf.name or "page.png")
            if rel.is_absolute() or ".." in rel.parts:
                rel = Path(rel.name)
            dest = job.upload_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                shutil.copyfileobj(uf.file, f)

        jobs[job_id] = job
        threading.Thread(target=_run_job, args=(job,), daemon=True).start()
        return {"job_id": job_id}

    @app.post("/api/test-connection")
    def test_connection(body: dict = Body(...)):
        """Ping the user's AI endpoint so a bad key/URL/model fails fast."""
        from manga_translator.config import TranslatorConfig
        from manga_translator.translation import test_llm_connection, TranslationError
        base = ProcessingConfig.load()
        tc = TranslatorConfig(
            base_url=str(body.get("llm_base_url") or "") or base.llm.base_url,
            api_key=str(body.get("llm_api_key") or "") or base.llm.api_key,
            model=str(body.get("llm_model") or "") or base.llm.model,
        )
        try:
            reply = test_llm_connection(tc)
            return {"ok": True, "reply": reply}
        except TranslationError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=200)
        except Exception as e:  # network errors etc.
            return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"},
                                status_code=200)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job.to_dict()

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str):
        zp = RUNS_DIR / job_id / "translated.zip"
        if not zp.exists():
            raise HTTPException(404, "archive not ready")
        return FileResponse(zp, filename=f"manga-translated-{job_id}.zip")

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def result_file(job_id: str, path: str):
        job_dir = (RUNS_DIR / job_id / "translated").resolve()
        fp = (job_dir / path).resolve()
        if not str(fp).startswith(str(job_dir)) or not fp.exists():
            raise HTTPException(404, "file not found")
        return FileResponse(fp)

    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


app = create_app()
