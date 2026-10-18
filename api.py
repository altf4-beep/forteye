# -*- coding: utf-8 -*-
"""
FORTEYE API: two-step asynchronous REST service (100 % local).

    POST /api/v1/analyze        X-API-Key, form field `file=@video.mov`  ->  {"job_id": "<uuid>"}
    GET  /api/v1/jobs/{job_id}  X-API-Key  ->  {job_id, status, verdict, violations[...]}

Extras (same key): GET /api/v1/jobs/{id}/progress, GET /api/v1/rules. Open: GET /health, GET / (web UI).

Run:   FORTEYE_API_KEY=<secret> uvicorn api:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
import secrets
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

import pipeline

logging.basicConfig(level=os.getenv("FORTEYE_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOG = logging.getLogger("forteye.api")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path(os.getenv("FORTEYE_DATA_DIR", BASE_DIR / "data" / "jobs")).resolve()
API_KEY = os.getenv("FORTEYE_API_KEY", "forteye-secret")
MAX_UPLOAD_MB = int(os.getenv("FORTEYE_MAX_UPLOAD_MB", "4096"))
WORKERS = max(1, int(os.getenv("FORTEYE_WORKERS", "1")))  # 1 = one video on the GPU at a time
KEEP_VIDEOS = os.getenv("FORTEYE_KEEP_VIDEOS", "0").lower() in {"1", "true", "yes"}
ALLOWED_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm"}
if API_KEY == "forteye-secret":
    LOG.warning("FORTEYE_API_KEY is not set: using the default key. Set your own secret before real use.")


# --------------------------------------------------------------------------- #
# strict response schemas (extra fields are rejected on the way in and never leak out)
# --------------------------------------------------------------------------- #
class EvidenceFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp_sec: float = Field(ge=0)
    image_base64: str = Field(min_length=1)


class Violation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = Field(min_length=1)
    timecode_sec: float = Field(ge=0)
    description: str = Field(min_length=1)
    evidence_frames: List[EvidenceFrame]


class JobResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Optional[Literal["no_violation", "violation"]] = None
    violations: List[Violation] = Field(default_factory=list)
    error: Optional[str] = None  # only present when status == "failed"


class JobCreated(BaseModel):
    job_id: str


class JobProgress(BaseModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    progress: float
    stage: str


# --------------------------------------------------------------------------- #
# job store (memory + result.json on disk, survives restarts)
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    filename: str = ""
    status: str = "processing"
    verdict: Optional[str] = None
    violations: List[dict] = field(default_factory=list)
    error: Optional[str] = None
    progress: float = 0.0
    stage: str = "В очереди"
    created: float = field(default_factory=time.time)


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()
        self._restore()

    def _restore(self) -> None:
        for d in self.root.iterdir():
            f = d / "result.json"
            if not (d.is_dir() and f.is_file()):
                continue
            try:
                job = Job(**json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                LOG.warning("skipping unreadable job file %s", f)
                continue
            if job.status == "processing":  # the server died mid-analysis
                job.status, job.error, job.stage = "failed", "Server restarted during analysis", "Прервано"
                self._write(job)
            self._jobs[job.id] = job

    def _write(self, job: Job) -> None:
        d = self.root / job.id
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "result.json.tmp"
        tmp.write_text(json.dumps(asdict(job), ensure_ascii=False), encoding="utf-8")
        tmp.replace(d / "result.json")

    def create(self, filename: str) -> Job:
        job = Job(id=str(uuid.uuid4()), filename=filename)
        with self._lock:
            self._jobs[job.id] = job
            self._write(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def progress(self, job_id: str, frac: float, stage: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == "processing":
                job.progress, job.stage = round(max(0.0, min(1.0, frac)), 3), stage

    def finish(self, job_id: str, verdict: str, violations: List[dict]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status, job.verdict, job.violations = "done", verdict, violations
            job.progress, job.stage, job.error = 1.0, "Готово", None
            self._write(job)

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status, job.error, job.stage = "failed", error, "Ошибка"
            self._write(job)

    def discard(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(self.root / job_id, ignore_errors=True)


store = JobStore(DATA_DIR)
executor = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="forteye-worker")
warmup = {"state": "pending", "error": None}


def _run_job(job_id: str, video: Path) -> None:
    try:
        engine = pipeline.get_engine()
        result = engine.analyze(str(video), progress=lambda frac, stage: store.progress(job_id, frac, stage))
        checked = JobResult(job_id=job_id, status="done", verdict=result["verdict"], violations=result["violations"])
        store.finish(job_id, checked.verdict, [v.model_dump() for v in checked.violations])
        LOG.info("job %s done: %s, %d violation(s)", job_id, checked.verdict, len(checked.violations))
    except Exception as exc:  # noqa: BLE001: any failure must end in status=failed, never a hung job
        LOG.exception("job %s failed", job_id)
        store.fail(job_id, f"{type(exc).__name__}: {exc}")
    finally:
        if not KEEP_VIDEOS:
            video.unlink(missing_ok=True)


def _warmup() -> None:
    try:
        engine = pipeline.get_engine()
        engine.load()
        # "degraded" = models loaded, but the VLM folder was missing, so only YOLO-based rules can fire
        warmup["state"] = "degraded" if engine.vlm_error else "ready"
        warmup["error"] = engine.vlm_error
        if engine.vlm_error:
            LOG.warning("running without VLM: %s", engine.vlm_error)
    except Exception as exc:  # noqa: BLE001
        warmup.update(state="error", error=f"{type(exc).__name__}: {exc}")
        LOG.error("model warm-up failed: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    executor.submit(_warmup)  # load YOLO + VLM in the background so the first job is not slow
    yield
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="FORTEYE", version="1.0", description="Local cashier-discipline violation detection",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("FORTEYE_CORS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])


def require_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key",
                            headers={"WWW-Authenticate": "ApiKey"})


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
@app.post("/api/v1/analyze", response_model=JobCreated, dependencies=[Depends(require_key)])
def analyze(file: UploadFile = File(...)) -> JobCreated:
    name = Path(file.filename or "video.mov").name
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")
    job = store.create(name)
    dest = DATA_DIR / job.id / f"video{ext}"
    size, limit = 0, MAX_UPLOAD_MB << 20
    try:
        with dest.open("wb") as out:
            while True:
                chunk = file.file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Empty file")
    except HTTPException:
        store.discard(job.id)
        raise
    except OSError as exc:
        store.discard(job.id)
        raise HTTPException(status_code=500, detail=f"Cannot store upload: {exc}") from exc
    executor.submit(_run_job, job.id, dest)
    LOG.info("job %s queued: %s (%.1f MB)", job.id, name, size / 1048576)
    return JobCreated(job_id=job.id)


def _get(job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/v1/jobs/{job_id}", response_model=JobResult, response_model_exclude_none=True,
         dependencies=[Depends(require_key)])
def get_job(job_id: str) -> JobResult:
    job = _get(job_id)
    done = job.status == "done"
    return JobResult(job_id=job.id, status=job.status, verdict=job.verdict if done else None,
                     violations=job.violations if done else [], error=job.error if job.status == "failed" else None)


@app.get("/api/v1/jobs/{job_id}/progress", response_model=JobProgress, dependencies=[Depends(require_key)])
def get_progress(job_id: str) -> JobProgress:
    job = _get(job_id)
    return JobProgress(job_id=job.id, status=job.status, progress=job.progress, stage=job.stage)


@app.get("/api/v1/rules", dependencies=[Depends(require_key)])
def get_rules() -> List[Dict[str, str]]:
    """Rule catalogue (code, violation type, severity, legal source) for UIs."""
    return pipeline.list_rules()


@app.get("/health")
def health() -> JSONResponse:
    engine = pipeline.get_engine()
    return JSONResponse({"status": "ok", "warmup": warmup, "capabilities": engine.capabilities(),
                         "workers": WORKERS})


# --------------------------------------------------------------------------- #
# web UI (the FORTEYE html design, wired to the API above)
# --------------------------------------------------------------------------- #
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    page = STATIC_DIR / "forteye.html"
    if page.is_file():
        return FileResponse(page, media_type="text/html; charset=utf-8")
    return JSONResponse({"service": "FORTEYE", "docs": "/docs"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=os.getenv("FORTEYE_HOST", "0.0.0.0"), port=int(os.getenv("FORTEYE_PORT", "8000")))
