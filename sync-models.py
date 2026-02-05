#!/usr/bin/env python3
"""
Model Sync App (v1.1)

Run this on BOTH machines.
Features:
- Web UI (no auth) for home LAN
- Configure local root + remote URL + remote root
- Compare trees (local always renders even if remote unreachable)
- Drag & drop file copy local<->remote
- Prefer rsync over SSH for big model files/folders
- HTTP streaming fallback for file copy (not directories)
- Download a URL (HuggingFace or any direct URL) on local/remote/both
- Job status + progress
- Peer check loop (health ping) at configurable interval via --peer-poll-seconds
- Before any copy: verify SOURCE file is stable (size not changing)

Start:
  python model_sync_app.py --host 0.0.0.0 --port 9090 --peer-poll-seconds 10

Open:
  http://<machine-ip>:9090
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# =============================================================================
# Helpers
# =============================================================================

CONFIG_FILE = Path.home() / ".model_sync_config.json"
JOB_LOG_MAX = 400


def now_ts() -> float:
    return time.time()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_relpath(p: Path, root: Path) -> str:
    p = p.resolve()
    root = root.resolve()
    try:
        rel = p.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path escapes root.")
    return rel.as_posix()


def wait_for_local_file_stable(path: Path, checks: int = 3, interval_sec: float = 2.0) -> bool:
    """
    Returns True if file size stays constant across N checks.
    Useful if file might still be downloading.
    """
    if not path.exists() or not path.is_file():
        return False
    last = None
    for _ in range(checks):
        size = path.stat().st_size
        if last is None:
            last = size
        else:
            if size != last:
                last = size
        time.sleep(interval_sec)
    return path.stat().st_size == last


# =============================================================================
# Config
# =============================================================================

@dataclass
class AppConfig:
    local_root: str = ""
    remote_url: str = ""          # e.g. http://192.168.1.50:9090
    remote_root: str = ""         # remote models directory root (for rsync path building)
    # rsync/ssh config (recommended)
    ssh_host: str = ""            # e.g. 192.168.1.50
    ssh_port: int = 22
    ssh_user: str = ""
    ssh_key_path: str = ""        # optional, e.g. ~/.ssh/id_ed25519
    prefer_rsync: bool = True
    max_depth: int = 6
    hf_token: str = ""  # HuggingFace token (optional)


    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def load() -> "AppConfig":
        if CONFIG_FILE.exists():
            try:
                return AppConfig(**json.loads(CONFIG_FILE.read_text()))
            except Exception:
                return AppConfig()
        return AppConfig()

    def save(self) -> None:
        CONFIG_FILE.write_text(json.dumps(self.to_json(), indent=2))


CONFIG = AppConfig.load()

# =============================================================================
# Jobs
# =============================================================================

@dataclass
class Job:
    id: str
    kind: str
    created: float
    status: str            # queued, running, done, error
    message: str = ""
    progress: float = 0.0  # 0..1
    bytes_done: int = 0
    bytes_total: int = 0
    log: List[str] = None

    def __post_init__(self):
        if self.log is None:
            self.log = []

    def add_log(self, line: str) -> None:
        self.log.append(line)
        if len(self.log) > JOB_LOG_MAX:
            self.log = self.log[-JOB_LOG_MAX:]


JOBS: Dict[str, Job] = {}
JOB_QUEUE: "queue.Queue[Tuple[str, callable]]" = queue.Queue()


def create_job(kind: str, message: str = "") -> Job:
    jid = uuid.uuid4().hex
    job = Job(id=jid, kind=kind, created=now_ts(), status="queued", message=message)
    JOBS[jid] = job
    return job


def job_worker_loop() -> None:
    while True:
        jid, fn = JOB_QUEUE.get()
        job = JOBS.get(jid)
        if not job:
            JOB_QUEUE.task_done()
            continue
        job.status = "running"
        try:
            fn(job)
            if job.status != "error":
                job.status = "done"
                job.progress = 1.0
        except Exception as e:
            job.status = "error"
            job.message = str(e)
            job.add_log(f"ERROR: {e}")
        finally:
            JOB_QUEUE.task_done()


WORKER_THREAD = threading.Thread(target=job_worker_loop, daemon=True)
WORKER_THREAD.start()

# =============================================================================
# Transfer logic (rsync + HTTP fallback)
# =============================================================================

def can_use_rsync(cfg: AppConfig) -> bool:
    if not cfg.prefer_rsync:
        return False
    if not (cfg.ssh_host and cfg.ssh_user and cfg.remote_root):
        return False
    if cfg.ssh_key_path and not Path(cfg.ssh_key_path).expanduser().exists():
        return False
    return shutil.which("rsync") is not None


def build_ssh_command(cfg: AppConfig) -> str:
    parts = ["ssh", "-p", str(cfg.ssh_port)]
    if cfg.ssh_key_path:
        parts += ["-i", str(Path(cfg.ssh_key_path).expanduser())]
    parts += ["-o", "StrictHostKeyChecking=accept-new"]
    return " ".join(parts)


def run_rsync(job: Job, src: str, dst: str, ssh: str) -> None:
    cmd = [
        "rsync",
        "-a",
        "--partial",
        "--inplace",
        "--info=progress2",
        "--no-inc-recursive",
        "-e",
        ssh,
        src,
        dst,
    ]
    job.add_log(" ".join(cmd))
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    last_percent = 0
    for line in p.stdout or []:
        line = line.rstrip()
        if line:
            job.add_log(line)
        m = re.search(r"(\d+)%", line)
        if m:
            percent = int(m.group(1))
            if percent >= last_percent:
                last_percent = percent
                job.progress = percent / 100.0
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"rsync failed (exit {rc})")


def http_stream_copy_push(job: Job, local_file: Path, remote_url: str, remote_rel: str) -> None:
    """
    Push local file to remote via HTTP streaming.
    Remote endpoint: POST /api/upload_stream?relpath=...
    """
    url = remote_url.rstrip("/") + "/api/upload_stream"
    size = local_file.stat().st_size
    job.bytes_total = size
    job.add_log(f"HTTP PUSH => {url}?relpath={remote_rel}")

    with local_file.open("rb") as f:
        def gen():
            done = 0
            while True:
                chunk = f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                done += len(chunk)
                job.bytes_done = done
                job.progress = done / max(1, size)
                yield chunk

        r = requests.post(url, params={"relpath": remote_rel}, data=gen(), timeout=3600)
        if r.status_code != 200:
            raise RuntimeError(f"Remote upload failed: {r.status_code} {r.text}")


def http_stream_copy_pull(job: Job, remote_url: str, remote_rel: str, local_file: Path) -> None:
    """
    Pull remote file via HTTP streaming.
    Remote endpoint: GET /api/download_stream?relpath=...
    """
    url = remote_url.rstrip("/") + "/api/download_stream"
    job.add_log(f"HTTP PULL <= {url}?relpath={remote_rel}")
    ensure_parent(local_file)

    with requests.get(url, params={"relpath": remote_rel}, stream=True, timeout=3600) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Remote download failed: {r.status_code} {r.text}")
        total = int(r.headers.get("Content-Length", "0") or "0")
        job.bytes_total = total
        done = 0
        with local_file.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                job.bytes_done = done
                job.progress = done / max(1, total) if total else 0.0


# =============================================================================
# Tree building
# =============================================================================

def build_tree(root: Path, max_depth: int) -> Dict[str, Any]:
    root = root.resolve()
    if not root.exists():
        return {"name": root.name, "type": "missing", "children": []}

    def node_for(path: Path, depth: int) -> Dict[str, Any]:
        if path.is_dir():
            n = {"name": path.name, "type": "dir", "children": []}
            if depth >= max_depth:
                n["children"] = [{"name": "…", "type": "more"}]
                return n
            try:
                entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except Exception:
                entries = []
            for e in entries:
                if e.name in {".DS_Store", ".git", "__pycache__"}:
                    continue
                n["children"].append(node_for(e, depth + 1))
            return n
        else:
            try:
                st = path.stat()
                return {"name": path.name, "type": "file", "size": st.st_size, "mtime": int(st.st_mtime)}
            except Exception:
                return {"name": path.name, "type": "file", "size": 0, "mtime": 0}

    return {"name": root.name, "type": "dir", "children": [node_for(root, 0)]}


# =============================================================================
# FastAPI Models
# =============================================================================

class ConfigIn(BaseModel):
    local_root: str = ""
    remote_url: str = ""
    remote_root: str = ""
    ssh_host: str = ""
    ssh_port: int = 22
    ssh_user: str = ""
    ssh_key_path: str = ""
    prefer_rsync: bool = True
    max_depth: int = 6
    hf_token: str = "" 



class CopyIn(BaseModel):
    direction: str = Field(..., description="push or pull")
    relpath: str = Field(..., description="relative path under root")


class UrlDownloadIn(BaseModel):
    url: str
    side: str = Field("both", description="local|remote|both")
    dest_relpath: str = Field("", description="optional: relative output path under root")


# =============================================================================
# Peer status (background health polling)
# =============================================================================

@dataclass
class PeerStatus:
    last_check: float = 0.0
    ok: bool = False
    detail: str = "not checked"


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI()
app.state.peer_poll_seconds = 10
app.state.peer_status = PeerStatus()


def peer_poll_loop():
    """
    Periodically check if remote is alive.
    This does NOT sync automatically; it just validates connectivity and improves UI feedback.
    """
    while True:
        time.sleep(max(1, int(app.state.peer_poll_seconds)))
        ps: PeerStatus = app.state.peer_status
        ps.last_check = now_ts()

        if not CONFIG.remote_url:
            ps.ok = False
            ps.detail = "remote_url not set"
            continue

        try:
            url = CONFIG.remote_url.rstrip("/") + "/api/health"
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                ps.ok = True
                ps.detail = "ok"
            else:
                ps.ok = False
                ps.detail = f"health returned {r.status_code}"
        except Exception as e:
            ps.ok = False
            ps.detail = f"{e}"


@app.on_event("startup")
def _startup():
    t = threading.Thread(target=peer_poll_loop, daemon=True)
    t.start()


@app.get("/", response_class=HTMLResponse)
def ui_index() -> str:
    return HTML_PAGE


@app.get("/api/health")
def api_health() -> Dict[str, Any]:
    return {"ok": True, "time": int(now_ts())}


@app.get("/api/peer_status")
def api_peer_status() -> Dict[str, Any]:
    ps: PeerStatus = app.state.peer_status
    return asdict(ps)


@app.get("/api/config")
def api_get_config() -> Dict[str, Any]:
    return CONFIG.to_json()


@app.post("/api/config")
def api_set_config(cfg: ConfigIn) -> Dict[str, Any]:
    global CONFIG
    CONFIG = AppConfig(**cfg.dict())
    CONFIG.save()
    return {"ok": True, "config": CONFIG.to_json()}


@app.get("/api/tree/local")
def api_tree_local() -> Dict[str, Any]:
    if not CONFIG.local_root:
        return {"error": "Set local_root in config."}
    return build_tree(Path(CONFIG.local_root), CONFIG.max_depth)


@app.get("/api/tree/remote")
def api_tree_remote() -> Dict[str, Any]:
    """
    Proxy remote tree through this server to avoid CORS issues.
    Always returns JSON, with {error: "..."} on failure.
    """
    if not CONFIG.remote_url:
        return {"error": "Set remote_url in config."}
    url = CONFIG.remote_url.rstrip("/") + "/api/tree/local"
    try:
        r = requests.get(url, timeout=8)
        return r.json()
    except Exception as e:
        return {"error": f"Remote unreachable: {e}"}


@app.get("/api/jobs")
def api_jobs() -> Dict[str, Any]:
    items = sorted(JOBS.values(), key=lambda j: j.created, reverse=True)
    return {"jobs": [asdict(j) for j in items[:50]]}


@app.get("/api/stat")
def api_stat(relpath: str) -> Dict[str, Any]:
    """
    Returns file/dir stat on this node (used for remote stability checks).
    """
    if not CONFIG.local_root:
        raise HTTPException(status_code=400, detail="Set local_root in config")

    root = Path(CONFIG.local_root).resolve()
    rel = Path(relpath)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail="Invalid relpath")

    p = (root / rel).resolve()
    _ = safe_relpath(p, root)

    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found")

    st = p.stat()
    return {
        "exists": True,
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
    }


@app.get("/api/is_stable")
def api_is_stable(relpath: str, checks: int = 3, interval_sec: float = 2.0) -> Dict[str, Any]:
    """
    Remote-callable stability check for THIS node.
    Only meaningful for files.
    """
    if not CONFIG.local_root:
        raise HTTPException(status_code=400, detail="Set local_root in config")

    root = Path(CONFIG.local_root).resolve()
    rel = Path(relpath)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail="Invalid relpath")

    p = (root / rel).resolve()
    _ = safe_relpath(p, root)

    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found")
    if not p.is_file():
        return {"stable": True, "note": "not a file (dir assumed stable check skipped)"}

    stable = wait_for_local_file_stable(p, checks=checks, interval_sec=interval_sec)
    return {"stable": stable, "size": p.stat().st_size}


@app.post("/api/copy")
def api_copy(req: CopyIn) -> Dict[str, Any]:
    if req.direction not in {"push", "pull"}:
        raise HTTPException(status_code=400, detail="direction must be push or pull")
    if not CONFIG.local_root or not CONFIG.remote_url or not CONFIG.remote_root:
        raise HTTPException(status_code=400, detail="Configure local_root, remote_url, remote_root first.")

    local_root = Path(CONFIG.local_root).resolve()
    remote_root = Path(CONFIG.remote_root)

    rel = Path(req.relpath)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail="Invalid relpath")

    local_path = (local_root / rel).resolve()

    job = create_job(kind=f"copy:{req.direction}", message=req.relpath)

    def ensure_remote_file_stable(job_obj: Job) -> None:
        """
        Ask remote node if its source file is stable before we pull from it.
        """
        url = CONFIG.remote_url.rstrip("/") + "/api/is_stable"
        job_obj.add_log(f"Checking remote stability: {url}?relpath={rel.as_posix()}")
        r = requests.get(url, params={"relpath": rel.as_posix(), "checks": 3, "interval_sec": 2.0}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Remote stability check failed: {r.status_code} {r.text}")
        data = r.json()
        if not data.get("stable", False):
            raise RuntimeError("Remote source file appears to be changing size (still downloading?). Try again later.")

    def do_copy(job_obj: Job) -> None:
        # PUSH (local -> remote)
        if req.direction == "push":
            if not local_path.exists():
                raise RuntimeError("Local path not found")

            # If pushing a file, ensure it's stable first
            if local_path.is_file():
                job_obj.add_log("Checking local file stability before transfer...")
                if not wait_for_local_file_stable(local_path, checks=3, interval_sec=2.0):
                    raise RuntimeError("Local source file appears to be changing size (still downloading?). Try again later.")

            if can_use_rsync(CONFIG):
                ssh = build_ssh_command(CONFIG)
                src = str(local_path) + ("/" if local_path.is_dir() else "")
                dst = f"{CONFIG.ssh_user}@{CONFIG.ssh_host}:{str(remote_root / rel)}"
                run_rsync(job_obj, src, dst, ssh)
                return

            # HTTP push fallback: files only
            if local_path.is_dir():
                raise RuntimeError("HTTP push of directories not supported. Configure rsync for folders.")
            http_stream_copy_push(job_obj, local_path, CONFIG.remote_url, rel.as_posix())
            return

        # PULL (remote -> local)
        # If pulling a file, ensure remote says it is stable first
        ensure_remote_file_stable(job_obj)

        if can_use_rsync(CONFIG):
            ssh = build_ssh_command(CONFIG)
            src = f"{CONFIG.ssh_user}@{CONFIG.ssh_host}:{str(remote_root / rel)}"
            dst = str(local_path)
            run_rsync(job_obj, src, dst, ssh)
            return

        # HTTP pull fallback: files only
        http_stream_copy_pull(job_obj, CONFIG.remote_url, rel.as_posix(), local_path)

        # After pull, optionally check local stability (cheap sanity)
        if local_path.exists() and local_path.is_file():
            job_obj.add_log("Post-transfer local stability check...")
            _ = wait_for_local_file_stable(local_path, checks=2, interval_sec=1.0)

    JOB_QUEUE.put((job.id, do_copy))
    return {"ok": True, "job_id": job.id}


@app.get("/api/download_stream")
def api_download_stream(relpath: str):
    if not CONFIG.local_root:
        raise HTTPException(status_code=400, detail="Set local_root in config")
    root = Path(CONFIG.local_root).resolve()
    rel = Path(relpath)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail="Invalid relpath")
    p = (root / rel).resolve()
    _ = safe_relpath(p, root)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Not found or not a file")

    def gen():
        with p.open("rb") as f:
            while True:
                b = f.read(8 * 1024 * 1024)
                if not b:
                    break
                yield b

    return StreamingResponse(
        gen(),
        media_type="application/octet-stream",
        headers={"Content-Length": str(p.stat().st_size)}
    )


@app.post("/api/upload_stream")
async def api_upload_stream(relpath: str, request: Request):
    if not CONFIG.local_root:
        raise HTTPException(status_code=400, detail="Set local_root in config")

    root = Path(CONFIG.local_root).resolve()
    rel = Path(relpath)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail="Invalid relpath")

    dest = (root / rel).resolve()
    _ = safe_relpath(dest, root)
    ensure_parent(dest)

    with dest.open("wb") as f:
        async for chunk in request.stream():
            f.write(chunk)

    return {"ok": True}


@app.post("/api/download_url")
def api_download_url(req: UrlDownloadIn) -> Dict[str, Any]:
    """
    Downloads a URL onto THIS node under local_root.
    """
    if not CONFIG.local_root:
        raise HTTPException(status_code=400, detail="Set local_root in config")

    job = create_job(kind="download_url", message=req.url)

    def do_dl(job_obj: Job) -> None:
        root = Path(CONFIG.local_root).resolve()

        if req.dest_relpath:
            rel = Path(req.dest_relpath)
            if rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError("Invalid dest_relpath")
            dest = (root / rel).resolve()
        else:
            name = req.url.rstrip("/").split("/")[-1].split("?")[0] or f"download_{int(time.time())}"
            dest = (root / name).resolve()

        _ = safe_relpath(dest, root)
        ensure_parent(dest)

        job_obj.add_log(f"Downloading => {dest}")
        #with requests.get(req.url, stream=True, timeout=3600) as r:
        headers = {}
        if CONFIG.hf_token:
            headers["Authorization"] = f"Bearer {CONFIG.hf_token}"

        with requests.get(req.url, headers=headers, stream=True, timeout=3600) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", "0") or "0")
            job_obj.bytes_total = total
            done = 0
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    job_obj.bytes_done = done
                    job_obj.progress = done / max(1, total) if total else 0.0

        # Ensure file is stable after download completes
        job_obj.add_log("Verifying downloaded file is stable...")
        if not wait_for_local_file_stable(dest, checks=3, interval_sec=1.0):
            raise RuntimeError("Downloaded file still changing size after download finished (unexpected).")

    JOB_QUEUE.put((job.id, do_dl))
    return {"ok": True, "job_id": job.id}


@app.post("/api/hf_download")
def api_hf_download(req: UrlDownloadIn) -> Dict[str, Any]:
    """
    Downloads a URL on local/remote/both.

    How remote download is done:
    - This node sends HTTP POST to remote node: /api/download_url
      with JSON: {url, side, dest_relpath}
    - Remote node queues a local download job and returns a job_id.
    """
    if req.side not in {"local", "remote", "both"}:
        raise HTTPException(status_code=400, detail="side must be local|remote|both")

    results = {"local": None, "remote": None}

    if req.side in {"local", "both"}:
        results["local"] = api_download_url(req)

    if req.side in {"remote", "both"}:
        if not CONFIG.remote_url:
            raise HTTPException(status_code=400, detail="Set remote_url in config")
        url = CONFIG.remote_url.rstrip("/") + "/api/download_url"
        r = requests.post(url, json=req.dict(), timeout=30)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Remote download_url failed: {r.text}")
        results["remote"] = r.json()

    return {"ok": True, "results": results}


# =============================================================================
# Web UI (single-page)
# =============================================================================

HTML_PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Model Sync (v1.1)</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 0; padding: 0; }
    header { padding: 12px 16px; border-bottom: 1px solid #ddd; display:flex; gap:16px; align-items:center; }
    h1 { font-size: 16px; margin:0; }
    main { padding: 12px 16px; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
    .row { display:flex; gap: 10px; flex-wrap: wrap; align-items:center; }
    label { font-size: 12px; color:#444; display:block; }
    input { padding: 6px 8px; border: 1px solid #ccc; border-radius: 6px; min-width: 260px; }
    button { padding: 7px 10px; border-radius: 8px; border: 1px solid #777; background: #fff; cursor: pointer; }
    button:hover { background:#f5f5f5; }
    .tree { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; max-height: 520px; overflow:auto; border:1px solid #eee; border-radius: 8px; padding: 8px; }
    ul { list-style: none; padding-left: 16px; margin: 4px 0; }
    li { margin: 2px 0; }
    .item { padding: 2px 4px; border-radius: 6px; display:inline-block; }
    .item.file { cursor: grab; }
    .item.dir { font-weight: 600; }
    .dropzone { border: 2px dashed #bbb; border-radius: 10px; padding: 10px; margin-top: 8px; }
    .dropzone.dragover { border-color: #444; background: #fafafa; }
    .muted { color:#666; font-size: 12px; }
    .jobs { max-height: 260px; overflow:auto; border:1px solid #eee; border-radius: 8px; padding: 8px; font-size: 12px; }
    .job { border-bottom: 1px dashed #ddd; padding: 6px 0; }
    .bar { height: 8px; border-radius: 10px; background: #eee; overflow:hidden; }
    .bar > div { height: 100%; background: #999; width: 0%; }
    .pill { display:inline-block; padding: 1px 6px; border:1px solid #ccc; border-radius: 999px; font-size: 11px; color:#444; }
  </style>
</head>
<body>
<header>
  <h1>Model Sync (v1.1)</h1>
  <span class="muted">Local tree always renders. Remote errors won’t break the page.</span>
</header>

<main>
  <div class="card">
    <div class="row">
      <div>
        <label>Local root</label>
        <input id="local_root" placeholder="/opt/ai/models"/>
      </div>
      <div>
        <label>Remote URL</label>
        <input id="remote_url" placeholder="http://192.168.1.50:9090"/>
      </div>
      <div>
        <label>Remote root</label>
        <input id="remote_root" placeholder="/opt/ai/models"/>
      </div>
      <div style="min-width:280px">
        <label>Prefer rsync</label>
        <input id="prefer_rsync" type="checkbox" style="min-width:auto"/>
      </div>
    </div>

    <div class="row" style="margin-top:10px">
      <div>
        <label>SSH host</label>
        <input id="ssh_host" placeholder="192.168.1.50"/>
      </div>
      <div>
        <label>SSH port</label>
        <input id="ssh_port" placeholder="22"/>
      </div>
      <div>
        <label>SSH user</label>
        <input id="ssh_user" placeholder="dave"/>
      </div>
      <div>
        <label>SSH key path (optional)</label>
        <input id="ssh_key_path" placeholder="~/.ssh/id_ed25519"/>
      </div>
      <div>
        <label>Tree max depth</label>
        <input id="max_depth" placeholder="6"/>
      </div>
      <div style="display:flex; align-items:flex-end; gap:8px">
        <button onclick="saveConfig()">Save config</button>
        <button onclick="refreshTrees()">Refresh trees</button>
      </div>
    </div>

    <div class="row" style="margin-top:12px">
      <div>
        <label>Download URL (HuggingFace or any direct URL)</label>
        <input id="dl_url" placeholder="https://huggingface.co/.../resolve/main/model.safetensors?download=true" style="min-width:740px"/>
      </div>
      <div>
        <label>Download destination (optional relpath under root)</label>
        <input id="dl_dest" placeholder="flux/model.safetensors"/>
      </div>
      <div>
        <label>Where</label>
        <select id="dl_side" style="padding:6px 8px; border-radius:8px; border:1px solid #ccc;">
          <option value="both">both</option>
          <option value="local">local</option>
          <option value="remote">remote</option>
        </select>
      </div>

      <div style="display:flex; align-items:flex-end; margin-top:12px; gap:12px">
        <button onclick="downloadUrl()">Download</button>
      </div>

      <div>
        <label>HuggingFace Token (optional)</label>
        <input id="hf_token" placeholder="hf_..." style="min-width:420px"/>
      </div>
    
    </div>

    <div class="row" style="margin-top:10px; justify-content:space-between">
      <div id="status" class="muted"></div>
      <div id="peer" class="muted"></div>
    </div>
  </div>

  <div class="grid" style="margin-top:16px">
    <div class="card">
      <div class="row" style="justify-content:space-between">
        <strong>Local</strong>
        <span class="pill">drag files ➜ Remote dropzone</span>
      </div>
      <div id="tree_local" class="tree"></div>
      <div id="drop_remote" class="dropzone">
        Drop here to COPY TO REMOTE (push)
        <div class="muted">Directories require rsync. Files can use rsync or HTTP.</div>
      </div>
    </div>

    <div class="card">
      <div class="row" style="justify-content:space-between">
        <strong>Remote</strong>
        <span class="pill">drag files ➜ Local dropzone</span>
      </div>
      <div id="tree_remote" class="tree"></div>
      <div id="drop_local" class="dropzone">
        Drop here to COPY TO LOCAL (pull)
        <div class="muted">Directories require rsync. Files can use rsync or HTTP.</div>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <div class="row" style="justify-content:space-between">
      <strong>Jobs</strong>
      <span class="muted">auto-refreshes every 1s</span>
    </div>
    <div id="jobs" class="jobs"></div>
  </div>
</main>

<script>
let dragged = null;
function el(id){ return document.getElementById(id); }
function setStatus(msg){ el("status").innerText = msg || ""; }

async function apiGet(path){
  const r = await fetch(path);
  return await r.json();
}
async function apiPost(path, body){
  const r = await fetch(path, {method:"POST", headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const txt = await r.text();
  try { return {status:r.status, json: JSON.parse(txt)}; }
  catch { return {status:r.status, json: {raw: txt}}; }
}

function renderTree(container, tree){
  container.innerHTML = "";
  if (tree.error){
    container.innerHTML = `<div class="muted">${tree.error}</div>`;
    return;
  }
  const rootChild = (tree.children && tree.children[0]) ? tree.children[0] : null;
  if (!rootChild){
    container.innerHTML = `<div class="muted">No data</div>`;
    return;
  }

  function makeUL(node, relBase){
    const ul = document.createElement("ul");
    if (!node.children) return ul;
    node.children.forEach(ch => {
      const li = document.createElement("li");
      const span = document.createElement("span");
      span.className = "item " + (ch.type || "");
      span.textContent = ch.name + (ch.type === "dir" ? "/" : "");
      const rel = (relBase ? (relBase + "/" + ch.name) : ch.name);
      span.dataset.rel = rel;

      if (ch.type === "file"){
        span.draggable = true;
        span.addEventListener("dragstart", (e) => {
          dragged = rel;
          e.dataTransfer.setData("text/plain", rel);
        });
      }

      li.appendChild(span);

      if (ch.type === "file"){
        const meta = document.createElement("span");
        meta.className = "muted";
        meta.style.marginLeft = "6px";
        if (ch.size !== undefined){
          meta.textContent = `(${ch.size} bytes)`;
        }
        li.appendChild(meta);
      }

      if (ch.children && ch.type === "dir"){
        li.appendChild(makeUL(ch, rel));
      }
      ul.appendChild(li);
    });
    return ul;
  }

  container.appendChild(makeUL(rootChild, ""));
}

async function loadConfig(){
  const cfg = await apiGet("/api/config");
  el("local_root").value = cfg.local_root || "";
  el("remote_url").value = cfg.remote_url || "";
  el("remote_root").value = cfg.remote_root || "";
  el("ssh_host").value = cfg.ssh_host || "";
  el("ssh_port").value = cfg.ssh_port || 22;
  el("ssh_user").value = cfg.ssh_user || "";
  el("ssh_key_path").value = cfg.ssh_key_path || "";
  el("prefer_rsync").checked = !!cfg.prefer_rsync;
  el("max_depth").value = cfg.max_depth || 6;
  el("hf_token").value = cfg.hf_token || "";
}

async function saveConfig(){
  const body = {
    local_root: el("local_root").value.trim(),
    remote_url: el("remote_url").value.trim(),
    remote_root: el("remote_root").value.trim(),
    ssh_host: el("ssh_host").value.trim(),
    ssh_port: parseInt(el("ssh_port").value.trim() || "22"),
    ssh_user: el("ssh_user").value.trim(),
    ssh_key_path: el("ssh_key_path").value.trim(),
    prefer_rsync: el("prefer_rsync").checked,
    max_depth: parseInt(el("max_depth").value.trim() || "6"),
    hf_token: el("hf_token").value.trim(),
  };
  const res = await apiPost("/api/config", body);
  if (res.status !== 200){
    setStatus("Save failed: " + JSON.stringify(res.json));
    return;
  }
  setStatus("Saved.");
  await refreshTrees();
}

async function refreshTrees(){
  setStatus("Refreshing trees...");

  // Always render local first; remote failure must not break local.
  try {
    const local = await apiGet("/api/tree/local");
    renderTree(el("tree_local"), local);
  } catch (e) {
    el("tree_local").innerHTML = `<div class="muted">Local tree error: ${e}</div>`;
  }

  try {
    const remote = await apiGet("/api/tree/remote");
    renderTree(el("tree_remote"), remote);
  } catch (e) {
    el("tree_remote").innerHTML = `<div class="muted">Remote tree unavailable: ${e}</div>`;
  }

  setStatus("Last refresh: " + new Date().toLocaleTimeString());
  setTimeout(()=>setStatus(""), 1500);
}

async function copy(direction, relpath){
  const res = await apiPost("/api/copy", {direction, relpath});
  if (res.status !== 200){
    alert("Copy failed: " + JSON.stringify(res.json));
  } else {
    setStatus("Started job: " + res.json.job_id);
    setTimeout(()=>setStatus(""), 1500);
  }
}

function setupDropzone(zoneEl, direction){
  zoneEl.addEventListener("dragover", (e) => {
    e.preventDefault();
    zoneEl.classList.add("dragover");
  });
  zoneEl.addEventListener("dragleave", () => zoneEl.classList.remove("dragover"));
  zoneEl.addEventListener("drop", (e) => {
    e.preventDefault();
    zoneEl.classList.remove("dragover");
    const rel = e.dataTransfer.getData("text/plain") || dragged;
    if (!rel){
      alert("Nothing dragged.");
      return;
    }
    copy(direction, rel);
  });
}

async function downloadUrl(){
  const url = el("dl_url").value.trim();
  if (!url){ alert("Enter a URL"); return; }
  const side = el("dl_side").value;
  const dest = el("dl_dest").value.trim();
  const res = await apiPost("/api/hf_download", {url, side, dest_relpath: dest});
  if (res.status !== 200){
    alert("Download failed: " + JSON.stringify(res.json));
  } else {
    setStatus("Download requested.");
    setTimeout(()=>setStatus(""), 1500);
  }
}

async function refreshJobs(){
  const data = await apiGet("/api/jobs");
  const jobs = data.jobs || [];
  const div = el("jobs");
  div.innerHTML = "";
  jobs.forEach(j => {
    const pct = Math.round((j.progress || 0) * 100);
    const bytes = (j.bytes_total ? `${j.bytes_done}/${j.bytes_total}` : (j.bytes_done ? `${j.bytes_done}` : ""));
    const row = document.createElement("div");
    row.className = "job";
    row.innerHTML = `
      <div class="row" style="justify-content:space-between">
        <div><strong>${j.kind}</strong> <span class="muted">${j.message || ""}</span></div>
        <div><span class="pill">${j.status}</span> <span class="muted">${pct}% ${bytes}</span></div>
      </div>
      <div class="bar"><div style="width:${pct}%"></div></div>
      <details style="margin-top:6px">
        <summary class="muted">log</summary>
        <pre style="white-space:pre-wrap; font-size:11px; margin:6px 0 0 0">${(j.log || []).slice(-30).join("\n")}</pre>
      </details>
    `;
    div.appendChild(row);
  });
}

async function refreshPeer(){
  const ps = await apiGet("/api/peer_status");
  const t = ps.last_check ? new Date(ps.last_check * 1000).toLocaleTimeString() : "never";
  el("peer").innerText = `Peer check: ${ps.ok ? "OK" : "DOWN"} (${ps.detail}) @ ${t}`;
}

(async function init(){
  await loadConfig();
  await refreshTrees();
  setupDropzone(el("drop_remote"), "push");
  setupDropzone(el("drop_local"), "pull");
  setInterval(refreshJobs, 1000);
  setInterval(refreshPeer, 2000);
})();
</script>
</body>
</html>
"""

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--peer-poll-seconds", type=int, default=10,
                        help="How often this instance pings the other instance /api/health")
    args = parser.parse_args()

    app.state.peer_poll_seconds = max(1, int(args.peer_poll_seconds))

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
