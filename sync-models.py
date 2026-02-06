#!/usr/bin/env python3
"""
Model Sync App (v1.3)

Run this on BOTH machines.
Features:
- Web UI (no auth) for home LAN
- Configure local root + remote URL + remote root
- Compare trees (local always renders even if remote unreachable)
- Drag & drop file copy local<->remote
- Prefer rsync over SSH for big model files/folders
- HTTP streaming fallback for file copy (not directories)
- Download a URL (HuggingFace or any direct URL) on local/remote/both
- Job status + progress + logs
- Peer check loop (health ping) at configurable interval via --peer-poll-seconds
- Sync buttons (<=500 files): Sync Local->Remote and Sync Remote->Local
  - Sync compares SIZE first
  - If sizes match, it only uses MTIME when "mtime strict" toggle is enabled
  - MTIME uses a tolerance (default 2s) to avoid filesystem timestamp quirks

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
from typing import Dict, Any, List, Tuple, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

# =============================================================================
# Helpers
# =============================================================================

CONFIG_FILE = Path.home() / ".model_sync_config.json"
JOB_LOG_MAX = 400
SYNC_MAX_FILES = 500  # as requested


def now_ts() -> float:
    return time.time()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def mtime_close(a: int, b: int, tol_sec: int = 2) -> bool:
    return abs(int(a) - int(b)) <= int(tol_sec)


def parse_rsync_version(text: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse output of `rsync --version` and return (major, minor, patch) if possible.
    Example first line: "rsync  version 3.2.7  protocol version 31"
    """
    m = re.search(r"rsync\s+version\s+(\d+)\.(\d+)\.(\d+)", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def get_rsync_version() -> Optional[Tuple[int, int, int]]:
    rsync_path = shutil.which("rsync")
    if not rsync_path:
        return None
    try:
        r = subprocess.run([rsync_path, "--version"], capture_output=True, text=True, check=False)
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        return parse_rsync_version(out)
    except Exception:
        return None


def supports_info_progress2(ver: Optional[Tuple[int, int, int]]) -> bool:
    # progress2 came in rsync 3.1.0
    if not ver:
        return False
    return ver >= (3, 1, 0)


def human_shell_quote(s: str) -> str:
    """
    Minimal quoting for logs / copy-paste convenience.
    """
    if re.search(r"[^\w@%+=:,./-]", s):
        return "'" + s.replace("'", "'\"'\"'") + "'"
    return s


def parse_hf_resolve_url(url: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse Hugging Face resolve URL into (repo_id, revision, file_path).

    Example:
      https://huggingface.co/owner/repo/resolve/main/path/to/file.safetensors?download=true
    -> repo_id="owner/repo", revision="main", file_path="path/to/file.safetensors"

    Returns None if not a recognizable HF resolve URL.
    """
    base = url.split("?", 1)[0].rstrip("/")
    m = re.match(r"^https?://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.*)$", base)
    if not m:
        return None
    repo_id, revision, file_path = m.group(1), m.group(2), m.group(3)
    if not repo_id or not revision or not file_path:
        return None
    return repo_id, revision, file_path


def build_hf_cli_command(url: str, dest: Path, token: str = "") -> Optional[str]:
    """
    Build a copy/paste-able modern `hf download` command for a Hugging Face resolve URL.

    - Uses modern `hf` CLI
    - Auth via HF_TOKEN env var (if provided)
    - Ensures --local-dir is a directory (not a filename)
    """
    parsed = parse_hf_resolve_url(url)
    if not parsed:
        return None

    repo_id, revision, file_path = parsed
    local_dir = dest.parent

    env_prefix = ""
    if token:
        env_prefix = f"HF_TOKEN={human_shell_quote(token)} "

    rev_part = ""
    if revision and revision not in {"main", "master"}:
        rev_part = f"--revision {human_shell_quote(revision)} "

    cmd = (
        f"{env_prefix}"
        "hf download "
        f"{human_shell_quote(repo_id)} "
        f"{human_shell_quote(file_path)} "
        f"{rev_part}"
        f"--local-dir {human_shell_quote(str(local_dir))}"
    )
    return cmd


def list_files_flat(root: Path, max_files: int = SYNC_MAX_FILES) -> Dict[str, Any]:
    """
    Returns a flat listing suitable for sync planning.
    Includes:
      - files: { relpath: {size, mtime} }
      - dirs:  [ relpath_dir, ... ]  (excluding root itself)
      - truncated: bool
    """
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        return {"files": {}, "dirs": [], "truncated": False}

    files: Dict[str, Dict[str, int]] = {}
    dirs: List[str] = []
    truncated = False

    # Walk directories; stop once we hit max_files
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)

        # record directory rel (skip root itself)
        if d != root:
            try:
                rel_dir = safe_relpath(d, root)
                dirs.append(rel_dir)
            except Exception:
                continue

        # ignore noise dirs
        dirnames[:] = [x for x in dirnames if x not in {".git", "__pycache__"}]

        for fn in filenames:
            if fn in {".DS_Store"}:
                continue
            p = d / fn
            try:
                st = p.stat()
                rel = safe_relpath(p, root)
            except Exception:
                continue
            files[rel] = {"size": int(st.st_size), "mtime": int(st.st_mtime)}

            if len(files) >= max_files:
                truncated = True
                break

        if truncated:
            break

    return {"files": files, "dirs": sorted(set(dirs)), "truncated": truncated}


# =============================================================================
# Config
# =============================================================================

@dataclass
class AppConfig:
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
    status: str
    message: str = ""
    progress: float = 0.0
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

def rsync_diag(cfg: AppConfig) -> Tuple[bool, str]:
    reasons = []
    if not cfg.prefer_rsync:
        reasons.append("prefer_rsync is False")
    if not cfg.ssh_host:
        reasons.append("ssh_host not set")
    if not cfg.ssh_user:
        reasons.append("ssh_user not set")
    if not cfg.remote_root:
        reasons.append("remote_root not set")

    if cfg.ssh_key_path:
        keyp = Path(cfg.ssh_key_path).expanduser()
        if not keyp.exists():
            reasons.append(f"ssh_key_path does not exist: {keyp}")

    rsync_path = shutil.which("rsync")
    if rsync_path is None:
        reasons.append("rsync not found in PATH")

    if reasons:
        return False, "; ".join(reasons)
    return True, f"OK (rsync={rsync_path})"


def can_use_rsync(cfg: AppConfig) -> bool:
    ok, _ = rsync_diag(cfg)
    return ok


def build_ssh_command(cfg: AppConfig) -> str:
    parts = ["ssh", "-p", str(cfg.ssh_port)]
    if cfg.ssh_key_path:
        parts += ["-i", str(Path(cfg.ssh_key_path).expanduser())]
    parts += ["-o", "StrictHostKeyChecking=accept-new"]
    return " ".join(parts)


def run_rsync(job: Job, src: str, dst: str, ssh: str) -> None:
    """
    Runs rsync and attempts to track progress.

    macOS often ships rsync 2.6.9 which DOES NOT support: --info=progress2
    So we pick flags based on detected rsync version.
    """
    rsync_path = shutil.which("rsync") or "rsync"
    ver = get_rsync_version()
    job.add_log(f"Detected rsync version: {ver if ver else 'unknown'} (path={rsync_path})")

    progress_args: List[str]
    if supports_info_progress2(ver):
        progress_args = ["--info=progress2"]
        job.add_log("rsync progress mode: --info=progress2")
    else:
        progress_args = ["--progress"]
        job.add_log("rsync progress mode: --progress (fallback; --info=progress2 not supported)")

    cmd = [
        rsync_path,
        "-a",
        "--partial",
        "--inplace",
        *progress_args,
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
    percent_re = re.compile(r"(\d+)%")

    for line in p.stdout or []:
        line = line.rstrip()
        if line:
            job.add_log(line)

        m = percent_re.search(line)
        if m:
            try:
                percent = int(m.group(1))
                if percent >= last_percent:
                    last_percent = percent
                    job.progress = percent / 100.0
            except Exception:
                pass

    rc = p.wait()
    job.add_log(f"rsync exit code: {rc}")
    if rc != 0:
        raise RuntimeError(f"rsync failed (exit {rc})")


def http_stream_copy_push(job: Job, local_file: Path, remote_url: str, remote_rel: str) -> None:
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
# Tree building (UI)
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
    dest_relpath: str = Field("", description="optional: absolute OR relative-to-local-root download directory")


class SyncIn(BaseModel):
    direction: str = Field(..., description="push or pull")
    mtime_strict: bool = Field(False, description="Only consider mtime when sizes match AND strict mode is enabled")
    mtime_tolerance_sec: int = Field(2, description="mtime tolerance in seconds (only used when strict enabled)")
    max_files: int = Field(SYNC_MAX_FILES, description="safety cap on number of files to consider")
    dry_run: bool = Field(False, description="plan only; do not enqueue copies")


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
    return asdict(app.state.peer_status)


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


@app.get("/api/is_stable")
def api_is_stable(relpath: str, checks: int = 3, interval_sec: float = 2.0) -> Dict[str, Any]:
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
    return {"stable": stable, "size": int(p.stat().st_size)}


@app.post("/api/mkdirs")
def api_mkdirs(dir_relpath: str) -> Dict[str, Any]:
    """
    Create a directory (and parents) under local_root.
    """
    if not CONFIG.local_root:
        raise HTTPException(status_code=400, detail="Set local_root in config")

    root = Path(CONFIG.local_root).resolve()
    rel = Path(dir_relpath)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail="Invalid dir_relpath")

    d = (root / rel).resolve()
    _ = safe_relpath(d, root)
    ensure_dir(d)
    return {"ok": True, "created": safe_relpath(d, root)}


@app.get("/api/files_flat")
def api_files_flat(max_files: int = SYNC_MAX_FILES) -> Dict[str, Any]:
    """
    Flat listing of up to max_files files for sync planning.
    """
    if not CONFIG.local_root:
        raise HTTPException(status_code=400, detail="Set local_root in config")
    mf = min(int(max_files), SYNC_MAX_FILES)
    return list_files_flat(Path(CONFIG.local_root), max_files=mf)


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
        url = CONFIG.remote_url.rstrip("/") + "/api/is_stable"
        job_obj.add_log(f"Checking remote stability: {url}?relpath={rel.as_posix()}")
        r = requests.get(url, params={"relpath": rel.as_posix(), "checks": 3, "interval_sec": 2.0}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Remote stability check failed: {r.status_code} {r.text}")
        data = r.json()
        if not data.get("stable", False):
            raise RuntimeError("Remote source file appears to be changing size (still downloading?). Try again later.")

    def ensure_target_parent(job_obj: Job) -> None:
        parent_rel = rel.parent.as_posix()
        if parent_rel in {"", "."}:
            return

        if req.direction == "push":
            url = CONFIG.remote_url.rstrip("/") + "/api/mkdirs"
            job_obj.add_log(f"Ensuring remote directory exists: {parent_rel}")
            r = requests.post(url, params={"dir_relpath": parent_rel}, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"Remote mkdirs failed: {r.status_code} {r.text}")
        else:
            # pull → ensure local dir exists
            ensure_dir((local_root / rel.parent).resolve())

    def do_copy(job_obj: Job) -> None:
        ok, why = rsync_diag(CONFIG)
        job_obj.add_log(f"rsync eligibility: {why}")

        if req.direction == "push":
            if not local_path.exists():
                raise RuntimeError("Local path not found")

            ensure_target_parent(job_obj)

            if local_path.is_file():
                job_obj.add_log("Checking local file stability before transfer...")
                if not wait_for_local_file_stable(local_path, checks=3, interval_sec=2.0):
                    raise RuntimeError("Local source file appears to be changing size (still downloading?). Try again later.")

            if can_use_rsync(CONFIG):
                job_obj.add_log("Copy strategy: rsync over SSH")
                ssh = build_ssh_command(CONFIG)
                src = str(local_path) + ("/" if local_path.is_dir() else "")
                dst = f"{CONFIG.ssh_user}@{CONFIG.ssh_host}:{str(remote_root / rel)}"
                run_rsync(job_obj, src, dst, ssh)
                return

            job_obj.add_log("Copy strategy: HTTP streaming fallback (rsync not eligible)")
            if local_path.is_dir():
                raise RuntimeError("HTTP push of directories not supported. Configure rsync for folders.")
            http_stream_copy_push(job_obj, local_path, CONFIG.remote_url, rel.as_posix())
            return

        # pull
        ensure_remote_file_stable(job_obj)
        ensure_target_parent(job_obj)

        if can_use_rsync(CONFIG):
            job_obj.add_log("Copy strategy: rsync over SSH")
            ssh = build_ssh_command(CONFIG)
            src = f"{CONFIG.ssh_user}@{CONFIG.ssh_host}:{str(remote_root / rel)}"
            dst = str(local_path)
            run_rsync(job_obj, src, dst, ssh)
            return

        job_obj.add_log("Copy strategy: HTTP streaming fallback (rsync not eligible)")
        http_stream_copy_pull(job_obj, CONFIG.remote_url, rel.as_posix(), local_path)

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
    if not CONFIG.local_root:
        raise HTTPException(status_code=400, detail="Set local_root in config")

    job = create_job(kind="download_url", message=req.url)

    def do_dl(job_obj: Job) -> None:
        root = Path(CONFIG.local_root).resolve()

        parsed = parse_hf_resolve_url(req.url)

        # Determine filename
        if parsed:
            _, _, file_path = parsed
            filename = Path(file_path).name
        else:
            filename = req.url.rstrip("/").split("/")[-1].split("?")[0]
            if not filename:
                filename = f"download_{int(time.time())}"

        # Determine destination directory
        if req.dest_relpath:
            dest_dir = Path(req.dest_relpath).expanduser()
            if dest_dir.is_absolute():
                dest_dir = dest_dir.resolve()
            else:
                dest_dir = (root / dest_dir).resolve()
        else:
            dest_dir = root

        dest_dir = dest_dir.resolve()
        if not str(dest_dir).startswith(str(root)):
            raise RuntimeError("Destination directory escapes local_root")

        dest = dest_dir / filename
        ensure_parent(dest)

        _ = safe_relpath(dest, root)

        job_obj.add_log(f"Downloading => {dest}")

        hf_cmd = build_hf_cli_command(req.url, dest, token=CONFIG.hf_token)
        if hf_cmd:
            job_obj.add_log("HuggingFace CLI equivalent command:")
            job_obj.add_log(hf_cmd)
        else:
            job_obj.add_log("Note: URL does not look like a HuggingFace /resolve/ URL; no hf CLI command derived.")

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

        job_obj.add_log("Verifying downloaded file is stable...")
        if not wait_for_local_file_stable(dest, checks=3, interval_sec=1.0):
            raise RuntimeError("Downloaded file still changing size after download finished (unexpected).")

    JOB_QUEUE.put((job.id, do_dl))
    return {"ok": True, "job_id": job.id}


@app.post("/api/hf_download")
def api_hf_download(req: UrlDownloadIn) -> Dict[str, Any]:
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


def compute_sync_plan(
    src_files: Dict[str, Dict[str, int]],
    dst_files: Dict[str, Dict[str, int]],
    mtime_strict: bool,
    mtime_tolerance_sec: int
) -> Tuple[List[str], Dict[str, int]]:
    """
    Return (to_copy_relpaths, stats)

    Rules:
      - missing on dst -> copy
      - size differs -> copy
      - size matches:
          - if mtime_strict is False -> skip
          - if mtime_strict is True:
              - if mtime differs beyond tolerance -> copy
              - else -> skip
    """
    to_copy: List[str] = []
    same_size_skipped = 0
    same_size_mtime_skipped = 0
    same_size_mtime_copy = 0
    size_diff_copy = 0
    missing_copy = 0

    for rel, sm in src_files.items():
        dm = dst_files.get(rel)
        if not dm:
            to_copy.append(rel)
            missing_copy += 1
            continue

        if int(sm.get("size", 0)) != int(dm.get("size", 0)):
            to_copy.append(rel)
            size_diff_copy += 1
            continue

        # sizes match:
        if not mtime_strict:
            same_size_skipped += 1
            continue

        if not mtime_close(int(sm.get("mtime", 0)), int(dm.get("mtime", 0)), tol_sec=mtime_tolerance_sec):
            to_copy.append(rel)
            same_size_mtime_copy += 1
        else:
            same_size_mtime_skipped += 1

    stats = {
        "missing_copy": missing_copy,
        "size_diff_copy": size_diff_copy,
        "same_size_skipped": same_size_skipped,
        "same_size_mtime_copy": same_size_mtime_copy,
        "same_size_mtime_skipped": same_size_mtime_skipped,
        "total_source_files": len(src_files),
        "total_dest_files": len(dst_files),
        "planned_copies": len(to_copy),
    }
    return to_copy, stats


@app.post("/api/sync")
def api_sync(req: SyncIn) -> Dict[str, Any]:
    if req.direction not in {"push", "pull"}:
        raise HTTPException(status_code=400, detail="direction must be push or pull")
    if not CONFIG.local_root or not CONFIG.remote_url:
        raise HTTPException(status_code=400, detail="Configure local_root and remote_url first.")
    if not CONFIG.remote_root:
        # not strictly required for HTTP fallback, but required for rsync directory copies and consistent behavior
        raise HTTPException(status_code=400, detail="Configure remote_root first (even if using HTTP fallback).")

    max_files = min(int(req.max_files), SYNC_MAX_FILES)
    job = create_job(kind=f"sync:{req.direction}", message=f"mtime_strict={req.mtime_strict}")

    def do_sync(job_obj: Job) -> None:
        job_obj.add_log(f"Sync start. direction={req.direction} max_files={max_files}")
        job_obj.add_log(f"mtime_strict={req.mtime_strict} mtime_tolerance_sec={req.mtime_tolerance_sec} dry_run={req.dry_run}")

        # Gather local flat list
        local_flat = list_files_flat(Path(CONFIG.local_root), max_files=max_files)

        # Gather remote flat list
        if not CONFIG.remote_url:
            raise RuntimeError("remote_url not set")
        url = CONFIG.remote_url.rstrip("/") + "/api/files_flat"
        job_obj.add_log(f"Fetching remote file list: {url}?max_files={max_files}")
        r = requests.get(url, params={"max_files": max_files}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Remote files_flat failed: {r.status_code} {r.text}")
        remote_flat = r.json()

        if local_flat.get("truncated") or remote_flat.get("truncated"):
            job_obj.add_log("WARNING: file list truncated due to max_files cap.")

        local_files = local_flat.get("files", {}) or {}
        remote_files = remote_flat.get("files", {}) or {}

        if req.direction == "push":
            src_files, dst_files = local_files, remote_files
        else:
            src_files, dst_files = remote_files, local_files

        to_copy, stats = compute_sync_plan(
            src_files=src_files,
            dst_files=dst_files,
            mtime_strict=bool(req.mtime_strict),
            mtime_tolerance_sec=int(req.mtime_tolerance_sec),
        )

        job_obj.add_log("Sync plan stats: " + json.dumps(stats, indent=2))
        if req.dry_run:
            job_obj.add_log("Dry-run mode: not enqueueing copy jobs.")
            return

        # Enqueue copy jobs sequentially (simple + reliable)
        total = max(1, len(to_copy))
        for idx, rel in enumerate(to_copy, start=1):
            job_obj.progress = (idx - 1) / total
            job_obj.add_log(f"[{idx}/{len(to_copy)}] enqueue copy {req.direction} rel={rel}")

            # Ensure target dir exists
            parent_rel = str(Path(rel).parent.as_posix())
            if parent_rel in {"", "."}:
                parent_rel = ""

            if req.direction == "push":
                if parent_rel:
                    mk = requests.post(CONFIG.remote_url.rstrip("/") + "/api/mkdirs",
                                       params={"dir_relpath": parent_rel}, timeout=30)
                    if mk.status_code != 200:
                        raise RuntimeError(f"Remote mkdirs failed for {parent_rel}: {mk.status_code} {mk.text}")
            else:
                # pull: ensure local directory exists
                ensure_dir((Path(CONFIG.local_root).resolve() / Path(rel).parent).resolve())

            # Enqueue actual copy job (reuses /api/copy logic)
            _ = api_copy(CopyIn(direction=req.direction, relpath=rel))

        job_obj.add_log(f"Sync queued {len(to_copy)} copy jobs.")
        job_obj.progress = 1.0

    JOB_QUEUE.put((job.id, do_sync))
    return {"ok": True, "job_id": job.id}


# =============================================================================
# Web UI (single-page)
# =============================================================================

HTML_PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Model Sync (v1.3)</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 0; padding: 0; }
    header { padding: 12px 16px; border-bottom: 1px solid #ddd; display:flex; gap:16px; align-items:center; justify-content:space-between; }
    h1 { font-size: 16px; margin:0; }
    main { padding: 12px 16px; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
    .row { display:flex; gap: 10px; flex-wrap: wrap; align-items:center; }
    label { font-size: 12px; color:#444; display:block; }
    input { padding: 6px 8px; border: 1px solid #ccc; border-radius: 6px; min-width: 260px; }
    button { padding: 7px 10px; border-radius: 8px; border: 1px solid #777; background: #fff; cursor: pointer; }
    button:hover { background:#f5f5f5; }
    .tree {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      max-height: 520px;
      overflow:auto;
      border:1px solid #eee;
      border-radius: 8px;
      padding: 8px;
      /* requested: background color for tree areas */
      background: #f6f8ff;
    }
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
    .peer { display:flex; align-items:center; gap:8px; }
    .dot { width:10px; height:10px; border-radius:999px; background:#bbb; display:inline-block; }
    .dot.ok { background:#2ecc71; }
    .dot.down { background:#e74c3c; }
    .small { font-size: 11px; color:#666; }
    .syncbar { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
    select { padding:6px 8px; border-radius:8px; border:1px solid #ccc; }
  </style>
</head>
<body>
<header>
  <div style="display:flex; align-items:center; gap:12px;">
    <h1>Model Sync (v1.3)</h1>
    <span class="muted">Local tree always renders. Remote errors won’t break the page.</span>
  </div>
  <div class="peer">
    <span id="peerDot" class="dot"></span>
    <span id="peerText" class="small">Peer: (not checked)</span>
  </div>
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
        <label>Download directory (absolute OR relative to local root)</label>
        <input id="dl_dest" placeholder="flux/ (or /opt/ai/comfy/models/flux/)"/>
      </div>
      <div>
        <label>Where</label>
        <select id="dl_side">
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
    </div>
  </div>

  <div class="grid" style="margin-top:16px">
    <div class="card">
      <div class="syncbar" style="justify-content:space-between">
        <div class="row" style="gap:12px;">
          <strong>Local</strong>
          <span class="pill">drag files ➜ Remote dropzone</span>
        </div>
        <div class="row" style="gap:10px;">
          <label style="display:flex; align-items:center; gap:8px; margin:0;">
            <input id="mtime_strict" type="checkbox" />
            mtime strict (only when size matches)
          </label>
          <button onclick="sync('push')">Sync ➜ Remote</button>
        </div>
      </div>

      <div id="tree_local" class="tree"></div>
      <div id="drop_remote" class="dropzone">
        Drop here to COPY TO REMOTE (push)
        <div class="muted">Directories require rsync. Files can use rsync or HTTP.</div>
      </div>
    </div>

    <div class="card">
      <div class="syncbar" style="justify-content:space-between">
        <div class="row" style="gap:12px;">
          <strong>Remote</strong>
          <span class="pill">drag files ➜ Local dropzone</span>
        </div>
        <div class="row" style="gap:10px;">
          <button onclick="sync('pull')">Sync ➜ Local</button>
        </div>
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

async function sync(direction){
  const mtimeStrict = !!el("mtime_strict").checked;
  const res = await apiPost("/api/sync", {
    direction,
    mtime_strict: mtimeStrict,
    mtime_tolerance_sec: 2,
    max_files: 500,
    dry_run: false
  });
  if (res.status !== 200){
    alert("Sync failed: " + JSON.stringify(res.json));
  } else {
    setStatus("Sync started (job): " + res.json.job_id);
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
      <details open style="margin-top:6px">
        <summary class="muted">log</summary>
        <pre style="white-space:pre-wrap; font-size:11px; margin:6px 0 0 0">${(j.log || []).slice(-30).join("\n")}</pre>
      </details>
    `;
    div.appendChild(row);
  });
}

async function refreshPeer(){
  try {
    const ps = await apiGet("/api/peer_status");
    const dot = el("peerDot");
    dot.classList.remove("ok","down");
    dot.classList.add(ps.ok ? "ok" : "down");
    const t = ps.last_check ? new Date(ps.last_check * 1000).toLocaleTimeString() : "never";
    el("peerText").innerText = `Peer: ${ps.ok ? "OK" : "DOWN"} (${ps.detail}) @ ${t}`;
  } catch (e) {
    const dot = el("peerDot");
    dot.classList.remove("ok");
    dot.classList.add("down");
    el("peerText").innerText = "Peer: (error reading status)";
  }
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
