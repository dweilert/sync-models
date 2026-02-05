# Model Sync (ComfyUI Models) — Home LAN Sync Tool

A lightweight Python app you run on **two machines** to keep a directory of AI models (ComfyUI, etc.) in sync.

- Web UI (no auth; intended for trusted home LAN)
- Configure:
  - local root path (models directory)
  - remote node URL
  - remote root path (for rsync path building)
  - optional SSH/rsync settings (recommended)
  - optional HuggingFace token
- Visual directory trees: **local + remote**
- Drag & drop file copy between nodes
- Download a HuggingFace (or any direct) URL on local/remote/both
- Job progress + logs
- Config persisted to `~/.model_sync_config.json`

> **Best transfer method:** `rsync` over SSH (fast, resumable, handles huge model files well).  
> HTTP fallback supports **files**, not directories.

---

## Repository contents

- `model_sync_app.py` — the FastAPI server + UI (single file)
- `requirements.txt`
- (optional) `test_model_sync.py`
- (optional) `requirements-dev.txt`

---

## Quick start

### 1) Create venv + install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### 2) Run on each machine

python sync-models.py --host 0.0.0.0 --port 9090 --peer-poll-seconds 10


## Required system tools (recommended)

For best performance and directory sync, install these OS packages:

Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y rsync openssh-client

macOS
brew install rsync
# ssh is already included on macOS


If you insist on password-based SSH automation, you can use sshpass (system package), but SSH keys are strongly recommended.

Configure the app (UI)

On each node:

Set Local root (your models directory)

Set Remote URL (the other node, e.g. http://192.168.1.50:9090)

Set Remote root (the remote machine’s models directory path)

(Optional) Set SSH fields for rsync:

SSH host (remote machine IP)

SSH user

SSH key path (optional if using ssh-agent)

(Optional) Set HuggingFace Token (for gated/private downloads)

Click Save config

Config is saved at:

~/.model_sync_config.json

SSH keys setup (recommended)

On node A:

ssh-keygen -t ed25519
ssh-copy-id user@<node-b-ip>
ssh user@<node-b-ip> "echo ok"


Then in the UI on node A:

SSH host: <node-b-ip>

SSH user: user

SSH key path: ~/.ssh/id_ed25519 (optional)

Repeat the other direction if you plan to copy both ways via rsync.

Copy behavior
Drag & drop

Drag a file from Local tree → drop into “COPY TO REMOTE” to push

Drag a file from Remote tree → drop into “COPY TO LOCAL” to pull

Directories

Directory transfers require rsync.

HTTP fallback is file-only.

Safety: “file stable” check

Before a file copy begins, the app checks that the source file size is not changing (useful if a download is still in progress).

HuggingFace downloads

Paste a file URL (HuggingFace resolve/main/... or any direct URL) into the UI and choose:

local / remote / both

How “remote download” works

If you choose remote or both, this node sends an HTTP request to the other node:

POST http://REMOTE:PORT/api/download_url

JSON body includes url and optional dest_relpath

The remote node then downloads directly (using its own saved HF token, if set) and reports progress in Jobs.

Tip: For many HuggingFace files, adding ?download=true can help ensure you get the raw file.

Ports and LAN access

--port sets the port for the web UI + API.

--host 0.0.0.0 exposes it on your LAN.

Example alternate port:

python model_sync_app.py --host 0.0.0.0 --port 9091 --peer-poll-seconds 10

Peer polling

At startup you can set how often the app pings the other node’s health endpoint:

python model_sync_app.py --peer-poll-seconds 5


This is a connectivity/health check; it does not automatically sync by itself.

Testing (optional)

Install dev deps:

pip install -r requirements-dev.txt


Run:

pytest -q

Security notes

This app has no authentication by design (home LAN).

Your HuggingFace token (if provided) is stored in plain text in ~/.model_sync_config.json.

Do not expose this service to the public internet.

Roadmap ideas (if you want them)

Directory sync via tar/zip streaming (HTTP fallback)

Deep compare mode (hash-based)

Filters/search in tree (e.g., “flux”, “sdxl”, “vae”)

Conflict policies (skip/overwrite/newer-wins)

Packaging as a small systemd service



Example .env (unchanged, shown for clarity)
MODEL_SYNC_PORT=9090
MODEL_SYNC_MODELS_DIR=/opt/ai/models
MODEL_SYNC_PEER_POLL_SECONDS=10
MODEL_SYNC_SSH_DIR=/home/youruser/.ssh
TZ=America/Chicago


(macOS example)

MODEL_SYNC_MODELS_DIR=/Users/dave/ai/models
MODEL_SYNC_SSH_DIR=/Users/dave/.ssh

Build & Run (unchanged)
docker compose build
docker compose up -d


Logs:

docker compose logs -f

How to confirm it’s working
1️⃣ Check that Docker created the volume
docker volume ls | grep model_sync_config


You should see something like:

local     model_sync_config

2️⃣ Inspect the volume
docker volume inspect model_sync_config


You’ll see Docker’s internal mount point (varies by OS).
