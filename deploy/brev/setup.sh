#!/bin/bash
# quorum on an NVIDIA Brev GPU VM (VM mode, Jupyter on). The Launchable's setup
# script fetches and runs this file once, at deploy.
#
# It starts two systemd units, both on loopback only (quorum has no auth):
#   quorum-llm  llama-server (CUDA, Docker) on 127.0.0.1:8005 with the chosen model
#   quorum      the quorum shim on 127.0.0.1:8017, in front of it
# Both come back after a stop/start. Reach them from the Jupyter notebook
# (~/workspace/quorum-quickstart.ipynb) or `brev port-forward <instance> --port 8017:8017`.
#
# Launch parameter QUORUM_MODEL: judge (default) | qwen3-4b-2507
# Setup output is kept in the Brev logs and in ~/workspace/quorum-setup.log.
set -euo pipefail

MODEL_CHOICE="${QUORUM_MODEL:-judge}"
REPO_URL="https://github.com/sypherin/quorum"
REPO_REF="${QUORUM_REF:-master}"
# pinned llama.cpp build: flags below are checked against it
IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda-b11151"

case "$MODEL_CHOICE" in
  judge)
    HF_REPO=AltronisSG/judgment-qc-gate-qwen3-4b-GGUF
    HF_REV=95641f5cbebd8761958ab01bd6c0e9192a574456
    FILE=judgment-qwen3-4b-Q8_0.gguf
    SHA256=556228ac91946bb8413fcd07ad17283060ba941d5832118fdc2efba2b5ec8299
    ALIAS=judgment-qc-gate-qwen3-4b ;;
  qwen3-4b-2507)
    HF_REPO=unsloth/Qwen3-4B-Instruct-2507-GGUF
    HF_REV=a06e946bb6b655725eafa393f4a9745d460374c9
    FILE=Qwen3-4B-Instruct-2507-Q8_0.gguf
    SHA256=391c1e410fd9f4cf2de2b510273b56a84c19ce18f4fa3bfb3774031dac4ef068
    ALIAS=qwen3-4b-2507 ;;
  *)
    echo "QUORUM_MODEL must be 'judge' or 'qwen3-4b-2507', got '$MODEL_CHOICE'" >&2
    exit 2 ;;
esac

# Brev VMs run as ubuntu; the script may be started as root or as ubuntu.
if id ubuntu >/dev/null 2>&1; then RUN_USER=ubuntu; else RUN_USER=$(id -un); fi
RUN_HOME=$(getent passwd "$RUN_USER" | cut -d: -f6)
WORK="$RUN_HOME/workspace"          # persists across stop/start
APP="$WORK/quorum"
MODELS="$WORK/models"
SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"
DOCKER=$(command -v docker) || { echo "[quorum-setup] FAILED: docker not found (use a VM-mode Launchable)" >&2; exit 1; }
# files under the workspace belong to the VM user: git refuses a repo owned by
# someone else ("dubious ownership"), so everything there runs as that user
as_user() { if [ "$(id -un)" = "$RUN_USER" ]; then "$@"; else sudo -u "$RUN_USER" -H "$@"; fi; }

as_user mkdir -p "$WORK"
exec > >(tee -a "$WORK/quorum-setup.log") 2>&1
log() { echo "[quorum-setup $(date -u +%H:%M:%S)] $*"; }
die() { echo "[quorum-setup] FAILED: $*" >&2; exit 1; }

log "1/6 GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || die "nvidia-smi failed: this Launchable needs an NVIDIA GPU instance"

log "2/6 code ($REPO_URL @ $REPO_REF)"
if [ -d "$APP/.git" ]; then
  as_user git -C "$APP" fetch -q origin "$REPO_REF"
  as_user git -C "$APP" checkout -q FETCH_HEAD
else
  as_user git clone -q --branch "$REPO_REF" "$REPO_URL" "$APP"
fi
log "quorum at $(as_user git -C "$APP" rev-parse --short HEAD)"

log "3/6 model $HF_REPO/$FILE"
as_user mkdir -p "$MODELS"
if [ -f "$MODELS/$FILE" ] && echo "$SHA256  $MODELS/$FILE" | sha256sum -c --quiet 2>/dev/null; then
  log "already present, sha256 ok"
else
  as_user curl -fL --retry 5 --retry-delay 10 -o "$MODELS/$FILE.partial" \
    "https://huggingface.co/$HF_REPO/resolve/$HF_REV/$FILE"
  echo "$SHA256  $MODELS/$FILE.partial" | sha256sum -c --quiet \
    || die "sha256 mismatch on $FILE: refusing to serve it"
  as_user mv "$MODELS/$FILE.partial" "$MODELS/$FILE"
  log "downloaded, sha256 ok"
fi

log "4/6 python env"
if ! as_user python3 -m venv "$APP/.venv" 2>/dev/null; then
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq python3-venv
  as_user python3 -m venv "$APP/.venv"
fi
as_user "$APP/.venv/bin/pip" install -q --upgrade pip
# serve: httpx uvicorn; calibrate: scipy; benchmark rerun: datasets pandas huggingface_hub
as_user "$APP/.venv/bin/pip" install -q httpx uvicorn scipy datasets pandas huggingface_hub

log "5/6 services"
$SUDO docker pull -q "$IMAGE"
# fail in seconds, not after a 10-minute health wait, when Docker cannot reach the GPU
$SUDO docker run --rm --gpus all --entrypoint /bin/true "$IMAGE" \
  || die "docker run --gpus all failed: the NVIDIA container toolkit is missing or broken"
$SUDO tee /etc/systemd/system/quorum-llm.service >/dev/null <<EOF
[Unit]
Description=llama-server for quorum ($ALIAS, loopback :8005)
After=docker.service
Requires=docker.service

[Service]
ExecStartPre=-$DOCKER rm -f quorum-llm
ExecStart=$DOCKER run --rm --name quorum-llm --gpus all -p 127.0.0.1:8005:8080 -v $MODELS:/models:ro $IMAGE -m /models/$FILE --alias $ALIAS --host 0.0.0.0 --port 8080 --ctx-size 4096 --parallel 1 -ngl 99 -fa on --jinja --reasoning-budget 0 --cache-ram 0
ExecStop=$DOCKER stop quorum-llm
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
$SUDO tee /etc/systemd/system/quorum.service >/dev/null <<EOF
[Unit]
Description=quorum shim (loopback :8017 -> llama-server :8005)
After=quorum-llm.service
Wants=quorum-llm.service

[Service]
User=$RUN_USER
WorkingDirectory=$APP
Environment=QUORUM_UPSTREAM=http://127.0.0.1:8005
Environment=QUORUM_MODEL_ALIAS=$ALIAS
ExecStart=$APP/.venv/bin/uvicorn quorum.serve:app --host 127.0.0.1 --port 8017
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
$SUDO systemctl daemon-reload
$SUDO systemctl enable -q quorum-llm.service quorum.service
$SUDO systemctl restart quorum-llm.service
$SUDO systemctl restart quorum.service

log "6/6 checks"
for _ in $(seq 1 120); do
  curl -sf 127.0.0.1:8005/health >/dev/null && break
  sleep 5
done
curl -sf 127.0.0.1:8005/health >/dev/null \
  || die "llama-server not healthy after 10 min: $SUDO journalctl -u quorum-llm -n 50"
# a model that silently landed on the CPU still answers, just 10x slower: check the offload
OFFLOAD=$($SUDO docker logs quorum-llm 2>&1 | grep -oE 'offloaded [0-9]+/[0-9]+ layers to GPU' | tail -1 || true)
[ -n "$OFFLOAD" ] || die "no GPU offload line in the llama-server log: $SUDO docker logs quorum-llm"
read -r DONE TOTAL < <(echo "$OFFLOAD" | sed -E 's/offloaded ([0-9]+)\/([0-9]+).*/\1 \2/')
[ "$DONE" -eq "$TOTAL" ] || die "only $OFFLOAD: the model is partly on the CPU"
log "llama-server: $OFFLOAD"

for _ in $(seq 1 30); do
  curl -sf 127.0.0.1:8017/healthz >/dev/null && break
  sleep 2
done
REPLY=$(curl -sf 127.0.0.1:8017/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Customer: my card was charged twice for one order and I need it fixed today.",
  "questions": {"urgent": {"type": "noul", "instructions": "Does this express urgency?"}}}') \
  || die "quorum did not answer: $SUDO journalctl -u quorum -n 50"
echo "$REPLY" | "$APP/.venv/bin/python" -c '
import json, sys
a = json.load(sys.stdin)["answers"]["urgent"]
assert a.get("answer") in ("yes", "no"), f"no answer in {a}"
p = a.get("noul")
# a model sure enough that "no" never reaches its top tokens returns no distribution
print("[quorum-setup] smoke: urgent =", a["answer"],
      f"P(yes) = {p:.3f}" if p is not None else "(no distribution: model fully confident)")'

# a copy at the top of the workspace, where Jupyter's file browser shows it; the
# clone's own copy stays pristine for the next fetch
as_user cp "$APP/deploy/brev/quickstart.ipynb" "$WORK/quorum-quickstart.ipynb"
log "done. quorum on 127.0.0.1:8017 ($ALIAS). Open workspace/quorum-quickstart.ipynb in Jupyter."
