#!/usr/bin/env bash
set -euo pipefail

echo "===== Ray Bootstrap ====="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Platform detection: AWS Batch vs Kubernetes
# Map platform-specific env vars to unified names:
#   NODE_INDEX, MAIN_INDEX, NUM_NODES, HEAD_IP
# ---------------------------------------------------------------------------
if [[ -n "${AWS_BATCH_JOB_NODE_INDEX:-}" ]]; then
  PLATFORM="aws_batch"
  echo "[Platform] AWS Batch detected"

  : "${AWS_BATCH_JOB_NODE_INDEX:?}"
  : "${AWS_BATCH_JOB_MAIN_NODE_INDEX:?}"
  : "${AWS_BATCH_JOB_NUM_NODES:?}"

  NODE_INDEX="${AWS_BATCH_JOB_NODE_INDEX}"
  MAIN_INDEX="${AWS_BATCH_JOB_MAIN_NODE_INDEX}"
  NUM_NODES="${AWS_BATCH_JOB_NUM_NODES}"
  HEAD_IP="${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS:-}"

elif [[ -n "${K8S_RANK:-}" ]]; then
  PLATFORM="k8s"
  echo "[Platform] Kubernetes detected"

  : "${K8S_RANK:?}"
  : "${K8S_WORLD_SIZE:?}"
  : "${K8S_MASTER_ADDR:?}"
  : "${K8S_MASTER_PORT:?}"

  NODE_INDEX="${K8S_RANK}"
  MAIN_INDEX="0"
  NUM_NODES="${K8S_WORLD_SIZE}"
  HEAD_IP="${K8S_MASTER_ADDR}"

else
  echo "[FATAL] Unknown platform. Set AWS Batch or K8S environment variables."
  exit 1
fi

# ---------------------------------------------------------------------------
# Common config
# ---------------------------------------------------------------------------
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L >/dev/null 2>&1 && nvidia-smi -L | wc -l || echo 0)}"
NUM_CPUS="${NUM_CPUS:-$(nproc)}"

# Temp dir: custom for AWS Batch, Ray default for K8s
TEMP_DIR_ARGS=()
if [[ "${PLATFORM}" == "aws_batch" ]]; then
  TEMP_DIR="${RAY_TEMP_DIR:-/tmp/instance_storage/ray_tmp}"
  mkdir -p "${TEMP_DIR}"
  TEMP_DIR_ARGS=(--temp-dir="${TEMP_DIR}")
fi

source "${SCRIPT_DIR}/.venv/bin/activate"

# Cleanup
ray stop --force >/dev/null 2>&1 || true
pkill -9 ray >/dev/null 2>&1 || true
sleep 2

export NUM_NODES  # used by the inline Python below

if [[ "${NODE_INDEX}" == "${MAIN_INDEX}" ]]; then
  echo "[Ray] Starting HEAD node..."

  ray start \
    --head \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}" \
    --num-cpus="${NUM_CPUS}" \
    --num-gpus="${NUM_GPUS}" \
    --disable-usage-stats \
    "${TEMP_DIR_ARGS[@]}"

  echo "[Ray] Head started. Waiting for workers..."

  python - <<'EOF'
import os, time, ray
ray.init(address="auto")
expected = int(os.environ["NUM_NODES"])
while True:
    alive = [n for n in ray.nodes() if n.get("Alive")]
    print(f"[Ray] Alive nodes: {len(alive)}/{expected}", flush=True)
    if len(alive) >= expected:
        break
    time.sleep(5)
print("[Ray] Cluster ready.", flush=True)
EOF

  exit 0

else
  if [[ -z "${HEAD_IP}" ]]; then
    echo "[Ray][FATAL] Worker does not know head IP."
    exit 1
  fi

  echo "[Ray] Starting WORKER node, connecting to ${HEAD_IP}:${RAY_PORT}..."

  ray start \
    --address="${HEAD_IP}:${RAY_PORT}" \
    --num-cpus="${NUM_CPUS}" \
    --num-gpus="${NUM_GPUS}" \
    --disable-usage-stats

  echo "[Ray] Worker joined."

  if [[ "${INTERACTIVE_DEBUG:-0}" == "1" ]] || pgrep -x sshd >/dev/null 2>&1; then
    echo "[Ray] Interactive debug detected; not blocking."
    exit 0
  fi

  tail -f /dev/null
fi
