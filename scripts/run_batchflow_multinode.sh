#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${BATCHFLOW_PYTHON:-$HOME/miniconda3/envs/batchflow/bin/python}"
TOPOLOGY_CONFIG="$ROOT/batchflow/config/topology/aws.yaml"

REMOTE_USER="${BATCHFLOW_REMOTE_USER:-ubuntu}"
REMOTE_ROOT="${BATCHFLOW_REMOTE_ROOT:-/home/ubuntu/batchflow-ad-artifacts}"
REMOTE_PYTHON="${BATCHFLOW_REMOTE_PYTHON:-/home/ubuntu/miniconda3/envs/batchflow/bin/python}"

LOG_DIR="$ROOT/.run_logs"
REMOTE_PID_FILE="/tmp/batchflow-node1.pid"

mkdir -p "$LOG_DIR"

NODE0_LOG="$LOG_DIR/node-0.log"
NODE1_LOG="$LOG_DIR/node-1.log"
EXPERIMENT_LOG="$LOG_DIR/experiment.log"


# ---------------------------------------------------------------------------
# Read addresses directly from the topology config.
# ---------------------------------------------------------------------------

read -r NODE1_HOST NODE1_PORT COORDINATOR_PORT < <(
    "$PYTHON" - "$TOPOLOGY_CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])

print(
    config.nodes["node-1"].host,
    config.nodes["node-1"].worker_port_start,
    config.coordinator.port,
)
PY
)

REMOTE="${REMOTE_USER}@${NODE1_HOST}"

SSH_OPTS=(
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o StrictHostKeyChecking=accept-new
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

wait_for_port() {
    local host="$1"
    local port="$2"
    local name="$3"
    local timeout="${4:-60}"

    "$PYTHON" - "$host" "$port" "$name" "$timeout" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
name = sys.argv[3]
timeout = float(sys.argv[4])

deadline = time.monotonic() + timeout

while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1):
            print(f"{name} ready at {host}:{port}")
            raise SystemExit(0)
    except OSError:
        time.sleep(0.5)

print(f"Timed out waiting for {name} at {host}:{port}", file=sys.stderr)
raise SystemExit(1)
PY
}


cleanup() {
    local exit_code=$?

    trap - EXIT INT TERM

    echo
    echo "Stopping BatchFlow..."

    if [[ -n "${NODE0_PID:-}" ]]; then
        kill -TERM -- "-${NODE0_PID}" 2>/dev/null || true
    fi

    if [[ -n "${REMOTE_SSH_PID:-}" ]]; then
        ssh "${SSH_OPTS[@]}" "$REMOTE" "
            if [[ -f '$REMOTE_PID_FILE' ]]; then
                pid=\$(cat '$REMOTE_PID_FILE')
                kill -TERM -- -\$pid 2>/dev/null || true
                rm -f '$REMOTE_PID_FILE'
            fi
        " >/dev/null 2>&1 || true

        kill "$REMOTE_SSH_PID" 2>/dev/null || true
    fi

    wait "${NODE0_PID:-}" 2>/dev/null || true
    wait "${REMOTE_SSH_PID:-}" 2>/dev/null || true

    echo "BatchFlow stopped."
    exit "$exit_code"
}


trap cleanup EXIT INT TERM


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON"
    exit 1
fi

if [[ ! -f "$TOPOLOGY_CONFIG" ]]; then
    echo "Topology config not found: $TOPOLOGY_CONFIG"
    exit 1
fi

echo "Checking remote node..."

ssh "${SSH_OPTS[@]}" "$REMOTE" "
    test -d '$REMOTE_ROOT' &&
    test -x '$REMOTE_PYTHON'
"

echo "Remote node reachable: $REMOTE"


# ---------------------------------------------------------------------------
# Node 0: coordinator + 48 local workers
# ---------------------------------------------------------------------------

echo
echo "Starting node-0..."
echo "  log: $NODE0_LOG"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

setsid "$PYTHON" -m batchflow.deployment.launch_batchflow \
    topology=aws \
    node_id=node-0 \
    >"$NODE0_LOG" 2>&1 &

NODE0_PID=$!

wait_for_port \
    "127.0.0.1" \
    "$COORDINATOR_PORT" \
    "BatchFlow coordinator"


# ---------------------------------------------------------------------------
# Node 1: 16 remote workers
# ---------------------------------------------------------------------------

echo
echo "Starting node-1..."
echo "  host: $REMOTE"
echo "  log:  $NODE1_LOG"

ssh "${SSH_OPTS[@]}" "$REMOTE" "bash -s" >"$NODE1_LOG" 2>&1 <<EOF &
set -euo pipefail

cd "$REMOTE_ROOT"
export PYTHONPATH=".:\\${PYTHONPATH:-}"

setsid "$REMOTE_PYTHON" -m batchflow.deployment.launch_batchflow \
    topology=aws \
    node_id=node-1 &

pid=\$!
echo "\$pid" > "$REMOTE_PID_FILE"

cleanup_remote() {
    kill -TERM -- -"\$pid" 2>/dev/null || true
    rm -f "$REMOTE_PID_FILE"
}

trap cleanup_remote EXIT INT TERM

wait "\$pid"
EOF

REMOTE_SSH_PID=$!

wait_for_port \
    "$NODE1_HOST" \
    "$NODE1_PORT" \
    "BatchFlow remote worker"


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

echo
echo "BatchFlow cluster ready."
echo
echo "  node-0: coordinator + 48 workers"
echo "  node-1: 16 workers"
echo "  total:  64 workers"
echo
echo "Starting experiment..."
echo

set +e

"$PYTHON" -m experiments.run_experiment "$@" \
    2>&1 | tee "$EXPERIMENT_LOG"

EXPERIMENT_EXIT_CODE=${PIPESTATUS[0]}

set -e

if [[ "$EXPERIMENT_EXIT_CODE" -eq 0 ]]; then
    echo
    echo "Experiment completed successfully."
else
    echo
    echo "Experiment failed with exit code $EXPERIMENT_EXIT_CODE."
    echo "Logs:"
    echo "  $NODE0_LOG"
    echo "  $NODE1_LOG"
    echo "  $EXPERIMENT_LOG"
fi

exit "$EXPERIMENT_EXIT_CODE"