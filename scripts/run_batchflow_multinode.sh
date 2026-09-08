#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${BATCHFLOW_PYTHON:-$HOME/miniconda3/envs/batchflow/bin/python}"

TOPOLOGY_CONFIG="$ROOT/batchflow/config/topology/aws.yaml"
SCHEDULER_CONFIG="$ROOT/batchflow/config/scheduler/default.yaml"

REMOTE_USER="${BATCHFLOW_REMOTE_USER:-ubuntu}"
REMOTE_ROOT="${BATCHFLOW_REMOTE_ROOT:-/home/ubuntu/batchflow-ad-artifacts}"
REMOTE_PYTHON="${BATCHFLOW_REMOTE_PYTHON:-/home/ubuntu/miniconda3/envs/batchflow/bin/python}"
SSH_KEY="${BATCHFLOW_SSH_KEY:-$HOME/.ssh/batchflow_node}"

LOG_DIR="$ROOT/.run_logs"
REMOTE_PID_FILE="/tmp/batchflow-node1.pid"

mkdir -p "$LOG_DIR"

NODE0_LOG="$LOG_DIR/node-0.log"
NODE1_LOG="$LOG_DIR/node-1.log"
EXPERIMENT_LOG="$LOG_DIR/experiment.log"


# ---------------------------------------------------------------------------
# Read topology
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
    -i "$SSH_KEY"
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

print(
    f"Timed out waiting for {name} at {host}:{port}",
    file=sys.stderr,
)
raise SystemExit(1)
PY
}


clear_redis_if_reuse_enabled() {
    "$PYTHON" - "$SCHEDULER_CONFIG" "$TOPOLOGY_CONFIG" <<'PY'
import sys

import redis
from omegaconf import OmegaConf


scheduler_path = sys.argv[1]
topology_path = sys.argv[2]

scheduler = OmegaConf.load(scheduler_path)
topology = OmegaConf.load(topology_path)

if not scheduler.get("reuse_enabled", False):
    print("Reuse disabled; skipping Redis setup.")
    raise SystemExit(0)

redis_config = topology.get("redis")

if redis_config is None:
    raise RuntimeError(
        "reuse_enabled=true but no Redis configuration was found "
        "in the topology."
    )

host = str(redis_config.host)
port = int(redis_config.get("port", 6379))
ssl = bool(redis_config.get("ssl", False))
db = int(redis_config.get("db", 0))
key_prefix = str(redis_config.get("key_prefix", "batchflow"))

print("Reuse enabled.")
print(f"Checking Redis at {host}:{port}...")

client = redis.Redis(
    host=host,
    port=port,
    ssl=ssl,
    db=db,
    socket_connect_timeout=5,
    socket_timeout=5,
)

if not client.ping():
    raise RuntimeError("Redis ping failed.")

print("Redis is reachable.")

pattern = f"{key_prefix}:*"
deleted = 0
buffer = []

for key in client.scan_iter(match=pattern, count=500):
    buffer.append(key)

    if len(buffer) >= 500:
        deleted += client.delete(*buffer)
        buffer.clear()

if buffer:
    deleted += client.delete(*buffer)

print(f"Cleared {deleted} cached BatchFlow entries.")
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

if [[ ! -f "$SCHEDULER_CONFIG" ]]; then
    echo "Scheduler config not found: $SCHEDULER_CONFIG"
    exit 1
fi

if [[ ! -f "$SSH_KEY" ]]; then
    echo "SSH key not found: $SSH_KEY"
    exit 1
fi

if [[ ! -r "$SSH_KEY" ]]; then
    echo "SSH key is not readable: $SSH_KEY"
    exit 1
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"


# ---------------------------------------------------------------------------
# Check remote node
# ---------------------------------------------------------------------------

echo "Checking remote node..."

ssh "${SSH_OPTS[@]}" "$REMOTE" "
    test -d '$REMOTE_ROOT' &&
    test -x '$REMOTE_PYTHON'
"

echo "Remote node reachable: $REMOTE"


# ---------------------------------------------------------------------------
# Redis / reuse
# ---------------------------------------------------------------------------

echo
echo "Checking reuse configuration..."

clear_redis_if_reuse_enabled


# ---------------------------------------------------------------------------
# Node 0: coordinator + local workers
# ---------------------------------------------------------------------------

echo
echo "Starting node-0..."
echo "  log: $NODE0_LOG"

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
# Node 1: remote workers
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
echo "  node-0: coordinator + local workers"
echo "  node-1: remote workers"
echo
echo "Starting experiment..."
echo

set +e

"$PYTHON" -m experiments.run_experiment \
    system=batchflow \
    "$@" \
    2>&1 | tee "$EXPERIMENT_LOG"

EXPERIMENT_EXIT_CODE=${PIPESTATUS[0]}

set -e

if [[ "$EXPERIMENT_EXIT_CODE" -eq 0 ]]; then
    echo
    echo "Experiment completed successfully."
else
    echo
    echo "Experiment failed with exit code $EXPERIMENT_EXIT_CODE."
    echo
    echo "Logs:"
    echo "  $NODE0_LOG"
    echo "  $NODE1_LOG"
    echo "  $EXPERIMENT_LOG"
fi

exit "$EXPERIMENT_EXIT_CODE"