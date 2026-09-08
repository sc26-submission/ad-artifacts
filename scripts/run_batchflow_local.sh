#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${BATCHFLOW_PYTHON:-$HOME/miniconda3/envs/batchflow/bin/python}"
SCHEDULER_CONFIG="$ROOT/batchflow/config/scheduler/default.yaml"

REDIS_HOST="${BATCHFLOW_REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${BATCHFLOW_REDIS_PORT:-6379}"
REDIS_KEY_PATTERN="${BATCHFLOW_REDIS_KEY_PATTERN:-batchflow:*}"

BF_PID=""
REDIS_STARTED_BY_SCRIPT=false


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

cleanup() {
    local exit_code=$?

    trap - EXIT INT TERM

    echo

    if [[ -n "$BF_PID" ]]; then
        echo "Stopping BatchFlow..."
        kill -TERM -- "-$BF_PID" 2>/dev/null || true
        wait "$BF_PID" 2>/dev/null || true
    fi

    if [[ "$REDIS_STARTED_BY_SCRIPT" == "true" ]]; then
        echo "Stopping local Redis..."
        sudo systemctl stop redis-server >/dev/null 2>&1 || true
    fi

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

if [[ ! -f "$SCHEDULER_CONFIG" ]]; then
    echo "Scheduler config not found: $SCHEDULER_CONFIG"
    exit 1
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"


# ---------------------------------------------------------------------------
# Check whether reuse is enabled
# ---------------------------------------------------------------------------

REUSE_ENABLED=$(
    "$PYTHON" - "$SCHEDULER_CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])
print("true" if config.get("reuse_enabled", False) else "false")
PY
)

echo "Reuse enabled: $REUSE_ENABLED"


# ---------------------------------------------------------------------------
# Local Redis
# ---------------------------------------------------------------------------

if [[ "$REUSE_ENABLED" == "true" ]]; then
    if redis-cli \
        -h "$REDIS_HOST" \
        -p "$REDIS_PORT" \
        ping 2>/dev/null | grep -q '^PONG$'; then

        echo "Local Redis already running."

    else
        echo "Starting local Redis..."
        sudo systemctl start redis-server
        REDIS_STARTED_BY_SCRIPT=true

        echo "Waiting for Redis..."

        for _ in $(seq 1 20); do
            if redis-cli \
                -h "$REDIS_HOST" \
                -p "$REDIS_PORT" \
                ping 2>/dev/null | grep -q '^PONG$'; then

                echo "Redis is ready."
                break
            fi

            sleep 0.5
        done

        if ! redis-cli \
            -h "$REDIS_HOST" \
            -p "$REDIS_PORT" \
            ping 2>/dev/null | grep -q '^PONG$'; then

            echo "Timed out waiting for Redis."
            exit 1
        fi
    fi

    echo "Clearing BatchFlow cache..."

    redis-cli \
        -h "$REDIS_HOST" \
        -p "$REDIS_PORT" \
        --scan \
        --pattern "$REDIS_KEY_PATTERN" |
        xargs -r -n 100 redis-cli \
            -h "$REDIS_HOST" \
            -p "$REDIS_PORT" \
            DEL \
            >/dev/null

    echo "BatchFlow cache cleared."
fi


# ---------------------------------------------------------------------------
# BatchFlow
# ---------------------------------------------------------------------------

echo
echo "Starting BatchFlow..."

setsid "$PYTHON" -m batchflow.deployment.launch_batchflow \
    topology=local \
    node_id=local \
    dataset=imagenet \
    > /tmp/batchflow.log 2>&1 &

BF_PID=$!


# ---------------------------------------------------------------------------
# Wait for coordinator
# ---------------------------------------------------------------------------

echo "Waiting for coordinator..."

for _ in $(seq 1 30); do
    if nc -z 127.0.0.1 50051 2>/dev/null; then
        echo "BatchFlow coordinator is ready."
        break
    fi

    if ! kill -0 "$BF_PID" 2>/dev/null; then
        echo "BatchFlow exited during startup:"
        cat /tmp/batchflow.log
        exit 1
    fi

    sleep 1
done

if ! nc -z 127.0.0.1 50051 2>/dev/null; then
    echo "Timed out waiting for BatchFlow coordinator:"
    cat /tmp/batchflow.log
    exit 1
fi


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

echo
echo "Starting experiment..."
echo

"$PYTHON" -m experiments.run_experiment \
    system=batchflow \
    "$@"