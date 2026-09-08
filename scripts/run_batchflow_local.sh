#!/usr/bin/env bash
set -euo pipefail

cleanup() {
    if [[ -n "${BF_PID:-}" ]]; then
        echo "Stopping BatchFlow..."
        kill "$BF_PID" 2>/dev/null || true
        wait "$BF_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "Starting BatchFlow..."
python -m batchflow.deployment.launch_batchflow \
    topology=local \
    node_id=local \
    dataset=imagenet \
    > /tmp/batchflow.log 2>&1 &

BF_PID=$!

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
    echo "Timed out waiting for BatchFlow coordinator."
    cat /tmp/batchflow.log
    exit 1
fi

python -m experiments.run_experiment \
    system=batchflow \
    "$@"