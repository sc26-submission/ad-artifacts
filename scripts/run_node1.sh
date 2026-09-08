#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate batchflow

export PYTHONPATH=".:${PYTHONPATH:-}"

exec python -m batchflow.deployment.launch_batchflow \
  topology=aws \
  node_id=node-1