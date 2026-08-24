# BatchFlow — SC26 Artifact

This repository contains the BatchFlow implementation and experiment framework used to evaluate BatchFlow against PyTorch, TensorSocket, and CoorDL.

The artifact contains:

- the BatchFlow coordinator and data workers;
- local and multi-node topology configurations;
- dataset and scheduler configurations;
- experiment runners for BatchFlow and the evaluated baselines;
- scripts for collecting per-job and aggregate results.

## Quick Start

The default BatchFlow configuration runs one coordinator and four workers on the local machine.

Start Redis and verify that it is reachable:

```bash
redis-cli ping
```

Expected output:

```text
PONG
```

Start BatchFlow:

```bash
python -m batchflow.deployment.launch_batchflow
```

A successful launch should report that the coordinator is ready, the configured workers have started, and BatchFlow is running.

Runtime logs are written under the configured logging directory, for example:

```text
batchflow/logs/
├── batchflow.log
├── coordinator.log
├── worker-local-worker-0.log
├── worker-local-worker-1.log
├── worker-local-worker-2.log
└── worker-local-worker-3.log
```

## Dataset Preparation

The evaluation uses ImageNet-1K, Open Images, and COCO, with datasets stored in Amazon S3 for the cloud experiments.

Detailed download, preprocessing, S3 upload, and expected directory-layout instructions are provided in:

[`datasets/prepare_datasets.md`](datasets/prepare_datasets.md)

After preparing a dataset, update the corresponding configuration under:

```text
batchflow/config/dataset/
```

For example:

```text
batchflow/config/dataset/imagenet.yaml
batchflow/config/dataset/openimages.yaml
batchflow/config/dataset/coco.yaml
```

The BatchFlow deployment and experiment workload must reference the same dataset.

## Deployment

BatchFlow uses a topology file to describe where the coordinator and workers run. Each machine participating in BatchFlow is represented as a node, and `node_id` identifies the node launched by the current process.

### Local Deployment

The default local topology runs the coordinator and workers on the same machine:

```yaml
# batchflow/config/topology/local.yaml

nodes:
  local:
    host: 127.0.0.1
    worker_count: 4
    worker_port_start: 60061

coordinator:
  node: local
  port: 50051

redis:
  host: 127.0.0.1
  port: 6379
  ssl: false

startup_timeout_seconds: 30.0
```

Start the default local topology with:

```bash
python -m batchflow.deployment.launch_batchflow
```

The default configuration uses:

```text
topology=local
node_id=local
```

so the same machine runs the coordinator and four workers.

### AWS / Multi-Node Deployment

A topology can contain multiple machines:

```yaml
# batchflow/config/topology/aws.yaml

nodes:
  node-0:
    host: 10.0.1.10
    worker_count: 8
    worker_port_start: 60061

  node-1:
    host: 10.0.2.10
    worker_count: 24
    worker_port_start: 61061

coordinator:
  node: node-0
  port: 50051

redis:
  host: batchflow-cache.example.amazonaws.com
  port: 6379
  ssl: false

startup_timeout_seconds: 60.0
```

Here:

- `node-0` runs the coordinator and 8 workers;
- `node-1` runs 24 workers;
- all workers connect to the coordinator on `node-0`;
- Redis provides shared storage for reusable batch payloads.

On `node-0`:

```bash
python -m batchflow.deployment.launch_batchflow topology=aws node_id=node-0
```

On `node-1`:

```bash
python -m batchflow.deployment.launch_batchflow topology=aws node_id=node-1
```

The launcher determines from the topology whether the current node runs the coordinator, workers, or both. There is no separate co-located or disaggregated deployment mode.

## Configuration

The main BatchFlow configuration selects the dataset, topology, and scheduler settings:

```yaml
defaults:
  - _self_
  - dataset: cifar10
  - topology: local
  - scheduler: default

node_id: local
```

The configuration groups have separate responsibilities:

```text
dataset    -> dataset location, batching, transforms, and shuffle settings
scheduler  -> scheduling and reusable-batch behavior
topology   -> coordinator, worker, and Redis placement
node_id    -> topology node launched on the current machine
```

### Redis and Reusable Batches

Redis stores shared reusable batch payloads. The topology specifies how BatchFlow reaches Redis:

```yaml
redis:
  host: 127.0.0.1
  port: 6379
  ssl: false
```

Shared reuse is controlled by the scheduler configuration:

```yaml
reuse_enabled: true
```

With reuse enabled, batches selected for reusable storage may be retained in Redis and reused by other jobs.

Transient batches remain in worker-local memory and expire automatically.

## Running Experiments

Experiments are launched through a single Hydra entry point:

```bash
python -m experiments.run_experiment system=<system> workload=<workload>
```

Supported systems are:

```text
pytorch
batchflow
tensorsocket
coordl
```

Available workloads include:

| Workload | Dataset | Jobs |
| --- | --- | ---: |
| `cifar10_1j_resnet18` | CIFAR-10 | 1 |
| `cifar10_4j_mixed` | CIFAR-10 | 4 |
| `imagenet_1j_resnet18` | ImageNet-1K | 1 |
| `imagenet_4j_mixed` | ImageNet-1K | 4 |
| `openimages_1j_vit_b_32` | Open Images | 1 |
| `openimages_4j_mixed` | Open Images | 4 |
| `coco_1j_albef_2` | COCO retrieval | 1 |
| `coco_4j_albef` | COCO retrieval | 4 |

The experiment framework loads the dataset configuration associated with the workload. The corresponding dataset must already be available at the configured location.

See [`datasets/prepare_datasets.md`](datasets/prepare_datasets.md) for dataset preparation instructions.

### PyTorch Baseline

PyTorch reads the configured dataset directly. No BatchFlow service is required.

```bash
python -m experiments.run_experiment \
  system=pytorch \
  workload=cifar10_1j_resnet18
```

For the four-job ImageNet workload:

```bash
python -m experiments.run_experiment \
  system=pytorch \
  workload=imagenet_4j_mixed
```

### BatchFlow

Start BatchFlow first using the same dataset as the workload.

For ImageNet:

```bash
# Terminal 1
python -m batchflow.deployment.launch_batchflow dataset=imagenet
```

Then run the experiment:

```bash
# Terminal 2
python -m experiments.run_experiment \
  system=batchflow \
  workload=imagenet_4j_mixed
```

The BatchFlow experiment client connects to the coordinator address configured in:

```text
experiments/config/system/batchflow.yaml
```

For a multi-node AWS deployment, launch the configured topology on each BatchFlow node before starting the experiment.

### TensorSocket Baseline

The experiment runner starts the TensorSocket producer automatically and then launches the configured training jobs:

```bash
python -m experiments.run_experiment \
  system=tensorsocket \
  workload=imagenet_4j_mixed
```

No separate TensorSocket service command is required.

### CoorDL Baseline

CoorDL uses Redis as a short-lived staging store.

Start Redis at the endpoint configured in:

```text
experiments/config/system/coordl.yaml
```

Then run:

```bash
python -m experiments.run_experiment \
  system=coordl \
  workload=imagenet_4j_mixed
```

The CoorDL runner starts its preparation-owner processes automatically and removes its staging namespace when the run completes.

## Experiment Output

Each run is written to:

```text
exp_results/<workload>/<system>/<run_id>/
```

The experiment reporter creates:

```text
config.yaml
per_batch_metrics_<job>.csv
per_job_summary.csv
aggregate_summary.csv
```

`config.yaml` contains the resolved configuration used for the run.

`per_batch_metrics_<job>.csv` contains one row for each warmup or measured training batch.

`per_job_summary.csv` contains measured summary statistics for each training job. Warmup batches are excluded from these statistics.

`aggregate_summary.csv` contains run-level throughput and configured cost-efficiency metrics across all jobs.

## Notes

- Multi-job workloads launch one training process per configured job.
- CUDA jobs are assigned by job order (`job 0 -> cuda:0`, `job 1 -> cuda:1`, etc.).
- The workload configuration controls the number of warmup and measured batches.
- Dataset paths and AWS endpoints must be updated to match the resources used for the experiment.
- Detailed dataset preparation instructions are available in [`datasets/prepare_datasets.md`](datasets/prepare_datasets.md).
