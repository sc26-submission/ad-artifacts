from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Iterable
from typing import Any

import torch

from batchflow.common.utils import ResourceMonitor
from experiments.common.training import TrainingComponents


def configure_process_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def resolve_device_and_amp(
    device: str,
    use_amp: bool | None,
    *,
    job_index: int,
) -> tuple[torch.device, bool]:
    requested = device.lower()

    if requested in {"auto", "cuda"}:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()

            if job_index >= device_count:
                raise RuntimeError(
                    f"Job {job_index} requires a GPU, but only {device_count} CUDA device(s) "
                    f"are available."
                )

            resolved = torch.device(f"cuda:{job_index}")

        elif requested == "cuda":
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")

        else:
            resolved = torch.device("cpu")

    else:
        resolved = torch.device(device)

        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Device {resolved} was requested, but CUDA is not available.")

    amp_enabled = resolved.type == "cuda" if use_amp is None else use_amp and resolved.type == "cuda"
    return resolved, amp_enabled


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_scalar(batch: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = batch.get(key)

    if value is None:
        return default

    if isinstance(value, torch.Tensor):
        return float(value.sum().item()) if value.numel() else default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_batch_scalar(
    batch: dict[str, Any],
    *keys: str,
    default: float = 0.0,
) -> float:
    for key in keys:
        if batch.get(key) is not None:
            return _batch_scalar(batch, key, default)

    return default


def _extract_batch_timings(batch: dict[str, Any]) -> tuple[float, float, float]:
    """Return I/O, decode, and transform time for any supported data backend."""

    if batch.get("trainer_decode_time_sec") is not None:
        io_time = _first_batch_scalar(
            batch,
            "coordinator_wait_total_time_sec",
            "batchflow_coordinator_wait_total_time_sec",
        ) + _first_batch_scalar(
            batch,
            "fetch_time_sec",
            "batchflow_fetch_time_sec",
        )

        decode_time = _batch_scalar(batch, "trainer_decode_time_sec")

        transform_time = _first_batch_scalar(
            batch,
            "trainer_transform_time_sec",
            "batchflow_worker_transform_time_sec",
            "worker_transform_time_sec",
        )

        return io_time, decode_time, transform_time

    return (
        _batch_scalar(batch, "io_time_sec"),
        _batch_scalar(batch, "decode_time_sec"),
        _batch_scalar(batch, "transform_time_sec"),
    )


def _extract_batch_indices(batch: dict[str, Any]) -> list[Any]:
    for key in ("batch_indices", "index", "sample_id"):
        value = batch.get(key)

        if value is None:
            continue

        if hasattr(value, "tolist"):
            value = value.tolist()

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        return [value]

    return []


def _generate_batch_id(indices: list[Any]) -> str | None:
    if not indices:
        return None

    payload = repr(tuple(indices)).encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _resolve_batch_id(batch: dict[str, Any], indices: list[Any]) -> str | None:
    batch_id = batch.get("batch_id")

    if batch_id is not None:
        return str(batch_id)

    return _generate_batch_id(indices)


def _should_log_batch(batch: int, total_batches: int, log_every_batches: int) -> bool:
    return (
        batch == 1
        or batch == total_batches
        or (log_every_batches > 0 and batch % log_every_batches == 0)
    )


def _log_warmup_progress(
    logger: logging.Logger,
    *,
    prefix: str,
    batch: int,
    warmup_batches: int,
    data_time: float,
    compute_time: float,
) -> None:
    logger.info(
        f"{prefix} | warmup={batch}/{warmup_batches} | "
        f"data={data_time:.4f}s | compute={compute_time:.4f}s"
    )


def _log_progress(
    logger: logging.Logger,
    *,
    prefix: str,
    batch: int,
    num_batches: int,
    total_samples: int,
    total_data_time: float,
    total_compute_time: float,
    total_batch_time: float,
) -> None:
    avg_data_time = total_data_time / batch
    avg_compute_time = total_compute_time / batch
    batches_per_sec = batch / max(total_batch_time, 1e-12)
    samples_per_sec = total_samples / max(total_batch_time, 1e-12)

    logger.info(
        f"{prefix} | batch={batch}/{num_batches} | "
        f"data={avg_data_time:.4f}s | compute={avg_compute_time:.4f}s | "
        f"throughput={samples_per_sec:.1f} samples/s ({batches_per_sec:.2f} batches/s)"
    )


def run_training_loop(
    *,
    mode: str,
    job_id: str,
    batch_iter: Iterable[dict[str, Any]],
    training: TrainingComponents,
    num_batches: int,
    warmup_batches: int,
    device: torch.device,
    use_amp: bool,
    on_batch_end: Callable[[dict[str, Any]], None],
    logger: logging.Logger,
    log_every_batches: int = 10,
) -> None:
    """Run warmup batches followed by exactly ``num_batches`` training batches."""

    job_id = job_id or "-"
    prefix = f"{mode} | job={job_id}"

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    iterator = iter(batch_iter)
    total_loop_batches = warmup_batches + num_batches

    completed_batches = 0
    total_samples = 0
    total_data_time = 0.0
    total_compute_time = 0.0
    total_batch_time = 0.0

    run_start = time.perf_counter()

    monitor = ResourceMonitor(sample_interval_seconds=0.25, logger=logger)
    monitor.start()

    logger.info(
        f"{prefix} | started | device={device} | amp={amp_enabled} | "
        f"batches={num_batches} | warmup={warmup_batches}"
    )

    try:
        training.model.train()

        for loop_batch in range(total_loop_batches):
            is_warmup = loop_batch < warmup_batches
            batch_start = time.perf_counter()

            data_start = time.perf_counter()

            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"{prefix} | data source ended early at "
                    f"batch {loop_batch + 1}/{total_loop_batches}"
                ) from exc

            data_time = time.perf_counter() - data_start

            result = training.run_batch(
                batch,
                scaler=scaler,
                amp_enabled=amp_enabled,
            )

            batch_time = time.perf_counter() - batch_start

            loss = result.loss
            batch_size = result.batch_size
            compute_time = result.compute_time_sec
            forward_time = result.forward_time_sec
            backward_time = result.backward_time_sec
            optimizer_time = result.optimizer_step_time_sec

            if not is_warmup:
                completed_batches += 1
                total_samples += batch_size
                total_data_time += data_time
                total_compute_time += compute_time
                total_batch_time += batch_time

            batch_indices = _extract_batch_indices(batch)
            batch_id = _resolve_batch_id(batch, batch_indices)
            io_time, decode_time, transform_time = _extract_batch_timings(batch)

            elapsed_time = time.perf_counter() - run_start
            samples_per_sec = total_samples / max(total_batch_time, 1e-12)
            batches_per_sec = completed_batches / max(total_batch_time, 1e-12)

            row = {
                "system": mode,
                "job_id": str(batch.get("job_id") or job_id),
                "device": str(device),
                "batch": loop_batch,
                "warmup": int(is_warmup),
                "batch_id": batch_id,
                "batch_size": batch_size,
                "batch_indices": batch_indices,
                "total_batch_time_sec": batch_time,
                "total_load_batch_time_sec": data_time,
                "total_model_compute_time_sec": compute_time,
                "batch_io_time_sec": io_time,
                "batch_decode_time_sec": decode_time,
                "batch_transform_time_sec": transform_time,
                "forward_pass_time_sec": forward_time,
                "backward_pass_time_sec": backward_time,
                "optimizer_step_time_sec": optimizer_time,
                "loss": loss,
                "samples_per_sec": samples_per_sec,
                "batches_per_sec": batches_per_sec,
                "elapsed_time_sec": elapsed_time,
            }

            on_batch_end(row)

            if is_warmup:
                warmup_batch = loop_batch + 1

                if _should_log_batch(warmup_batch, warmup_batches, log_every_batches):
                    _log_warmup_progress(
                        logger,
                        prefix=prefix,
                        batch=warmup_batch,
                        warmup_batches=warmup_batches,
                        data_time=data_time,
                        compute_time=compute_time,
                    )

                continue

            if _should_log_batch(completed_batches, num_batches, log_every_batches):
                _log_progress(
                    logger,
                    prefix=prefix,
                    batch=completed_batches,
                    num_batches=num_batches,
                    total_samples=total_samples,
                    total_data_time=total_data_time,
                    total_compute_time=total_compute_time,
                    total_batch_time=total_batch_time,
                )

    finally:
        close = getattr(iterator, "close", None)

        if callable(close):
            close()

        monitor.stop()

    elapsed_time = time.perf_counter() - run_start

    logger.info(
        f"{prefix} | finished | batches={completed_batches} | samples={total_samples} | "
        f"elapsed={elapsed_time:.2f}s"
    )