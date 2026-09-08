from __future__ import annotations

import logging
import queue
import time
from typing import Any

from torch.utils.data import IterableDataset, get_worker_info

from batchflow.integrations.pytorch.config import (
    BatchFlowTorchConfig,
    TrainerRuntimeMetricsBuffer,
)
from batchflow.integrations.pytorch.prefetcher import (
    BatchItem,
    EndItem,
    ErrorItem,
    MultiThreadBatchFlowPrefetcher,
)


LOGGER = logging.getLogger(__name__)


class BatchFlowIterator:
    def __init__(self, config: BatchFlowTorchConfig) -> None:
        config.validate()

        self.config = config
        self.runtime_metrics = TrainerRuntimeMetricsBuffer()

        self.prefetcher = MultiThreadBatchFlowPrefetcher(
            config,
            runtime_metrics=self.runtime_metrics,
        )

        self._closed = False
        self._consumed_batches = 0
        self._queue_empty_events = 0
        self._last_status_log_time = time.monotonic()

        self.prefetcher.start()

        LOGGER.debug(
            f"BatchFlow prefetch started | "
            f"job={self._job_id} "
            f"dataset={self.config.dataset_id} "
            f"max_batches={self.config.max_batches} "
            f"prefetch_capacity={self.prefetcher.maxsize()} "
            f"fetch_workers={self.config.parallel_fetch_workers}"
        )

    @property
    def _job_id(self) -> str:
        return self.config.job_id or "-"

    def __iter__(self) -> "BatchFlowIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        if self._closed:
            raise StopIteration

        while True:
            try:
                item = self.prefetcher.get_item(
                    timeout_seconds=self.config.ready_queue_timeout_seconds,
                )

            except queue.Empty:
                self._queue_empty_events += 1

                if self.config.log_trainer_waits:
                    LOGGER.warning(
                        f"BatchFlow waiting for data | "
                        f"job={self._job_id} "
                        f"next_batch={self._consumed_batches + 1} "
                        f"wait_events={self._queue_empty_events}"
                    )

                continue

            if isinstance(item, BatchItem):
                self._consumed_batches += 1

                item.batch["trainer_queue_size_after_get"] = int(
                    self.prefetcher.qsize()
                )
                item.batch["trainer_queue_empty_events"] = int(
                    self._queue_empty_events
                )

                self._log_status()
                return item.batch

            if isinstance(item, EndItem):
                LOGGER.debug(
                    f"BatchFlow prefetch finished | "
                    f"job={self._job_id} "
                    f"batches={self._consumed_batches} "
                    f"wait_events={self._queue_empty_events}"
                )

                self.close()
                raise StopIteration

            if isinstance(item, ErrorItem):
                LOGGER.error(
                    f"BatchFlow prefetch failed | "
                    f"job={self._job_id} "
                    f"batches={self._consumed_batches} "
                    f"error={type(item.error).__name__}: {item.error}"
                )

                self.close()
                raise item.error

            self.close()

            raise RuntimeError(
                f"Unexpected BatchFlow prefetch item: "
                f"{type(item).__name__}"
            )

    def update_runtime_metrics(
        self,
        *,
        data_bottleneck_percent: float,
        avg_data_time_sec: float,
        avg_compute_time_sec: float,
        avg_coordinator_wait_total_time_sec: float,
        avg_coordinator_pending_polls: float,
    ) -> None:
        self.runtime_metrics.update(
            data_bottleneck_percent=data_bottleneck_percent,
            avg_data_time_sec=avg_data_time_sec,
            avg_compute_time_sec=avg_compute_time_sec,
            avg_coordinator_wait_total_time_sec=(
                avg_coordinator_wait_total_time_sec
            ),
            avg_coordinator_pending_polls=avg_coordinator_pending_polls,
        )

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self.prefetcher.shutdown()

        LOGGER.debug(
            f"BatchFlow prefetch closed | "
            f"job={self._job_id} "
            f"batches={self._consumed_batches}"
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _log_status(self) -> None:
        now = time.monotonic()

        log_for_batch_count = (
            self.config.log_every_n_batches > 0
            and self._consumed_batches > 0
            and self._consumed_batches % self.config.log_every_n_batches == 0
        )

        log_for_interval = (
            self.config.log_interval_seconds > 0
            and now - self._last_status_log_time
            >= self.config.log_interval_seconds
        )

        if not log_for_batch_count and not log_for_interval:
            return

        prefetched = self.prefetcher.qsize()
        prefetch_capacity = self.prefetcher.maxsize()

        LOGGER.debug(
            f"BatchFlow prefetch status | "
            f"job={self._job_id} "
            f"batches={self._consumed_batches}/{self.config.max_batches} "
            f"prefetched={prefetched}/{prefetch_capacity}"
        )

        self._last_status_log_time = now


class BatchFlowIterableDataset(IterableDataset):
    def __init__(
        self,
        config: BatchFlowTorchConfig | None = None,
        *,
        coordinator_address: str = "127.0.0.1:50051",
        dataset_id: str = "synthetic-dataset",
        max_batches: int = 100,
        job_id: str = "",
        job_index: int = 0,
        lookahead_batches: int = 16,
        request_poll_interval_seconds: float = 0.2,
        fetch_timeout_seconds: float | None = 30.0,
        coordinator_timeout_seconds: float | None = None,
        max_ready_batches: int = 4,
        parallel_fetch_workers: int = 4,
        ready_queue_timeout_seconds: float = 0.5,
        pin_memory: bool = True,
        finish_job_on_close: bool = True,
        log_interval_seconds: float = 2.0,
        log_every_n_batches: int = 20,
        log_trainer_waits: bool = False,
        log_pending_batch_every_n_polls: int = 50,
    ) -> None:
        super().__init__()

        if config is None:
            config = BatchFlowTorchConfig(
                coordinator_address=coordinator_address,
                dataset_id=dataset_id,
                max_batches=max_batches,
                job_id=job_id,
                job_index=job_index,
                lookahead_batches=lookahead_batches,
                request_poll_interval_seconds=request_poll_interval_seconds,
                fetch_timeout_seconds=fetch_timeout_seconds,
                coordinator_timeout_seconds=coordinator_timeout_seconds,
                max_ready_batches=max_ready_batches,
                parallel_fetch_workers=parallel_fetch_workers,
                ready_queue_timeout_seconds=ready_queue_timeout_seconds,
                pin_memory=pin_memory,
                finish_job_on_close=finish_job_on_close,
                log_interval_seconds=log_interval_seconds,
                log_every_n_batches=log_every_n_batches,
                log_trainer_waits=log_trainer_waits,
                log_pending_batch_every_n_polls=(
                    log_pending_batch_every_n_polls
                ),
            )

        config.validate()
        self.config = config

    def __iter__(self) -> BatchFlowIterator:
        worker_info = get_worker_info()

        if worker_info is not None:
            raise RuntimeError(
                "BatchFlowIterableDataset requires "
                "DataLoader(..., batch_size=None, num_workers=0). "
                "PyTorch DataLoader worker sharding is not currently supported."
            )

        return BatchFlowIterator(self.config)