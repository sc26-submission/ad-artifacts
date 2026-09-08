from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from batchflow.clients.coordinator_client import CoordinatorGrpcClient
from batchflow.clients.redis_client import RedisFetchClient
from batchflow.clients.worker_client import WorkerFetchClient
from batchflow.integrations.pytorch.config import (
    BatchFlowTorchConfig,
    TrainerRuntimeMetricsBuffer,
)
from batchflow.integrations.pytorch.decoding import decode_payload, pin_memory_batch
from batchflow.proto import batchflow_pb2


LOGGER = logging.getLogger(__name__)

_QUEUE_WAIT_SECONDS = 0.2
_THREAD_JOIN_TIMEOUT_SECONDS = 5.0


@dataclass(slots=True)
class BatchItem:
    batch: dict[str, Any]


@dataclass(slots=True)
class ErrorItem:
    error: Exception


@dataclass(frozen=True, slots=True)
class EndItem:
    pass


@dataclass(frozen=True, slots=True)
class FetchTask:
    sequence: int
    job_id: str
    batch_id: str
    cache_key: str
    epoch: int
    batch_index: int

    location: str
    fetch_host: str
    fetch_port: int
    fetch_key: str
    payload_format: str
    dataset_format: str

    handle_status: str
    cache_result: str
    client_cache_result: str

    coordinator_wait_total_time_sec: float
    coordinator_rpc_time_sec: float
    coordinator_sleep_time_sec: float
    coordinator_pending_polls: int
    coordinator_miss_polls: int
    coordinator_in_flight_polls: int


@dataclass(slots=True)
class FetchedTaskResult:
    task: FetchTask
    batch: dict[str, Any]
    payload_bytes: int
    fetch_time_sec: float
    decode_time_sec: float
    pin_time_sec: float


class MultiThreadBatchFlowPrefetcher:
    def __init__(
        self,
        config: BatchFlowTorchConfig,
        *,
        runtime_metrics: TrainerRuntimeMetricsBuffer | None = None,
    ) -> None:
        config.validate()

        self.config = config
        self.runtime_metrics = runtime_metrics

        self._job_id = config.job_id or f"torch-job-{uuid.uuid4().hex[:8]}"

        queue_size = max(1, config.max_ready_batches)

        self._ready_queue: queue.Queue[BatchItem | ErrorItem | EndItem] = queue.Queue(
            maxsize=queue_size
        )
        self._task_queue: queue.Queue[FetchTask | None] = queue.Queue(maxsize=queue_size)
        self._fetched_queue: queue.Queue[FetchedTaskResult] = queue.Queue(maxsize=queue_size)

        self._stop_event = threading.Event()
        self._end_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()

        self._coordinator_client: CoordinatorGrpcClient | None = None

        self._coordinator_done = False
        self._end_sent = False
        self._completed_normally = False
        self._shutdown = False
        self._job_closed = False

        self._scheduled_batches = 0
        self._published_batches = 0
        self._last_status_log_time = time.monotonic()

        self._coordinator_thread = threading.Thread(
            target=self._coordinator_loop,
            name="batchflow-pytorch-coordinator",
            daemon=True,
        )
        self._publish_thread = threading.Thread(
            target=self._ordered_publish_loop,
            name="batchflow-pytorch-publish",
            daemon=True,
        )
        self._fetch_threads = [
            threading.Thread(
                target=self._fetch_worker_loop,
                name=f"batchflow-pytorch-fetch-{index}",
                daemon=True,
            )
            for index in range(max(1, config.parallel_fetch_workers))
        ]

    @property
    def job_id(self) -> str:
        return self._job_id

    def start(self) -> None:
        LOGGER.debug(
            f"BatchFlow prefetch starting | job={self._job_id} | "
            f"dataset={self.config.dataset_id} | workers={len(self._fetch_threads)} | "
            f"buffer={self._ready_queue.maxsize} | lookahead={self.config.lookahead_batches}"
        )

        for thread in self._fetch_threads:
            thread.start()

        self._publish_thread.start()
        self._coordinator_thread.start()

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True

        self.stop()

        if self.config.finish_job_on_close:
            self._finish_job()

        self._close_coordinator_client()

        LOGGER.debug(
            f"BatchFlow prefetch stopped | job={self._job_id} | "
            f"scheduled={self._scheduled_batches} | published={self._published_batches}"
        )

    def stop(self) -> None:
        self._stop_event.set()

        self._join_thread(self._coordinator_thread)

        for thread in self._fetch_threads:
            self._join_thread(thread)

        self._join_thread(self._publish_thread)

    def close_job(self) -> None:
        self._finish_job()
        self._close_coordinator_client()

    def get_item(self, timeout_seconds: float) -> BatchItem | ErrorItem | EndItem:
        item = self._ready_queue.get(timeout=timeout_seconds)

        # Only reaching EndItem through the consumer means the iterator actually
        # completed normally. Finishing coordinator scheduling is not enough.
        if isinstance(item, EndItem):
            self._completed_normally = True

        return item

    def qsize(self) -> int:
        return self._ready_queue.qsize()

    def maxsize(self) -> int:
        return self._ready_queue.maxsize

    def update_runtime_metrics(
        self,
        *,
        data_bottleneck_percent: float,
        avg_data_time_sec: float,
        avg_compute_time_sec: float,
        avg_coordinator_wait_total_time_sec: float,
        avg_coordinator_pending_polls: float,
    ) -> None:
        if self.runtime_metrics is None:
            return

        self.runtime_metrics.update(
            data_bottleneck_percent=data_bottleneck_percent,
            avg_data_time_sec=avg_data_time_sec,
            avg_compute_time_sec=avg_compute_time_sec,
            avg_coordinator_wait_total_time_sec=avg_coordinator_wait_total_time_sec,
            avg_coordinator_pending_polls=avg_coordinator_pending_polls,
        )

    def _coordinator_loop(self) -> None:
        client = CoordinatorGrpcClient(self.config.coordinator_address)
        self._coordinator_client = client

        try:
            client.connect()

            response = client.start_job(
                job_id=self._job_id,
                dataset_id=self.config.dataset_id,
                lookahead_batches=self.config.lookahead_batches,
                metadata={"job_index": str(self.config.job_index)},
            )

            self._job_id = response.job_id

            LOGGER.debug(
                f"BatchFlow job registered | job={self._job_id} | "
                f"dataset={self.config.dataset_id}"
            )

            while not self._stop_event.is_set():
                if self._scheduled_batches >= self.config.max_batches:
                    self._mark_coordinator_done()
                    return

                task = self._get_next_fetch_task(client)

                if task is None:
                    self._mark_coordinator_done()
                    return

                if not self._put_task(task):
                    return

                self._scheduled_batches += 1

        except Exception as exc:
            if not self._stop_event.is_set():
                LOGGER.exception(f"BatchFlow coordinator failed | job={self._job_id}")
                self._signal_error(exc)

        finally:
            for _ in self._fetch_threads:
                self._put_task_sentinel()

    def _get_next_fetch_task(
        self,
        client: CoordinatorGrpcClient,
    ) -> FetchTask | None:
        wait_start = time.perf_counter()

        rpc_time = 0.0
        sleep_time = 0.0
        pending_polls = 0
        miss_polls = 0
        in_flight_polls = 0

        while not self._stop_event.is_set():
            rpc_start = time.perf_counter()

            response = client.get_next_batch(
                self._job_id,
                runtime_feedback=self._runtime_metrics_snapshot(),
                timeout_seconds=self.config.coordinator_timeout_seconds,
            )

            rpc_time += time.perf_counter() - rpc_start

            if response.done:
                return None

            handle = response.batch_handle
            metadata = _metadata_to_dict(handle.metadata)
            cache_result = metadata.get("cache_result", "")

            if handle.status == batchflow_pb2.BATCH_HANDLE_STATUS_PENDING:
                pending_polls += 1

                if cache_result == "miss":
                    miss_polls += 1
                elif cache_result == "in_flight":
                    in_flight_polls += 1

                self._maybe_log_pending_batch(
                    pending_polls=pending_polls,
                    cache_result=cache_result,
                )

                sleep_start = time.perf_counter()
                time.sleep(self.config.request_poll_interval_seconds)
                sleep_time += time.perf_counter() - sleep_start
                continue

            self._validate_ready_handle(handle)

            batch_index = int(response.batch_index) if response.has_batch_index else -1
            wait_time = time.perf_counter() - wait_start

            return FetchTask(
                sequence=self._scheduled_batches,
                job_id=self._job_id,
                batch_id=handle.batch_id,
                cache_key=handle.cache_key,
                epoch=int(response.epoch),
                batch_index=batch_index,
                location=handle.location,
                fetch_host=handle.fetch_host,
                fetch_port=int(handle.fetch_port),
                fetch_key=handle.fetch_key,
                payload_format=handle.payload_format,
                dataset_format=handle.dataset_format,
                handle_status=_handle_status_name(handle.status),
                cache_result=cache_result,
                client_cache_result="hot" if pending_polls == 0 else "miss",
                coordinator_wait_total_time_sec=wait_time,
                coordinator_rpc_time_sec=rpc_time,
                coordinator_sleep_time_sec=sleep_time,
                coordinator_pending_polls=pending_polls,
                coordinator_miss_polls=miss_polls,
                coordinator_in_flight_polls=in_flight_polls,
            )

        return None

    def _fetch_worker_loop(self) -> None:
        worker_client = WorkerFetchClient()
        redis_client = RedisFetchClient(timeout_seconds=self.config.fetch_timeout_seconds)

        try:
            while not self._stop_event.is_set():
                try:
                    task = self._task_queue.get(timeout=_QUEUE_WAIT_SECONDS)
                except queue.Empty:
                    self._maybe_send_end()
                    continue

                try:
                    if task is None:
                        self._maybe_send_end()
                        return

                    result = self._fetch_and_decode(worker_client, redis_client, task)

                    if not self._put_fetched_result(result):
                        return

                except Exception as exc:
                    if not self._stop_event.is_set():
                        LOGGER.exception(f"BatchFlow fetch failed | job={self._job_id}")
                        self._signal_error(exc)
                        return

                finally:
                    self._task_queue.task_done()
                    self._maybe_send_end()

        finally:
            redis_client.close()
            worker_client.close()

    def _fetch_and_decode(
        self,
        worker_client: WorkerFetchClient,
        redis_client: RedisFetchClient,
        task: FetchTask,
    ) -> FetchedTaskResult:
        fetch_start = time.perf_counter()
        payload = self._fetch_payload(worker_client, redis_client, task)
        fetch_time = time.perf_counter() - fetch_start

        decode_start = time.perf_counter()
        batch = decode_payload(payload, payload_format=task.payload_format)
        decode_time = time.perf_counter() - decode_start

        pin_time = 0.0

        if self.config.pin_memory:
            pin_start = time.perf_counter()
            batch = pin_memory_batch(batch)
            pin_time = time.perf_counter() - pin_start

        return FetchedTaskResult(
            task=task,
            batch=batch,
            payload_bytes=len(payload),
            fetch_time_sec=fetch_time,
            decode_time_sec=decode_time,
            pin_time_sec=pin_time,
        )

    def _fetch_payload(
        self,
        worker_client: WorkerFetchClient,
        redis_client: RedisFetchClient,
        task: FetchTask,
    ) -> bytes:
        if task.location.startswith(("redis://", "rediss://")):
            return redis_client.fetch_batch(
                location=task.location,
                key=task.fetch_key,
            )

        return worker_client.fetch_batch(
            host=task.fetch_host,
            port=task.fetch_port,
            key=task.fetch_key,
            timeout_seconds=self.config.fetch_timeout_seconds,
        )

    def _ordered_publish_loop(self) -> None:
        client = CoordinatorGrpcClient(self.config.coordinator_address)
        next_sequence = 0
        buffered: dict[int, FetchedTaskResult] = {}

        try:
            client.connect()

            while not self._stop_event.is_set():
                try:
                    result = self._fetched_queue.get(timeout=_QUEUE_WAIT_SECONDS)
                except queue.Empty:
                    self._maybe_send_end()
                    continue

                try:
                    buffered[result.task.sequence] = result

                    while next_sequence in buffered:
                        ready = buffered.pop(next_sequence)
                        self._acknowledge_and_publish(client, ready)
                        next_sequence += 1

                finally:
                    self._fetched_queue.task_done()

                self._maybe_send_end()

        except Exception as exc:
            if not self._stop_event.is_set():
                LOGGER.exception(f"BatchFlow publisher failed | job={self._job_id}")
                self._signal_error(exc)

        finally:
            client.close()

    def _acknowledge_and_publish(
        self,
        client: CoordinatorGrpcClient,
        result: FetchedTaskResult,
    ) -> None:
        task = result.task

        client.acknowledge_batch(
            job_id=task.job_id,
            batch_id=task.batch_id,
            epoch=task.epoch,
            batch_index=task.batch_index,
            timeout_seconds=self.config.coordinator_timeout_seconds,
        )

        self._decorate_batch(result)

        if not self._put_ready_item(BatchItem(batch=result.batch)):
            return

        self._published_batches += 1
        self._maybe_log_status(result)

    def _decorate_batch(self, result: FetchedTaskResult) -> None:
        task = result.task

        result.batch.update(
            {
                "job_id": task.job_id,
                "dataset_id": self.config.dataset_id,
                "batch_id": task.batch_id,
                "cache_key": task.cache_key,
                "epoch": task.epoch,
                "batch_index": task.batch_index,
                "handle_status": task.handle_status,
                "dataset_format": task.dataset_format,
                "payload_format": task.payload_format,
                "fetch_location": task.location,
                "cache_result": task.cache_result,
                "client_cache_result": task.client_cache_result,
                "pending_polls_before_batch": task.coordinator_pending_polls,
                "miss_polls_before_batch": task.coordinator_miss_polls,
                "in_flight_polls_before_batch": task.coordinator_in_flight_polls,
                "coordinator_wait_total_time_sec": task.coordinator_wait_total_time_sec,
                "coordinator_rpc_time_sec": task.coordinator_rpc_time_sec,
                "coordinator_sleep_time_sec": task.coordinator_sleep_time_sec,
                "fetch_time_sec": result.fetch_time_sec,
                "trainer_decode_time_sec": result.decode_time_sec,
                "trainer_pin_time_sec": result.pin_time_sec,
                "payload_bytes": result.payload_bytes,
                "prefetch_queue_size_before_put": self._ready_queue.qsize(),
            }
        )

    def _validate_ready_handle(self, handle: Any) -> None:
        if handle.status == batchflow_pb2.BATCH_HANDLE_STATUS_FAILED:
            raise RuntimeError(f"Coordinator returned failed batch handle: {handle}")

        if handle.status != batchflow_pb2.BATCH_HANDLE_STATUS_READY:
            raise RuntimeError(f"Unexpected BatchFlow handle status: {handle.status}")

        if not handle.fetch_key:
            raise RuntimeError(f"Batch handle is missing fetch_key: {handle}")

        if not handle.payload_format:
            raise RuntimeError(f"Batch handle is missing payload_format: {handle}")

        if not handle.dataset_format:
            raise RuntimeError(f"Batch handle is missing dataset_format: {handle}")

        if handle.location.startswith(("redis://", "rediss://")):
            return

        if handle.location.startswith("grpc://") or not handle.location:
            if handle.fetch_host and handle.fetch_port > 0:
                return

            raise RuntimeError(
                f"Batch handle is missing worker fetch information: "
                f"host={handle.fetch_host!r}, port={handle.fetch_port!r}, "
                f"key={handle.fetch_key!r}"
            )

        raise RuntimeError(f"Unsupported batch location: {handle.location!r}")

    def _runtime_metrics_snapshot(self) -> Any:
        if self.runtime_metrics is None:
            return None

        return self.runtime_metrics.snapshot()

    def _maybe_log_pending_batch(
        self,
        *,
        pending_polls: int,
        cache_result: str,
    ) -> None:
        every = self.config.log_pending_batch_every_n_polls

        if every <= 0 or pending_polls % every != 0:
            return

        LOGGER.debug(
            f"BatchFlow batch pending | job={self._job_id} | "
            f"polls={pending_polls} | cache={cache_result or '-'}"
        )

    def _maybe_log_status(self, result: FetchedTaskResult) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return

        now = time.monotonic()

        batch_triggered = (
            self.config.log_every_n_batches > 0
            and self._published_batches % self.config.log_every_n_batches == 0
        )
        interval_triggered = (
            self.config.log_interval_seconds > 0
            and now - self._last_status_log_time >= self.config.log_interval_seconds
        )

        if not batch_triggered and not interval_triggered:
            return

        LOGGER.debug(
            f"BatchFlow prefetch status | job={self._job_id} | "
            f"published={self._published_batches}/{self.config.max_batches} | "
            f"ready_for_trainer={self._ready_queue.qsize()}/{self._ready_queue.maxsize} | "
            f"queued_for_fetch={self._task_queue.qsize()} | "
            f"fetched_waiting_publish={self._fetched_queue.qsize()} | "
            f"last_fetch={result.fetch_time_sec:.4f}s | "
            f"last_decode={result.decode_time_sec:.4f}s"
        )

        self._last_status_log_time = now

    def _mark_coordinator_done(self) -> None:
        self._coordinator_done = True
        self._maybe_send_end()

    def _maybe_send_end(self) -> None:
        with self._end_lock:
            if self._end_sent or not self._coordinator_done:
                return

            if not self._task_queue.empty():
                return

            if not self._fetched_queue.empty():
                return

            if self._published_batches < self._scheduled_batches:
                return

            self._end_sent = True

        self._put_ready_item(EndItem())

    def _signal_error(self, exc: Exception) -> None:
        self._put_ready_item(ErrorItem(error=exc))
        self._stop_event.set()

    def _put_task(self, task: FetchTask) -> bool:
        while not self._stop_event.is_set():
            try:
                self._task_queue.put(task, timeout=_QUEUE_WAIT_SECONDS)
                return True
            except queue.Full:
                continue

        return False

    def _put_task_sentinel(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._task_queue.put(None, timeout=_QUEUE_WAIT_SECONDS)
                return
            except queue.Full:
                continue

    def _put_fetched_result(self, result: FetchedTaskResult) -> bool:
        while not self._stop_event.is_set():
            try:
                self._fetched_queue.put(result, timeout=_QUEUE_WAIT_SECONDS)
                return True
            except queue.Full:
                continue

        return False

    def _put_ready_item(self, item: BatchItem | ErrorItem | EndItem) -> bool:
        while not self._stop_event.is_set():
            try:
                self._ready_queue.put(item, timeout=_QUEUE_WAIT_SECONDS)
                return True
            except queue.Full:
                continue

        return False

    def _finish_job(self) -> None:
        if self._job_closed:
            return

        self._job_closed = True
        client = self._coordinator_client

        if client is None:
            return

        status = (
            batchflow_pb2.JOB_STATUS_COMPLETED
            if self._completed_normally
            else batchflow_pb2.JOB_STATUS_CANCELLED
        )
        reason = (
            "PyTorch iterator completed"
            if self._completed_normally
            else "PyTorch iterator closed"
        )

        try:
            client.finish_job(
                job_id=self._job_id,
                reason=reason,
                status=status,
            )
        except Exception:
            LOGGER.exception(f"Failed to finish BatchFlow job | job={self._job_id}")

    def _close_coordinator_client(self) -> None:
        if self._coordinator_client is None:
            return

        self._coordinator_client.close()
        self._coordinator_client = None

    def _join_thread(self, thread: threading.Thread) -> None:
        if not thread.is_alive():
            return

        thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)

        if thread.is_alive():
            LOGGER.warning(
                f"BatchFlow thread did not stop cleanly | "
                f"job={self._job_id} | thread={thread.name}"
            )


def _metadata_to_dict(items: Any) -> dict[str, str]:
    return {item.key: item.value for item in items}


def _handle_status_name(value: int) -> str:
    mapping = {
        batchflow_pb2.BATCH_HANDLE_STATUS_PENDING: "pending",
        batchflow_pb2.BATCH_HANDLE_STATUS_READY: "ready",
        batchflow_pb2.BATCH_HANDLE_STATUS_FAILED: "failed",
    }
    return mapping.get(value, f"unknown({value})")