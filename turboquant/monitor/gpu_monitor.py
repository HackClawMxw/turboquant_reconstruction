"""
GPU compute and memory load monitor.

Provides real-time monitoring of:
  - GPU utilization (compute throughput %)
  - Memory usage (used / total / free)
  - Per-layer TQ memory breakdown
  - Active request count and KV cache pressure

Supports:
  - Single-node: local GPU monitoring
  - Multi-node: aggregated monitoring across nodes

Usage:
    monitor = GPUMonitor(interval_ms=1000)
    monitor.start()
    # ... inference runs ...
    stats = monitor.snapshot()
    monitor.stop()
"""

import logging
import threading
import time
from typing import Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("turboquant.monitor")


@dataclass
class GPUStats:
    """GPU statistics snapshot."""
    gpu_id: int
    timestamp: float

    # Compute
    gpu_utilization_pct: float = 0.0     # GPU compute utilization %
    memory_utilization_pct: float = 0.0   # Memory controller utilization %

    # Memory
    memory_total_bytes: int = 0
    memory_used_bytes: int = 0
    memory_free_bytes: int = 0
    memory_allocated_bytes: int = 0       # PyTorch allocated
    memory_reserved_bytes: int = 0        # PyTorch reserved

    # TQ-specific
    tq_total_memory_bytes: int = 0        # TQ KV cache memory
    tq_active_requests: int = 0           # Active TQ requests
    tq_compressed_tokens: int = 0         # Total compressed tokens

    # Derived
    @property
    def memory_used_pct(self) -> float:
        if self.memory_total_bytes == 0:
            return 0.0
        return (self.memory_used_bytes / self.memory_total_bytes) * 100

    @property
    def tq_memory_pct(self) -> float:
        if self.memory_total_bytes == 0:
            return 0.0
        return (self.tq_total_memory_bytes / self.memory_total_bytes) * 100

    def __str__(self) -> str:
        return (
            f"GPU {self.gpu_id}: "
            f"Compute={self.gpu_utilization_pct:.1f}% "
            f"Mem={self.memory_used_pct:.1f}% "
            f"(TQ={self.tq_memory_pct:.1f}%, {self.tq_active_requests} reqs, "
            f"{self.tq_compressed_tokens} tokens) "
            f"Alloc={self.memory_allocated_bytes / 1e9:.2f}GB"
        )


class GPUMonitor:
    """
    Single-node GPU monitor.

    Periodically samples GPU statistics and TQ state.
    Can run as a background thread or be polled manually.
    """

    def __init__(
        self,
        gpu_ids: Optional[list[int]] = None,
        interval_ms: int = 1000,
        tq_stats_provider: Optional[Callable[[], dict]] = None,
    ):
        """
        Args:
            gpu_ids: GPU IDs to monitor. None = all available GPUs.
            interval_ms: Sampling interval in milliseconds.
            tq_stats_provider: Callback that returns TQ stats dict.
                Called each sampling period.
        """
        self.interval_ms = interval_ms
        self.tq_stats_provider = tq_stats_provider

        # Determine GPU IDs
        if gpu_ids is not None:
            self.gpu_ids = gpu_ids
        else:
            try:
                import torch
                self.gpu_ids = list(range(torch.cuda.device_count()))
            except Exception:
                self.gpu_ids = []

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_stats: dict[int, GPUStats] = {}
        self._lock = threading.Lock()

    def start(self):
        """Start background monitoring thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="tq-gpu-monitor",
        )
        self._thread.start()
        logger.info(f"[TQ Monitor] Started (interval={self.interval_ms}ms, GPUs={self.gpu_ids})")

    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("[TQ Monitor] Stopped")

    def snapshot(self) -> dict[int, GPUStats]:
        """Get latest GPU statistics."""
        with self._lock:
            return dict(self._latest_stats)

    def sample_once(self) -> dict[int, GPUStats]:
        """Take a single sample of GPU statistics."""
        stats = {}
        tq_stats = {}
        if self.tq_stats_provider is not None:
            try:
                tq_stats = self.tq_stats_provider()
            except Exception as e:
                logger.warning(f"[TQ Monitor] Error getting TQ stats: {e}")

        for gpu_id in self.gpu_ids:
            stats[gpu_id] = self._sample_gpu(gpu_id, tq_stats)

        with self._lock:
            self._latest_stats = stats

        return stats

    def _sample_gpu(self, gpu_id: int, tq_stats: dict) -> GPUStats:
        """Sample statistics for a single GPU."""
        s = GPUStats(gpu_id=gpu_id, timestamp=time.time())

        try:
            import torch

            # GPU utilization via torch.cuda
            props = torch.cuda.get_device_properties(gpu_id)
            s.memory_total_bytes = props.total_memory

            # Memory stats
            s.memory_allocated_bytes = torch.cuda.memory_allocated(gpu_id)
            s.memory_reserved_bytes = torch.cuda.memory_reserved(gpu_id)

            # Use pynvml for utilization if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                s.gpu_utilization_pct = util.gpu
                s.memory_utilization_pct = util.memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                s.memory_used_bytes = mem_info.used
                s.memory_free_bytes = mem_info.free
                pynvml.nvmlShutdown()
            except ImportError:
                # Fallback: estimate from PyTorch stats
                s.memory_used_bytes = s.memory_allocated_bytes
                s.memory_free_bytes = s.memory_total_bytes - s.memory_used_bytes
                s.gpu_utilization_pct = -1  # Unknown

        except Exception as e:
            logger.warning(f"[TQ Monitor] Error sampling GPU {gpu_id}: {e}")

        # TQ stats
        s.tq_total_memory_bytes = tq_stats.get("total_memory_bytes", 0)
        s.tq_active_requests = tq_stats.get("active_requests", 0)
        s.tq_compressed_tokens = tq_stats.get("compressed_tokens", 0)

        return s

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self.sample_once()
            except Exception as e:
                logger.warning(f"[TQ Monitor] Sample error: {e}")
            time.sleep(self.interval_ms / 1000.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class MultiNodeMonitor:
    """
    Multi-node GPU monitor (placeholder).

    For multi-node deployments, this would aggregate GPU stats
    from multiple nodes via network communication.

    Implementation options:
    - Shared file system (NFS) for stats exchange
    - Direct network communication (TCP/UDP)
    - Integration with monitoring systems (Prometheus, Grafana)

    TODO: Implement multi-node aggregation.
    """

    def __init__(
        self,
        node_id: str = "node_0",
        nodes: Optional[list[str]] = None,
        port: int = 0,
    ):
        self.node_id = node_id
        self.nodes = nodes or [node_id]
        self.local_monitor = GPUMonitor()

    def start(self):
        """Start monitoring on this node."""
        self.local_monitor.start()

    def stop(self):
        """Stop monitoring on this node."""
        self.local_monitor.stop()

    def get_cluster_stats(self) -> dict[str, dict[int, GPUStats]]:
        """Get stats from all nodes (placeholder)."""
        return {self.node_id: self.local_monitor.snapshot()}
