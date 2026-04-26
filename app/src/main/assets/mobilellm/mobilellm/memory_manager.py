"""
MobileLLM Memory Manager
Tracks RAM usage, enforces budgets, and triggers garbage collection
for mobile-constrained environments.
"""

import gc
import os
import sys
import time
import logging
import threading
import psutil
from typing import Optional, Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Device RAM profiles
# ──────────────────────────────────────────────
@dataclass
class DeviceProfile:
    name: str
    total_ram_gb: float
    # How much RAM we allow the model engine to use
    model_ram_budget_gb: float
    # Reserve for OS + app overhead
    os_reserve_gb: float
    # KV-cache size limit (tokens)
    max_kv_tokens: int
    # How many layers to prefetch ahead
    prefetch_layers: int
    # Quantization to use by default
    default_quantization: str  # 'none' | '8bit' | '4bit'

    @property
    def model_ram_budget_bytes(self) -> int:
        return int(self.model_ram_budget_gb * 1024 ** 3)


DEVICE_PROFILES: Dict[str, DeviceProfile] = {
    "extreme_low":  DeviceProfile("Extreme Low (≤3 GB)",  total_ram_gb=3,  model_ram_budget_gb=1.5, os_reserve_gb=1.5, max_kv_tokens=512,  prefetch_layers=1, default_quantization="4bit"),
    "low":          DeviceProfile("Low (4 GB)",           total_ram_gb=4,  model_ram_budget_gb=2.5, os_reserve_gb=1.5, max_kv_tokens=1024, prefetch_layers=1, default_quantization="4bit"),
    "medium":       DeviceProfile("Medium (6 GB)",        total_ram_gb=6,  model_ram_budget_gb=4.0, os_reserve_gb=2.0, max_kv_tokens=2048, prefetch_layers=2, default_quantization="4bit"),
    "standard":     DeviceProfile("Standard (8 GB)",      total_ram_gb=8,  model_ram_budget_gb=6.0, os_reserve_gb=2.0, max_kv_tokens=4096, prefetch_layers=2, default_quantization="8bit"),
    "high":         DeviceProfile("High (12 GB)",         total_ram_gb=12, model_ram_budget_gb=9.5, os_reserve_gb=2.5, max_kv_tokens=8192, prefetch_layers=3, default_quantization="8bit"),
    "ultra":        DeviceProfile("Ultra (16 GB+)",       total_ram_gb=16, model_ram_budget_gb=13,  os_reserve_gb=3.0, max_kv_tokens=16384,prefetch_layers=4, default_quantization="none"),
}


def detect_device_profile() -> DeviceProfile:
    """Auto-detect the best profile for the current device."""
    available_gb = psutil.virtual_memory().total / (1024 ** 3)
    if available_gb <= 3.5:
        return DEVICE_PROFILES["extreme_low"]
    elif available_gb <= 5:
        return DEVICE_PROFILES["low"]
    elif available_gb <= 7:
        return DEVICE_PROFILES["medium"]
    elif available_gb <= 10:
        return DEVICE_PROFILES["standard"]
    elif available_gb <= 14:
        return DEVICE_PROFILES["high"]
    else:
        return DEVICE_PROFILES["ultra"]


# ──────────────────────────────────────────────
# Memory Manager
# ──────────────────────────────────────────────
class MemoryManager:
    """
    Central RAM budget manager.
    - Tracks allocations by named slots
    - Enforces hard and soft limits
    - Background monitor thread warns before OOM
    - Provides aggressive GC helpers
    """

    def __init__(self, profile: Optional[DeviceProfile] = None, warn_pct: float = 0.85):
        self.profile = profile or detect_device_profile()
        self.warn_pct = warn_pct          # Warn when RAM usage hits this fraction of budget
        self._lock = threading.Lock()
        self._allocations: Dict[str, int] = {}  # name → bytes
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()

        logger.info(
            f"[MemoryManager] Profile: {self.profile.name} | "
            f"Budget: {self.profile.model_ram_budget_gb:.1f} GB | "
            f"Default quant: {self.profile.default_quantization}"
        )

    # ── Public API ──────────────────────────────

    def register(self, name: str, size_bytes: int) -> bool:
        """Register an allocation. Returns False if it would exceed budget."""
        with self._lock:
            projected = sum(self._allocations.values()) + size_bytes
            if projected > self.profile.model_ram_budget_bytes:
                logger.warning(
                    f"[MemoryManager] Cannot register '{name}' ({_fmt(size_bytes)}): "
                    f"would exceed budget ({_fmt(projected)} > {_fmt(self.profile.model_ram_budget_bytes)})"
                )
                return False
            self._allocations[name] = size_bytes
            logger.debug(f"[MemoryManager] Registered '{name}': {_fmt(size_bytes)} | Total: {_fmt(self.used_bytes)}")
            return True

    def release(self, name: str):
        """Release a named allocation."""
        with self._lock:
            if name in self._allocations:
                freed = self._allocations.pop(name)
                logger.debug(f"[MemoryManager] Released '{name}': {_fmt(freed)}")

    def release_all(self):
        with self._lock:
            self._allocations.clear()
        self.force_gc()

    @property
    def used_bytes(self) -> int:
        return sum(self._allocations.values())

    @property
    def free_bytes(self) -> int:
        return max(0, self.profile.model_ram_budget_bytes - self.used_bytes)

    @property
    def utilization(self) -> float:
        return self.used_bytes / self.profile.model_ram_budget_bytes

    def system_free_bytes(self) -> int:
        return psutil.virtual_memory().available

    def force_gc(self):
        """Aggressive memory reclamation."""
        gc.collect()
        # Attempt to return memory to OS on CPython
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def assert_fits(self, size_bytes: int, label: str = ""):
        """Raise MemoryError if allocation would not fit in budget."""
        if size_bytes > self.free_bytes:
            raise MemoryError(
                f"[MobileLLM] Not enough RAM for {label or 'operation'}: "
                f"need {_fmt(size_bytes)}, have {_fmt(self.free_bytes)} free "
                f"(budget {_fmt(self.profile.model_ram_budget_bytes)})"
            )

    # ── Background Monitor ───────────────────────

    def start_monitor(self, interval: float = 5.0):
        """Start background thread that warns on high RAM usage."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True, name="MemMonitor"
        )
        self._monitor_thread.start()

    def stop_monitor(self):
        self._stop_monitor.set()

    def _monitor_loop(self, interval: float):
        while not self._stop_monitor.is_set():
            sys_avail = self.system_free_bytes()
            util = self.utilization
            if util >= self.warn_pct:
                logger.warning(
                    f"[MemoryManager] ⚠ High usage: {util*100:.0f}% of budget | "
                    f"System free: {_fmt(sys_avail)}"
                )
            if sys_avail < 300 * 1024 * 1024:  # < 300 MB free system-wide
                logger.error("[MemoryManager] 🚨 Critical: system RAM near exhausted! Forcing GC.")
                self.force_gc()
            time.sleep(interval)

    # ── Context Manager ──────────────────────────

    def slot(self, name: str, size_bytes: int):
        """Use as context manager: `with mem.slot('layer_3', sz): ...`"""
        return _MemSlot(self, name, size_bytes)

    def summary(self) -> str:
        lines = [
            f"Device Profile : {self.profile.name}",
            f"RAM Budget     : {self.profile.model_ram_budget_gb:.1f} GB",
            f"Used           : {_fmt(self.used_bytes)} ({self.utilization*100:.1f}%)",
            f"Free (budget)  : {_fmt(self.free_bytes)}",
            f"Free (system)  : {_fmt(self.system_free_bytes())}",
            "Allocations    :",
        ]
        with self._lock:
            for name, sz in sorted(self._allocations.items(), key=lambda x: -x[1]):
                lines.append(f"  {name:<35} {_fmt(sz)}")
        return "\n".join(lines)


class _MemSlot:
    def __init__(self, mgr: MemoryManager, name: str, size_bytes: int):
        self._mgr = mgr
        self._name = name
        self._size = size_bytes

    def __enter__(self):
        self._mgr.register(self._name, self._size)
        return self

    def __exit__(self, *_):
        self._mgr.release(self._name)
        self._mgr.force_gc()


# ── Helpers ──────────────────────────────────────

def _fmt(b: int) -> str:
    if b >= 1024 ** 3:
        return f"{b/1024**3:.2f} GB"
    elif b >= 1024 ** 2:
        return f"{b/1024**2:.1f} MB"
    return f"{b/1024:.0f} KB"
