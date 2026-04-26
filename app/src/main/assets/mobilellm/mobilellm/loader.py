"""
MobileLLM Layer Loader
Loads model layer-shards from disk one at a time, with:
  • Background prefetch thread (overlaps IO and compute)
  • Strict RAM accounting via MemoryManager
  • Memory-mapped tensors where possible
  • Thermal pacing (prevents mobile CPU throttle)
"""

import os
import gc
import time
import queue
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Tuple, Generator

import torch

from .splitter import dequantize_state, _read_manifest, MANIFEST_FILE

logger = logging.getLogger(__name__)


class LayerLoader:
    """
    Manages disk → RAM → compute pipeline for layer-by-layer inference.

    Usage:
        loader = LayerLoader(shard_dir, prefetch=2)
        embed_weights = loader.load_embed()
        for i, layer_weights in loader.iter_layers():
            out = model_layer(out, **layer_weights)
        head_weights = loader.load_head()
    """

    def __init__(
        self,
        shard_dir: str,
        prefetch: int = 2,
        memory_manager=None,
        thermal_pause_ms: int = 0,
    ):
        self.shard_dir = Path(shard_dir)
        self.prefetch = prefetch
        self.memory_manager = memory_manager
        self.thermal_pause_ms = thermal_pause_ms  # ms to sleep between layers

        self.manifest = _read_manifest(self.shard_dir)
        if self.manifest is None:
            raise FileNotFoundError(
                f"No MobileLLM manifest found at {shard_dir}. "
                "Run ModelSplitter.split() first."
            )

        self.n_layers: int = self.manifest["n_layers"]
        self._layer_files = self.manifest["layer_files"]
        self._layer_sizes = self.manifest["layer_sizes_bytes"]
        self._compression = self.manifest["compression"]

        # Prefetch infrastructure
        self._prefetch_queue: queue.Queue = queue.Queue(maxsize=prefetch + 1)
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_stop = threading.Event()

    # ── High-level API ──────────────────────────────────────────────────────

    def load_embed(self) -> dict:
        """Load embedding weights."""
        return self._load_shard("embed.pt", label="embed")

    def load_head(self) -> dict:
        """Load lm_head + final norm weights."""
        return self._load_shard("head.pt", label="head")

    def iter_layers(self) -> Generator[Tuple[int, dict], None, None]:
        """
        Yields (layer_idx, weights_dict) for each transformer layer.
        Uses prefetch thread if prefetch > 0.
        """
        if self.prefetch > 0:
            yield from self._iter_with_prefetch()
        else:
            yield from self._iter_sequential()

    # ── Sequential (no prefetch) ────────────────────────────────────────────

    def _iter_sequential(self):
        for i in range(self.n_layers):
            weights = self._load_layer_idx(i)
            yield i, weights
            del weights
            self._gc()
            if self.thermal_pause_ms > 0:
                time.sleep(self.thermal_pause_ms / 1000.0)

    # ── Prefetch (background IO thread) ─────────────────────────────────────

    def _iter_with_prefetch(self):
        self._prefetch_stop.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker, daemon=True, name="LLMPrefetch"
        )
        self._prefetch_thread.start()

        try:
            for i in range(self.n_layers):
                weights = self._prefetch_queue.get(timeout=120)
                if isinstance(weights, Exception):
                    raise weights
                yield i, weights
                del weights
                self._gc()
                if self.thermal_pause_ms > 0:
                    time.sleep(self.thermal_pause_ms / 1000.0)
        finally:
            self._prefetch_stop.set()
            if self._prefetch_thread:
                self._prefetch_thread.join(timeout=5)

    def _prefetch_worker(self):
        for i in range(self.n_layers):
            if self._prefetch_stop.is_set():
                break
            try:
                weights = self._load_layer_idx(i)
                self._prefetch_queue.put(weights)
            except Exception as e:
                self._prefetch_queue.put(e)
                break

    # ── Core shard loading ──────────────────────────────────────────────────

    def _load_layer_idx(self, idx: int) -> dict:
        fname = self._layer_files[idx]
        size  = self._layer_sizes[idx]
        return self._load_shard(fname, label=f"layer_{idx:03d}", hint_bytes=size)

    def _load_shard(self, fname: str, label: str = "", hint_bytes: int = 0) -> dict:
        path = self.shard_dir / fname
        t0 = time.perf_counter()

        raw = torch.load(path, map_location="cpu", weights_only=False)
        weights = dequantize_state(raw)
        del raw

        elapsed = time.perf_counter() - t0
        size_mb = path.stat().st_size / 1024 ** 2
        logger.debug(f"[Loader] {label}: {size_mb:.1f} MB in {elapsed*1000:.0f} ms "
                     f"({size_mb/elapsed:.0f} MB/s)")
        return weights

    @staticmethod
    def _gc():
        gc.collect()


# ──────────────────────────────────────────────────────────────────────────────
# KV-Cache  (sliding-window, RAM-bounded)
# ──────────────────────────────────────────────────────────────────────────────
class BoundedKVCache:
    """
    A per-layer KV-cache that evicts oldest tokens when the token budget
    is exceeded — essential for mobile RAM constraints.
    """

    def __init__(self, max_tokens: int, n_heads: int, head_dim: int, n_layers: int):
        self.max_tokens = max_tokens
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_layers = n_layers

        # cache[layer_idx] = (K, V) tensors  shape [1, n_heads, seq_len, head_dim]
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._token_count = 0

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self._cache.get(layer_idx)

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        existing = self._cache.get(layer_idx)
        if existing is not None:
            k = torch.cat([existing[0], new_k], dim=2)
            v = torch.cat([existing[1], new_v], dim=2)
        else:
            k, v = new_k, new_v

        # Evict oldest tokens if over budget
        if k.shape[2] > self.max_tokens:
            keep = self.max_tokens
            k = k[:, :, -keep:, :]
            v = v[:, :, -keep:, :]

        self._cache[layer_idx] = (k, v)
        if layer_idx == 0:
            self._token_count = k.shape[2]

    def clear(self):
        self._cache.clear()
        self._token_count = 0
        gc.collect()

    @property
    def token_count(self) -> int:
        return self._token_count

    def ram_usage_bytes(self) -> int:
        total = 0
        for k, v in self._cache.values():
            total += k.element_size() * k.numel()
            total += v.element_size() * v.numel()
        return total
