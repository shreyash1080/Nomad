"""
MobileAirLLM AutoModel
One-liner API to load and run any supported model on mobile.

Usage:
    from mobilellm import AutoModel
    model = AutoModel.from_pretrained("meta-llama/Llama-2-30b-hf")
    for chunk in model.stream("Tell me a story"):
        print(chunk, end="", flush=True)
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Iterator, Union

from .memory_manager import MemoryManager, detect_device_profile, DeviceProfile
from .splitter import ModelSplitter
from .engine import MobileInferenceEngine

logger = logging.getLogger(__name__)

# Default shard cache (same as HF cache convention)
DEFAULT_SHARD_CACHE = os.path.expanduser("~/.cache/mobilellm/shards")


class AutoModel:
    """
    Drop-in, AirLLM-style API for MobileAirLLM.

    Example:
        model = AutoModel.from_pretrained(
            "meta-llama/Llama-2-30b-hf",
            compression="4bit",
            max_ram_gb=6.0,
        )
        print(model.generate("What is quantum computing?", max_new_tokens=200))
    """

    def __init__(self, engine: MobileInferenceEngine, shard_dir: str):
        self._engine = engine
        self._shard_dir = shard_dir

    # ── Constructor ───────────────────────────────────────────────────────────
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,                        # HF repo ID or local path
        shard_dir: Optional[str] = None,        # Where to store/find layer shards
        compression: Optional[str] = None,      # "4bit" | "8bit" | "none" (auto if None)
        max_ram_gb: Optional[float] = None,     # Override RAM budget
        device_profile: Optional[str] = None,  # "low"|"medium"|"standard"|"high"|"ultra"
        prefetch: Optional[int] = None,         # Layers to prefetch (auto if None)
        max_kv_tokens: Optional[int] = None,    # KV cache size (auto if None)
        thermal_pause_ms: int = 50,             # Thermal pacing between layers
        delete_original: bool = False,          # Delete HF model after sharding
        hf_token: Optional[str] = None,        # HuggingFace API token
        force_resplit: bool = False,            # Re-shard even if shards exist
    ) -> "AutoModel":

        # ── 1. Resolve device profile ─────────────────────────────────────
        from .memory_manager import DEVICE_PROFILES
        if device_profile:
            profile = DEVICE_PROFILES[device_profile]
        else:
            profile = detect_device_profile()

        if max_ram_gb:
            profile.model_ram_budget_gb = max_ram_gb

        compression = compression or profile.default_quantization
        prefetch     = prefetch     or profile.prefetch_layers
        max_kv_tokens = max_kv_tokens or profile.max_kv_tokens

        logger.info(f"[AutoModel] Profile: {profile.name}")
        logger.info(f"[AutoModel] Compression: {compression} | Prefetch: {prefetch} | MaxKV: {max_kv_tokens}")

        # ── 2. Resolve shard directory ────────────────────────────────────
        safe_name = model_path.replace("/", "--").replace("\\", "--")
        shard_dir = shard_dir or os.path.join(DEFAULT_SHARD_CACHE, safe_name, compression)
        os.makedirs(shard_dir, exist_ok=True)

        # ── 3. Split model into shards (skip if already done) ─────────────
        splitter = ModelSplitter(
            model_path=model_path,
            shard_dir=shard_dir,
            compression=compression,
            delete_original=delete_original,
            hf_token=hf_token,
        )
        splitter.split(force=force_resplit)

        # ── 4. Load tokenizer ─────────────────────────────────────────────
        tokenizer = cls._load_tokenizer(model_path, hf_token)

        # ── 5. Load model config ──────────────────────────────────────────
        model_config = cls._load_config(model_path, hf_token)

        # ── 6. Build engine ───────────────────────────────────────────────
        mm = MemoryManager(profile)
        mm.start_monitor()

        engine = MobileInferenceEngine(
            shard_dir=shard_dir,
            tokenizer=tokenizer,
            model_config=model_config,
            memory_manager=mm,
            prefetch=prefetch,
            max_kv_tokens=max_kv_tokens,
            thermal_pause_ms=thermal_pause_ms,
        )

        return cls(engine, shard_dir)

    # ── Inference API ─────────────────────────────────────────────────────────
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream tokens as they are generated."""
        yield from self._engine.stream_generate(prompt, **kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate and return the complete response."""
        return self._engine.generate(prompt, **kwargs)

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def reset(self):
        """Clear conversation KV cache."""
        self._engine.reset()

    @property
    def tokenizer(self):
        return self._engine.tokenizer

    def stats(self) -> dict:
        return self._engine.stats()

    # ── Internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _load_tokenizer(model_path: str, hf_token: Optional[str] = None):
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(model_path, token=hf_token, use_fast=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(model_path, token=hf_token, use_fast=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    @staticmethod
    def _load_config(model_path: str, hf_token: Optional[str] = None) -> dict:
        # Try local config.json first
        local_cfg = os.path.join(model_path, "config.json")
        if os.path.isfile(local_cfg):
            with open(local_cfg) as f:
                return json.load(f)
        # HF Hub
        try:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(model_path, "config.json", token=hf_token)
            with open(cfg_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config.json: {e}. Using defaults.")
            return {}
