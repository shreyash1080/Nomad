"""
MobileLLM Model Splitter
Decomposes a HuggingFace model into per-layer shard files on disk.
Each shard = one transformer layer + associated weights.
This enables one-layer-at-a-time loading, which is the core trick
for running 30 B+ models in 8 GB RAM.
"""

import os
import json
import shutil
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Shard manifest  (written alongside the shards so loading is reproducible)
# ──────────────────────────────────────────────────────────────────────────────
MANIFEST_FILE = "mobilellm_manifest.json"

def _write_manifest(shard_dir: Path, info: dict):
    (shard_dir / MANIFEST_FILE).write_text(json.dumps(info, indent=2))

def _read_manifest(shard_dir: Path) -> Optional[dict]:
    p = shard_dir / MANIFEST_FILE
    return json.loads(p.read_text()) if p.exists() else None


# ──────────────────────────────────────────────────────────────────────────────
# Layer-key resolver  (handles LLaMA / Mistral / Phi / Qwen / Falcon naming)
# ──────────────────────────────────────────────────────────────────────────────
LAYER_PATTERNS = [
    # LLaMA / Mistral / Qwen2 / Phi-3
    ("model.layers", "model.layers.{i}."),
    # GPT-NeoX / Falcon
    ("gpt_neox.layers", "gpt_neox.layers.{i}."),
    # Bloom
    ("transformer.h", "transformer.h.{i}."),
    # GPT-2 / Falcon legacy
    ("transformer.layers", "transformer.layers.{i}."),
    # OPT
    ("model.decoder.layers", "model.decoder.layers.{i}."),
    # MPT
    ("transformer.blocks", "transformer.blocks.{i}."),
]

def _detect_layer_prefix(state_dict_keys: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (container_key, layer_fmt_string) or (None, None)."""
    for container, fmt in LAYER_PATTERNS:
        if any(k.startswith(container + ".0.") for k in state_dict_keys):
            return container, fmt
    return None, None


def _layer_keys(state_dict: dict, layer_idx: int, prefix_fmt: str) -> dict:
    """Extract all state-dict entries for layer `layer_idx`."""
    prefix = prefix_fmt.format(i=layer_idx)
    return {k: v for k, v in state_dict.items() if k.startswith(prefix)}


def _non_layer_keys(state_dict: dict, container_key: str) -> dict:
    """Everything that is NOT inside a numbered transformer layer."""
    return {k: v for k, v in state_dict.items() if not k.startswith(container_key + ".")}


# ──────────────────────────────────────────────────────────────────────────────
# Quantization helpers  (block-wise, weights-only — same idea as AirLLM)
# ──────────────────────────────────────────────────────────────────────────────
def _quantize_4bit(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Block-wise 4-bit quantization (NF4-like, integer approximation).
    Returns (quantized_uint8, scales, zeros) where each block of 64 values
    shares one scale+zero pair.
    """
    BLOCK = 64
    orig_shape = tensor.shape
    flat = tensor.float().flatten()
    pad = (BLOCK - flat.numel() % BLOCK) % BLOCK
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, BLOCK)

    mn = blocks.min(dim=1).values
    mx = blocks.max(dim=1).values
    scales = (mx - mn) / 15.0          # 4-bit → 0..15
    scales = scales.clamp(min=1e-8)
    zeros = mn

    q = ((blocks - zeros.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
    # Pack two 4-bit values into one byte
    q_packed = (q[:, 0::2] | (q[:, 1::2] << 4))
    return q_packed, scales.half(), zeros.half(), orig_shape, pad


def _dequantize_4bit(q_packed, scales, zeros, orig_shape, pad) -> torch.Tensor:
    BLOCK = 64
    lo = (q_packed & 0x0F).float()
    hi = ((q_packed >> 4) & 0x0F).float()
    q = torch.stack([lo, hi], dim=2).view(q_packed.shape[0], BLOCK)
    val = q * scales.float().unsqueeze(1) + zeros.float().unsqueeze(1)
    flat = val.flatten()
    if pad:
        flat = flat[:-pad]
    return flat.view(orig_shape)


def _quantize_8bit(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simple absmax int8 quantization per row."""
    orig_shape = tensor.shape
    flat = tensor.float()
    scale = flat.abs().max().clamp(min=1e-8) / 127.0
    q = (flat / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.half(), orig_shape


def _dequantize_8bit(q, scale, orig_shape) -> torch.Tensor:
    return q.float() * scale.float()


def _apply_quantization(state: dict, compression: str) -> dict:
    """Return a new state-dict with weights compressed."""
    if compression == "none":
        return {k: v.half() for k, v in state.items()}   # at least FP16

    out = {}
    for key, tensor in state.items():
        if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16) and tensor.numel() > 64:
            if compression == "4bit":
                q, sc, zr, sh, pad = _quantize_4bit(tensor)
                out[key] = {"type": "4bit", "q": q, "scale": sc, "zero": zr, "shape": list(sh), "pad": pad}
            elif compression == "8bit":
                q, sc, sh = _quantize_8bit(tensor)
                out[key] = {"type": "8bit", "q": q, "scale": sc, "shape": list(sh)}
        else:
            out[key] = tensor.half()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main splitter
# ──────────────────────────────────────────────────────────────────────────────
class ModelSplitter:
    """
    Splits a HuggingFace model into per-layer shards stored as .pt files.

    Directory layout after splitting:
        shard_dir/
            mobilellm_manifest.json   ← metadata
            embed.pt                  ← embedding + non-layer weights
            layer_000.pt              ← transformer layer 0
            layer_001.pt              ← transformer layer 1
            ...
            head.pt                   ← lm_head + final norm
    """

    def __init__(
        self,
        model_path: str,
        shard_dir: str,
        compression: str = "4bit",      # "none" | "8bit" | "4bit"
        delete_original: bool = False,
        hf_token: Optional[str] = None,
    ):
        self.model_path = model_path
        self.shard_dir = Path(shard_dir)
        self.compression = compression
        self.delete_original = delete_original
        self.hf_token = hf_token

    # ── Public ──────────────────────────────────────────────────────────────

    def split(self, force: bool = False) -> dict:
        """
        Perform the split. Returns the manifest dict.
        If already split (manifest exists) skips unless force=True.
        """
        manifest = _read_manifest(self.shard_dir)
        if manifest and not force:
            logger.info(f"[Splitter] Already split at {self.shard_dir}. Loading manifest.")
            return manifest

        self.shard_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Splitter] Loading model from {self.model_path} …")
        state_dict = self._load_state_dict()

        keys = list(state_dict.keys())
        container_key, layer_fmt = _detect_layer_prefix(keys)
        if container_key is None:
            raise ValueError(
                "Cannot detect transformer layer structure. "
                "Supported architectures: LLaMA, Mistral, Qwen, Falcon, Bloom, OPT, MPT."
            )

        # Count layers
        n_layers = 0
        while any(k.startswith(layer_fmt.format(i=n_layers)) for k in keys):
            n_layers += 1
        logger.info(f"[Splitter] Detected {n_layers} layers | container: {container_key}")

        # Separate embed / head / body
        non_layer = _non_layer_keys(state_dict, container_key)
        embed_keys = {k: v for k, v in non_layer.items() if "embed" in k or "wte" in k or "wpe" in k}
        head_keys  = {k: v for k, v in non_layer.items() if "lm_head" in k or "norm" in k or k not in embed_keys}

        # Save embed shard
        logger.info("[Splitter] Saving embed shard …")
        torch.save(_apply_quantization(embed_keys, self.compression), self.shard_dir / "embed.pt")

        # Save per-layer shards
        layer_sizes = []
        for i in range(n_layers):
            layer_state = _layer_keys(state_dict, i, layer_fmt)
            qstate = _apply_quantization(layer_state, self.compression)
            fname = f"layer_{i:03d}.pt"
            torch.save(qstate, self.shard_dir / fname)
            sz = (self.shard_dir / fname).stat().st_size
            layer_sizes.append(sz)
            logger.info(f"[Splitter]   Layer {i:3d}/{n_layers} saved ({sz/1024**2:.1f} MB)")

        # Save head shard
        logger.info("[Splitter] Saving head shard …")
        torch.save(_apply_quantization(head_keys, self.compression), self.shard_dir / "head.pt")

        # Write manifest
        manifest = {
            "version": "1.0",
            "model_path": str(self.model_path),
            "n_layers": n_layers,
            "container_key": container_key,
            "layer_fmt": layer_fmt,
            "compression": self.compression,
            "layer_files": [f"layer_{i:03d}.pt" for i in range(n_layers)],
            "layer_sizes_bytes": layer_sizes,
            "total_shard_bytes": sum(layer_sizes) + (self.shard_dir / "embed.pt").stat().st_size
        }
        _write_manifest(self.shard_dir, manifest)
        logger.info(f"[Splitter] ✓ Split complete. Total shard size: {sum(layer_sizes)/1024**3:.2f} GB")

        if self.delete_original:
            logger.warning(f"[Splitter] Deleting original model at {self.model_path} …")
            shutil.rmtree(self.model_path, ignore_errors=True)

        return manifest

    # ── Internal ────────────────────────────────────────────────────────────

    def _load_state_dict(self) -> dict:
        """Load state dict from HF repo or local path, CPU-only."""
        # Local path
        if os.path.isdir(self.model_path):
            return self._load_from_local(self.model_path)

        # HuggingFace hub download
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id=self.model_path,
            token=self.hf_token,
            ignore_patterns=["*.bin.index.json"],
        )
        return self._load_from_local(local)

    @staticmethod
    def _load_from_local(path: str) -> dict:
        import glob
        state = {}
        # safetensors preferred
        sf_files = glob.glob(os.path.join(path, "*.safetensors"))
        if sf_files:
            from safetensors.torch import load_file
            for f in sorted(sf_files):
                logger.info(f"  Loading {os.path.basename(f)} …")
                state.update(load_file(f, device="cpu"))
            return state
        # fallback: pytorch bin
        bin_files = glob.glob(os.path.join(path, "*.bin"))
        if bin_files:
            for f in sorted(bin_files):
                logger.info(f"  Loading {os.path.basename(f)} …")
                state.update(torch.load(f, map_location="cpu", weights_only=True))
            return state
        raise FileNotFoundError(f"No .safetensors or .bin files found in {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Dequantize helper  (used by loader)
# ──────────────────────────────────────────────────────────────────────────────
def dequantize_state(state: dict) -> dict:
    """Convert a quantized shard back to FP32 tensors for computation."""
    out = {}
    for key, val in state.items():
        if isinstance(val, dict):
            t = val["type"]
            if t == "4bit":
                out[key] = _dequantize_4bit(val["q"], val["scale"], val["zero"], tuple(val["shape"]), val["pad"])
            elif t == "8bit":
                out[key] = _dequantize_8bit(val["q"], val["scale"], tuple(val["shape"]))
        elif isinstance(val, torch.Tensor):
            out[key] = val.float()
        else:
            out[key] = val
    return out
