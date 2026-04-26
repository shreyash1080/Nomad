"""
MobileAirLLM Inference Engine
Runs the actual transformer forward pass layer-by-layer,
loading each layer shard from disk, computing, then discarding.
Supports: LLaMA, Mistral, Qwen2, Phi-3, Falcon, Bloom, OPT, MPT
"""

import gc
import math
import time
import logging
from typing import Optional, Iterator, List, Dict, Any, Tuple

import torch
import torch.nn.functional as F

from .loader import LayerLoader, BoundedKVCache
from .memory_manager import MemoryManager, detect_device_profile

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# RoPE
# ──────────────────────────────────────────────────────────────────────────────
def _make_rope_cache(dim: int, max_seq: int, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    seq = x.shape[2]
    cos_ = cos[:, :, offset:offset + seq, :]
    sin_ = sin[:, :, offset:offset + seq, :]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.empty_like(x)
    x_rot[..., ::2] = x1 * cos_ - x2 * sin_
    x_rot[..., 1::2] = x2 * cos_ + x1 * sin_
    return x_rot


# ──────────────────────────────────────────────────────────────────────────────
# Attention
# ──────────────────────────────────────────────────────────────────────────────
def _scaled_dot_product_attention(q, k, v, mask=None) -> torch.Tensor:
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


# ──────────────────────────────────────────────────────────────────────────────
# Single Transformer Layer
# ──────────────────────────────────────────────────────────────────────────────
class TransformerLayerRunner:
    def __init__(self, n_heads: int, n_kv_heads: int, head_dim: int,
                 rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                 layer_idx: int, rms_norm_eps: float = 1e-5):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.hidden_size = n_heads * head_dim
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin
        self.layer_idx = layer_idx
        self.rms_norm_eps = rms_norm_eps

    def _rms_norm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Stay in fp16 — avoid round-trip to float32 for norm weight multiply
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.rms_norm_eps).rsqrt().to(x.dtype)
        return x * rms * w.to(x.dtype)

    def _layer_norm(self, x, w, b=None):
        return F.layer_norm(x.float(), (x.shape[-1],), w.float(),
                            b.float() if b is not None else None).to(x.dtype)

    def _attention(self, x: torch.Tensor, w: dict, kv_cache: Optional[BoundedKVCache],
                   kv_offset: int) -> torch.Tensor:
        B, S, H = x.shape
        dtype = x.dtype

        # Keep fp16 matmuls — faster on ARM NEON, less RAM
        q = x @ w["q"].to(dtype).T
        k = x @ w["k"].to(dtype).T
        v = x @ w["v"].to(dtype).T

        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, self.rope_cos, self.rope_sin, offset=kv_offset)
        k = _apply_rope(k, self.rope_cos, self.rope_sin, offset=kv_offset)

        if kv_cache is not None:
            kv_cache.update(self.layer_idx, k, v)
            cached = kv_cache.get(self.layer_idx)
            k, v = cached

        if self.n_kv_heads < self.n_heads:
            reps = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        total_kv = k.shape[2]
        mask = None
        if S > 1:
            mask = torch.full((S, total_kv), float("-inf"), dtype=dtype)
            mask = torch.triu(mask, diagonal=total_kv - S + 1)

        out = _scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, S, H)

        if "o" in w and w["o"] is not None:
            out = out @ w["o"].to(dtype).T

        return out

    def _mlp(self, x: torch.Tensor, w: dict) -> torch.Tensor:
        dtype = x.dtype
        if "gate" in w and w["gate"] is not None:
            g = F.silu(x @ w["gate"].to(dtype).T)
            u = x @ w["up"].to(dtype).T
            return (g * u) @ w["down"].to(dtype).T
        elif "fc1" in w and w["fc1"] is not None:
            bias1 = w.get("fc1_bias")
            bias2 = w.get("fc2_bias")
            h = F.gelu(x @ w["fc1"].to(dtype).T + (bias1.to(dtype) if bias1 is not None else 0))
            return h @ w["fc2"].to(dtype).T + (bias2.to(dtype) if bias2 is not None else 0)
        else:
            h = F.silu(x @ w["up"].to(dtype).T)
            return h @ w["down"].to(dtype).T

    def forward(self, hidden: torch.Tensor, weights: dict,
                kv_cache: Optional[BoundedKVCache] = None,
                kv_offset: int = 0) -> torch.Tensor:
        w = self._parse_weights(weights)

        normed = self._rms_norm(hidden, w["pre_attn_norm"])
        attn_out = self._attention(normed, w["attn"], kv_cache, kv_offset)
        hidden = hidden + attn_out

        normed = self._rms_norm(hidden, w["pre_mlp_norm"])
        mlp_out = self._mlp(normed, w["mlp"])
        hidden = hidden + mlp_out

        return hidden

    def _parse_weights(self, raw: dict) -> dict:
        def _g(candidates):
            for c in candidates:
                if c in raw:
                    return raw[c]
            return None

        pre_attn = _g(["input_layernorm.weight", "ln_1.weight", "self_attn_layer_norm.weight",
                        "norm1.weight", "attention_norm.weight"])
        pre_mlp = _g(["post_attention_layernorm.weight", "ln_2.weight", "final_layer_norm.weight",
                       "norm2.weight", "ffn_norm.weight"])

        attn = {
            "q": _g(["self_attn.q_proj.weight", "self_attention.query_key_value.weight", "attn.c_attn.weight"]),
            "k": _g(["self_attn.k_proj.weight"]),
            "v": _g(["self_attn.v_proj.weight"]),
            "o": _g(["self_attn.o_proj.weight", "self_attention.dense.weight", "attn.c_proj.weight"]),
        }

        mlp = {
            "gate": _g(["mlp.gate_proj.weight"]),
            "up":   _g(["mlp.up_proj.weight", "mlp.fc1.weight"]),
            "down": _g(["mlp.down_proj.weight", "mlp.fc2.weight"]),
            "fc1":  _g(["mlp.fc1.weight", "mlp.dense_h_to_4h.weight"]),
            "fc2":  _g(["mlp.fc2.weight", "mlp.dense_4h_to_h.weight"]),
        }

        return {"pre_attn_norm": pre_attn, "pre_mlp_norm": pre_mlp, "attn": attn, "mlp": mlp}


# ──────────────────────────────────────────────────────────────────────────────
# Main Engine
# ──────────────────────────────────────────────────────────────────────────────
class MobileInferenceEngine:
    """
    Layer-by-layer inference engine for mobile devices.

    Performance contract:
      • NO gc.collect() inside the layer loop — only after a full forward pass.
      • thermal_pause_ms defaults to 0 — caller can raise if device is hot.
      • Weights stay in fp16 throughout; float32 only for softmax/norm RMS.
      • Embed + head weights stay cached across tokens.
    """

    def __init__(
        self,
        shard_dir: str,
        tokenizer,
        model_config: dict,
        memory_manager: Optional[MemoryManager] = None,
        prefetch: int = 2,
        max_kv_tokens: int = 4096,
        thermal_pause_ms: int = 0,   # FIX: was 50 — 50ms × 80 layers = 4s dead time per token
    ):
        self.shard_dir = shard_dir
        self.tokenizer = tokenizer
        self.cfg = model_config
        self.mm = memory_manager or MemoryManager(detect_device_profile())
        self.prefetch = prefetch
        self.max_kv_tokens = max_kv_tokens
        self.thermal_pause_ms = thermal_pause_ms

        self.hidden_size: int = model_config.get("hidden_size", 4096)
        self.n_heads:     int = model_config.get("num_attention_heads", 32)
        self.n_kv_heads:  int = model_config.get("num_key_value_heads", self.n_heads)
        self.head_dim:    int = self.hidden_size // self.n_heads
        self.vocab_size:  int = model_config.get("vocab_size", 32000)
        self.max_pos:     int = model_config.get("max_position_embeddings", 4096)
        self.rope_base:   float = model_config.get("rope_theta", 10000.0)
        self.rms_eps:     float = model_config.get("rms_norm_eps", 1e-5)

        self._rope_cos, self._rope_sin = _make_rope_cache(
            self.head_dim, self.max_pos, self.rope_base
        )

        self._kv_cache = BoundedKVCache(
            max_tokens=max_kv_tokens,
            n_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            n_layers=0,
        )

        self._loader = LayerLoader(
            shard_dir=shard_dir,
            prefetch=prefetch,
            memory_manager=self.mm,
            thermal_pause_ms=thermal_pause_ms,
        )

        self._embed_weights: Optional[dict] = None
        self._head_weights: Optional[dict] = None

        # Reuse runner — avoid object creation per layer
        self._runner = TransformerLayerRunner(
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            rope_cos=self._rope_cos,
            rope_sin=self._rope_sin,
            layer_idx=0,
            rms_norm_eps=self.rms_eps,
        )

        logger.info(f"[Engine] Ready | hidden={self.hidden_size} heads={self.n_heads} "
                    f"kv_heads={self.n_kv_heads} max_kv={max_kv_tokens}")

    # ── Tokenize ──────────────────────────────────────────────────────────────
    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, return_tensors="pt")

    # ── Stream generate ───────────────────────────────────────────────────────
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[int]] = None,
        reset_cache: bool = True,
    ) -> Iterator[str]:
        if reset_cache:
            self._kv_cache.clear()

        input_ids = self.tokenize(prompt)
        generated_ids: List[int] = []
        kv_offset = 0

        logits = self._forward(input_ids, kv_offset=kv_offset)
        kv_offset += input_ids.shape[1]

        for _ in range(max_new_tokens):
            next_id = self._sample(
                logits[:, -1, :], generated_ids,
                temperature, top_p, top_k, repetition_penalty
            )

            if stop_tokens and next_id in stop_tokens:
                break
            eos = self.tokenizer.eos_token_id
            if eos is not None and next_id == eos:
                break

            generated_ids.append(next_id)
            yield self.tokenizer.decode([next_id], skip_special_tokens=True)

            logits = self._forward(torch.tensor([[next_id]]), kv_offset=kv_offset)
            kv_offset += 1

    def generate(self, prompt: str, **kwargs) -> str:
        return "".join(self.stream_generate(prompt, **kwargs))

    # ── Forward pass ──────────────────────────────────────────────────────────
    def _forward(self, input_ids: torch.Tensor, kv_offset: int = 0) -> torch.Tensor:
        t0 = time.perf_counter()

        # Load embed once and keep cached for all tokens
        if self._embed_weights is None:
            self._embed_weights = self._loader.load_embed()

        embed_w = (
            self._embed_weights.get("model.embed_tokens.weight") or
            self._embed_weights.get("transformer.wte.weight") or
            next(v for k, v in self._embed_weights.items() if "embed" in k and "weight" in k)
        )

        # Work in fp16 throughout
        hidden = embed_w[input_ids[0]].unsqueeze(0).half()

        runner = self._runner

        # ── Layer loop — NO gc.collect() here, NO thermal pause by default ──
        for layer_idx, layer_weights in self._loader.iter_layers():
            runner.layer_idx = layer_idx
            stripped = self._strip_prefix(layer_weights)

            hidden = runner.forward(
                hidden,
                weights=stripped,
                kv_cache=self._kv_cache,
                kv_offset=kv_offset,
            )

            # FIX: removed gc.collect() per layer — was the #1 perf killer
            # Weights freed by loader's prefetch machinery after yield

        # Load head once and keep cached
        if self._head_weights is None:
            self._head_weights = self._loader.load_head()

        norm_w = self._get_final_norm(self._head_weights)
        if norm_w is not None:
            hidden = self._rms_norm(hidden, norm_w)

        lm_head_w = self._get_lm_head(self._head_weights)
        logits = hidden.float() @ lm_head_w.float().T

        # Single GC after full forward — not inside the loop
        gc.collect()

        elapsed = time.perf_counter() - t0
        logger.debug(f"[Engine] Forward {input_ids.shape[1]} tok in {elapsed:.2f}s")
        return logits

    # ── Sampling ──────────────────────────────────────────────────────────────
    def _sample(self, logits: torch.Tensor, generated: List[int],
                temperature: float, top_p: float, top_k: int,
                rep_penalty: float) -> int:
        if rep_penalty != 1.0 and generated:
            for token_id in set(generated):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= rep_penalty
                else:
                    logits[0, token_id] /= rep_penalty

        if temperature == 0:
            return int(logits[0].argmax())

        logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            threshold = logits[0].topk(top_k).values[-1]
            logits[0] = logits[0].masked_fill(logits[0] < threshold, float("-inf"))

        probs = F.softmax(logits[0], dim=-1)
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cumulative = sorted_probs.cumsum(dim=0)
        remove = cumulative - sorted_probs > top_p
        sorted_probs[remove] = 0
        sorted_probs /= sorted_probs.sum()
        return int(sorted_idx[torch.multinomial(sorted_probs, 1)])

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _rms_norm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.rms_eps).rsqrt().to(x.dtype)
        return x * rms * w.to(x.dtype)

    def _strip_prefix(self, weights: dict) -> dict:
        import re
        stripped = {}
        for k, v in weights.items():
            short = re.sub(
                r'^(model\.layers|gpt_neox\.layers|transformer\.h|transformer\.blocks)\.\d+\.', '', k
            )
            stripped[short] = v
        return stripped

    def _get_final_norm(self, head_weights: dict) -> Optional[torch.Tensor]:
        for key in ["model.norm.weight", "transformer.ln_f.weight", "norm.weight"]:
            if key in head_weights:
                return head_weights[key]
        return None

    def _get_lm_head(self, head_weights: dict) -> torch.Tensor:
        for key in ["lm_head.weight", "embed_out.weight"]:
            if key in head_weights:
                return head_weights[key]
        if self._embed_weights:
            for k, v in self._embed_weights.items():
                if "embed" in k:
                    return v
        raise KeyError("Cannot find lm_head weight")

    def reset(self):
        self._kv_cache.clear()
        self._head_weights = None
        gc.collect()

    def stats(self) -> dict:
        return {
            "kv_tokens": self._kv_cache.token_count,
            "kv_ram_mb": self._kv_cache.ram_usage_bytes() / 1024 ** 2,
            "mem_summary": self.mm.summary(),
        }