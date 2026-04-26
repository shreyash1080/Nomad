"""
MobileAirLLM — Run 30B+ LLMs on 8 GB RAM mobile devices
Layer-by-layer inference with 4-bit quantization and disk streaming.

Quick start:
    from mobilellm import AutoModel
    model = AutoModel.from_pretrained("meta-llama/Llama-2-30b-hf", compression="4bit")
    print(model.generate("Hello, world!"))
"""

from .auto_model import AutoModel
from .engine import MobileInferenceEngine
from .memory_manager import MemoryManager, detect_device_profile, DEVICE_PROFILES
from .splitter import ModelSplitter
from .loader import LayerLoader, BoundedKVCache
from .server import run_server

__version__ = "1.0.0"
__author__ = "MobileAirLLM"
__all__ = [
    "AutoModel",
    "MobileInferenceEngine",
    "MemoryManager",
    "ModelSplitter",
    "LayerLoader",
    "BoundedKVCache",
    "detect_device_profile",
    "DEVICE_PROFILES",
    "run_server",
]
