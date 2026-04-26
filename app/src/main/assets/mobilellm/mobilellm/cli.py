#!/usr/bin/env python3
"""
mobilellm CLI — command-line interface for MobileAirLLM

Commands:
  mobilellm serve                     Start HTTP server for Android app
  mobilellm split <model>             Pre-split model into shards
  mobilellm run <model> "<prompt>"    Quick inference from terminal
  mobilellm info                      Show device profile and RAM info
"""

import sys
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def cmd_serve(args):
    from mobilellm.server import run_server
    run_server(host=args.host, port=args.port)


def cmd_split(args):
    from mobilellm.splitter import ModelSplitter
    splitter = ModelSplitter(
        model_path=args.model,
        shard_dir=args.output or f"./shards/{args.model.replace('/', '--')}",
        compression=args.compression,
        delete_original=args.delete_original,
        hf_token=args.hf_token,
    )
    manifest = splitter.split(force=args.force)
    print(f"\n✅ Split complete!")
    print(f"   Layers    : {manifest['n_layers']}")
    print(f"   Shards at : {splitter.shard_dir}")
    total_gb = manifest['total_shard_bytes'] / 1024**3
    print(f"   Total size: {total_gb:.2f} GB ({args.compression})")


def cmd_run(args):
    from mobilellm import AutoModel
    print(f"Loading {args.model} …")
    model = AutoModel.from_pretrained(
        args.model,
        compression=args.compression,
        shard_dir=args.shard_dir,
        hf_token=args.hf_token,
    )
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*60}\n")
    for chunk in model.stream(args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature):
        print(chunk, end="", flush=True)
    print(f"\n{'='*60}")


def cmd_info(args):
    from mobilellm.memory_manager import detect_device_profile
    import psutil
    profile = detect_device_profile()
    mem = psutil.virtual_memory()
    print(f"\n📱 MobileAirLLM Device Info")
    print(f"{'='*40}")
    print(f"Profile           : {profile.name}")
    print(f"Total RAM         : {mem.total/1024**3:.1f} GB")
    print(f"Available RAM     : {mem.available/1024**3:.1f} GB")
    print(f"Model RAM budget  : {profile.model_ram_budget_gb:.1f} GB")
    print(f"Default quant     : {profile.default_quantization}")
    print(f"KV cache limit    : {profile.max_kv_tokens} tokens")
    print(f"Prefetch layers   : {profile.prefetch_layers}")
    print(f"\n💡 Recommended models for {profile.name}:")
    budget = profile.model_ram_budget_gb
    if budget >= 12:
        print("   • 70B model  @ 4-bit (≈ 35 GB → 14 GB compressed)")
    if budget >= 6:
        print("   • 30B model  @ 4-bit (≈ 16 GB →  8 GB compressed) ✓")
        print("   • 13B model  @ 8-bit (≈  6.5 GB)")
    if budget >= 3:
        print("   • 7B model   @ 4-bit (≈  4 GB)")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="mobilellm",
        description="MobileAirLLM — 30B models on 8 GB RAM"
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    p_serve = sub.add_parser("serve", help="Start HTTP server for Android app")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.set_defaults(func=cmd_serve)

    # split
    p_split = sub.add_parser("split", help="Pre-split a model into layer shards")
    p_split.add_argument("model", help="HuggingFace repo ID or local path")
    p_split.add_argument("-o", "--output", help="Output shard directory")
    p_split.add_argument("-c", "--compression", default="4bit", choices=["none","8bit","4bit"])
    p_split.add_argument("--hf-token", default=None)
    p_split.add_argument("--delete-original", action="store_true")
    p_split.add_argument("--force", action="store_true")
    p_split.set_defaults(func=cmd_split)

    # run
    p_run = sub.add_parser("run", help="Quick inference from terminal")
    p_run.add_argument("model", help="HuggingFace repo ID or local path")
    p_run.add_argument("prompt", help="Input prompt")
    p_run.add_argument("-c", "--compression", default="4bit")
    p_run.add_argument("--shard-dir", default=None)
    p_run.add_argument("--max-tokens", type=int, default=256)
    p_run.add_argument("--temperature", type=float, default=0.7)
    p_run.add_argument("--hf-token", default=None)
    p_run.set_defaults(func=cmd_run)

    # info
    p_info = sub.add_parser("info", help="Show device profile and recommendations")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
