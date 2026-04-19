#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_llama.sh  –  Clone llama.cpp into the Android project
# Run this ONCE from the PocketLLM project root before opening in Android Studio
# ─────────────────────────────────────────────────────────────────────────────
set -e

LLAMA_DIR="app/src/main/cpp/llama.cpp"
LLAMA_TAG="b4388"   # tested stable tag — update to latest if you like

if [ -d "$LLAMA_DIR" ]; then
  echo "✅  llama.cpp already present at $LLAMA_DIR — skipping clone"
  exit 0
fi

echo "⬇️  Cloning llama.cpp @ tag $LLAMA_TAG ..."
git clone --depth 1 --branch "$LLAMA_TAG" \
  https://github.com/ggerganov/llama.cpp.git \
  "$LLAMA_DIR"

echo "✅  llama.cpp cloned successfully!"
echo ""
echo "Next steps:"
echo "  1. Open this folder in Android Studio"
echo "  2. Let Gradle sync (it will compile llama.cpp via CMake automatically)"
echo "  3. Connect your phone and press Run ▶"
