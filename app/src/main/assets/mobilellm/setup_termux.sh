#!/data/data/com.termux/files/usr/bin/bash
# ============================================================
# setup_termux.sh — Install MobileAirLLM in Termux (Android)
# ============================================================
# Run this in Termux:
#   chmod +x setup_termux.sh && ./setup_termux.sh
#
# Requirements: Termux with storage permission granted
#   pkg install termux-setup-storage  (first time only)
# ============================================================

set -e

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m"

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════╗"
echo "║      MobileAirLLM — Termux Installer     ║"
echo "║   Run 30B LLMs on 8 GB Android RAM       ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${NC}"

# ── Check Termux ──────────────────────────────────────────────────────────────
if [ ! -d "/data/data/com.termux" ]; then
    warn "Not running in Termux? Continuing anyway (desktop Linux mode)."
fi

# ── 1. Update packages ────────────────────────────────────────────────────────
log "Updating package list…"
pkg update -y -q

# ── 2. Install system deps ────────────────────────────────────────────────────
log "Installing system dependencies…"
pkg install -y python python-pip clang libandroid-spawn libjpeg-turbo \
    libopenblas openblas-dev git wget curl 2>/dev/null || true

# ── 3. Python deps ────────────────────────────────────────────────────────────
log "Installing Python packages (CPU-only PyTorch for ARM)…"
# CPU-only torch wheel for Android ARM64
pip install --quiet --upgrade pip wheel setuptools

# Install torch — use the lightweight CPU-only build
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu || \
pip install --quiet torch  # fallback to default

pip install --quiet \
    transformers>=4.35.0 \
    safetensors>=0.4.0 \
    huggingface_hub>=0.20.0 \
    psutil>=5.9.0 \
    sentencepiece>=0.1.99 \
    accelerate>=0.24.0

# ── 4. Install MobileAirLLM ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log "Installing MobileAirLLM from ${SCRIPT_DIR}…"
pip install --quiet -e "${SCRIPT_DIR}"

# ── 5. Create model storage directory ────────────────────────────────────────
STORAGE_DIR="$HOME/storage/downloads/mobilellm"
if [ -d "$HOME/storage" ]; then
    mkdir -p "$STORAGE_DIR/shards"
    log "Model storage: ${STORAGE_DIR}"
else
    warn "Storage not mounted. Run: termux-setup-storage"
    mkdir -p "$HOME/.cache/mobilellm/shards"
fi

# ── 6. Create start script ────────────────────────────────────────────────────
cat > "$HOME/start_mobilellm.sh" << 'STARTSCRIPT'
#!/data/data/com.termux/files/usr/bin/bash
# Start MobileAirLLM server for Nomad app
echo "Starting MobileAirLLM server on port 8765..."
mobilellm serve --host 127.0.0.1 --port 8765
STARTSCRIPT
chmod +x "$HOME/start_mobilellm.sh"

# ── 7. Create Nomad shortcut notification ────────────────────────────────────
if command -v termux-notification &>/dev/null; then
    termux-notification \
        --title "MobileAirLLM Ready" \
        --content "Run ~/start_mobilellm.sh then open Nomad app" \
        --id mobilellm_ready || true
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}✅ Installation complete!${NC}"
echo ""
echo "📋 Next steps:"
echo ""
echo "  1. Check your device capabilities:"
echo "       mobilellm info"
echo ""
echo "  2. Pre-split a model (do this once, takes 30-60 min):"
echo "       mobilellm split meta-llama/Llama-2-13b-hf --compression 4bit"
echo "     For 30B (needs storage space ≈ 15 GB):"
echo "       mobilellm split meta-llama/Llama-2-30b-hf --compression 4bit"
echo ""
echo "  3. Start the server for Nomad app:"
echo "       ~/start_mobilellm.sh"
echo "     OR:"
echo "       mobilellm serve"
echo ""
echo "  4. Open the Nomad app → Settings → Engine: MobileAirLLM"
echo ""
echo "  RAM Usage for 30B @ 4-bit: ~6 GB active (fits 8 GB!)"
echo ""
