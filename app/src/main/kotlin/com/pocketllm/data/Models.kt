package com.pocketllm.data

import com.google.gson.annotations.SerializedName

// ── Chat message ─────────────────────────────────────────────────────────────

enum class Role { USER, ASSISTANT, SYSTEM }

data class ChatMessage(
    val id: String      = java.util.UUID.randomUUID().toString(),
    val role: Role,
    val content: String,
    val isStreaming: Boolean = false
)

// ── Model catalog entry ───────────────────────────────────────────────────────

data class ModelInfo(
    val id: String,
    val name: String,
    val description: String,
    val sizeLabel: String,        // e.g. "4.1 GB"
    val ramRequired: String,      // e.g. "6 GB RAM"
    val downloadUrl: String,
    val filename: String,
    val paramCount: String,       // e.g. "7B"
    val quantization: String,     // e.g. "Q4_K_M"
    val isRecommended: Boolean = false
) {
    val fileSizeBytes: Long get() {
        val num = sizeLabel.replace(" GB", "").replace(" MB", "").toDoubleOrNull() ?: 0.0
        return if (sizeLabel.contains("GB"))
            (num * 1_073_741_824).toLong()
        else
            (num * 1_048_576).toLong()
    }
}

// ── Built-in model catalog ────────────────────────────────────────────────────
// All models are GGUF quantized — downloadable from HuggingFace

object ModelCatalog {
    val models: List<ModelInfo> = listOf(

        // ── Ultra-light (8 GB phones) ──────────────────────────────────────
        ModelInfo(
            id          = "phi3-mini-q4",
            name        = "Phi-3 Mini",
            description = "Microsoft's compact powerhouse. Great reasoning for its size.",
            sizeLabel   = "2.2 GB",
            ramRequired = "4 GB RAM",
            downloadUrl = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
            filename    = "phi3-mini-q4.gguf",
            paramCount  = "3.8B",
            quantization= "Q4_K_M",
            isRecommended = true
        ),
        ModelInfo(
            id          = "gemma2-2b-q4",
            name        = "Gemma 2 2B",
            description = "Google's tiny but capable model. Fast on any phone.",
            sizeLabel   = "1.6 GB",
            ramRequired = "3 GB RAM",
            downloadUrl = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
            filename    = "gemma2-2b-q4.gguf",
            paramCount  = "2B",
            quantization= "Q4_K_M"
        ),
        ModelInfo(
            id          = "tinyllama-q4",
            name        = "TinyLlama 1.1B",
            description = "Smallest usable model. 1.1B params, runs on any Android phone.",
            sizeLabel   = "0.6 GB",
            ramRequired = "2 GB RAM",
            downloadUrl = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            filename    = "tinyllama-1.1b-q4.gguf",
            paramCount  = "1.1B",
            quantization= "Q4_K_M"
        ),

        // ── Mid-range (12 GB phones) ───────────────────────────────────────
        ModelInfo(
            id          = "llama3.2-3b-q4",
            name        = "Llama 3.2 3B",
            description = "Meta's latest compact model. Strong instruction following.",
            sizeLabel   = "2.0 GB",
            ramRequired = "5 GB RAM",
            downloadUrl = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            filename    = "llama3.2-3b-q4.gguf",
            paramCount  = "3B",
            quantization= "Q4_K_M",
            isRecommended = true
        ),
        ModelInfo(
            id          = "qwen2.5-7b-q4",
            name        = "Qwen 2.5 7B",
            description = "Alibaba's 7B model. Excellent at coding and math.",
            sizeLabel   = "4.4 GB",
            ramRequired = "7 GB RAM",
            downloadUrl = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            filename    = "qwen2.5-7b-q4.gguf",
            paramCount  = "7B",
            quantization= "Q4_K_M"
        ),

        // ── High-end (16–24 GB phones) ─────────────────────────────────────
        ModelInfo(
            id          = "llama3.1-8b-q4",
            name        = "Llama 3.1 8B",
            description = "Best overall quality for high-RAM phones.",
            sizeLabel   = "4.9 GB",
            ramRequired = "8 GB RAM",
            downloadUrl = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            filename    = "llama3.1-8b-q4.gguf",
            paramCount  = "8B",
            quantization= "Q4_K_M",
            isRecommended = true
        ),
        ModelInfo(
            id          = "mistral-7b-q5",
            name        = "Mistral 7B v0.3",
            description = "Fast, high-quality 7B model. Great for creative writing.",
            sizeLabel   = "5.1 GB",
            ramRequired = "8 GB RAM",
            downloadUrl = "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
            filename    = "mistral-7b-q5.gguf",
            paramCount  = "7B",
            quantization= "Q5_K_M"
        )
    )
}
