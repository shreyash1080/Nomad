package com.eigen.data

import android.net.Uri

enum class Role { USER, ASSISTANT, SYSTEM }

data class ChatMessage(
    val id: Long               = System.currentTimeMillis() + (Math.random() * 1000).toLong(),
    val role: Role,
    val content: String,
    val thought: String?       = null,
    val isStreaming: Boolean   = false,
    val attachment: Attachment? = null
)

data class ModelInfo(
    val id: String,
    val name: String,
    val description: String,
    val sizeLabel: String,
    val ramRequired: String,       // display string e.g. "4 GB RAM"
    val downloadUrl: String,
    val filename: String,
    val paramCount: String,        // display string e.g. "3.8B"
    val quantization: String,
    val isRecommended: Boolean = false,
    val supportsThinking: Boolean = false
) {
    /** Numeric GB value parsed from ramRequired — used for auto-load filtering */
    val ramRequirementGb: Double get() =
        ramRequired.replace(" GB RAM", "", ignoreCase = true)
            .replace(" GB", "", ignoreCase = true)
            .trim().toDoubleOrNull() ?: 99.0

    /** Numeric param count — used for ranking best model to auto-load */
    val paramCountValue: Double get() =
        paramCount.replace("B", "", ignoreCase = true)
            .replace("b", "", ignoreCase = true)
            .trim().toDoubleOrNull() ?: 0.0

    val fileSizeBytes: Long get() {
        val num = sizeLabel.replace(" GB", "", ignoreCase = true)
            .replace(" MB", "", ignoreCase = true)
            .trim().toDoubleOrNull() ?: 0.0
        return if (sizeLabel.contains("GB", ignoreCase = true)) (num * 1_073_741_824).toLong()
        else (num * 1_048_576).toLong()
    }
}

object ModelCatalog {
    val models: List<ModelInfo> = listOf(
        ModelInfo(
            id = "tinyllama-q4", name = "TinyLlama 1.1B",
            description = "Smallest model. Runs on any Android phone.",
            sizeLabel = "0.6 GB", ramRequired = "2 GB RAM",
            downloadUrl = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            filename = "tinyllama-1.1b-q4.gguf", paramCount = "1.1B", quantization = "Q4_K_M"
        ),
        ModelInfo(
            id = "gemma2-2b-q4", name = "Gemma 2 2B",
            description = "Google's compact model. Optimized for mobile.",
            sizeLabel = "1.6 GB", ramRequired = "3 GB RAM",
            downloadUrl = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
            filename = "gemma2-2b-q4.gguf", paramCount = "2B", quantization = "Q4_K_M",
            isRecommended = true
        ),
        ModelInfo(
            id = "gemma2-9b-q4", name = "Gemma 2 9B",
            description = "Google's powerful mid-size model. High quality answers.",
            sizeLabel = "5.4 GB", ramRequired = "10 GB RAM",
            downloadUrl = "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf",
            filename = "gemma2-9b-q4.gguf", paramCount = "9B", quantization = "Q4_K_M"
        ),
        ModelInfo(
            id = "phi3-mini-q4", name = "Phi-3 Mini",
            description = "Microsoft compact model. Great reasoning for its size.",
            sizeLabel = "2.2 GB", ramRequired = "4 GB RAM",
            downloadUrl = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
            filename = "phi3-mini-q4.gguf", paramCount = "3.8B", quantization = "Q4_K_M",
            isRecommended = true
        ),
        ModelInfo(
            id = "llama3.2-3b-q4", name = "Llama 3.2 3B",
            description = "Meta's latest compact model. Strong instruction following.",
            sizeLabel = "2.0 GB", ramRequired = "5 GB RAM",
            downloadUrl = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            filename = "llama3.2-3b-q4.gguf", paramCount = "3B", quantization = "Q4_K_M",
            isRecommended = true
        ),
        ModelInfo(
            id = "qwen2.5-7b-q4", name = "Qwen 2.5 7B",
            description = "Excellent at coding and math.",
            sizeLabel = "4.4 GB", ramRequired = "7 GB RAM",
            downloadUrl = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            filename = "qwen2.5-7b-q4.gguf", paramCount = "7B", quantization = "Q4_K_M"
        ),
        ModelInfo(
            id = "llama3.1-8b-q4", name = "Llama 3.1 8B",
            description = "Best quality for high-RAM phones.",
            sizeLabel = "4.9 GB", ramRequired = "8 GB RAM",
            downloadUrl = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            filename = "llama3.1-8b-q4.gguf", paramCount = "8B", quantization = "Q4_K_M",
            isRecommended = true
        ),
        ModelInfo(
            id = "mistral-7b-q5", name = "Mistral 7B v0.3",
            description = "Great for creative writing.",
            sizeLabel = "5.1 GB", ramRequired = "8 GB RAM",
            downloadUrl = "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
            filename = "mistral-7b-q5.gguf", paramCount = "7B", quantization = "Q5_K_M"
        )
    )
}