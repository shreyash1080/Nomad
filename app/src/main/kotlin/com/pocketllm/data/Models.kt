package com.pocketllm.data

import android.net.Uri

enum class Role { USER, ASSISTANT, SYSTEM }

data class ChatMessage(
    val id: Long              = System.currentTimeMillis() + (Math.random() * 1000).toLong(),
    val role: Role,
    val content: String,
    val isStreaming: Boolean  = false,
    val attachment: Attachment? = null   // ← NEW: attached file
)

data class ModelInfo(
    val id: String,
    val name: String,
    val description: String,
    val sizeLabel: String,
    val ramRequired: String,
    val downloadUrl: String,
    val filename: String,
    val paramCount: String,
    val quantization: String,
    val isRecommended: Boolean = false
) {
    val fileSizeBytes: Long get() {
        val num = sizeLabel.replace(" GB","").replace(" MB","").toDoubleOrNull() ?: 0.0
        return if (sizeLabel.contains("GB")) (num * 1_073_741_824).toLong()
        else (num * 1_048_576).toLong()
    }

    val ramRequirementGb: Double get() {
        return ramRequired.replace(" GB RAM", "").toDoubleOrNull() ?: 4.0
    }

    val paramCountValue: Double get() {
        return paramCount.replace("B", "").toDoubleOrNull() ?: 0.0
    }
}

object ModelCatalog {
    val models: List<ModelInfo> = listOf(
        ModelInfo("tinyllama-q4","TinyLlama 1.1B","Smallest model. Fast on any phone.","0.6 GB","2 GB RAM",
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "tinyllama-1.1b-q4.gguf","1.1B","Q4_K_M"),
        ModelInfo("gemma2-2b-q4","Gemma 2 2B","Google's tiny model. Fast on any phone.","1.6 GB","3 GB RAM",
            "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
            "gemma2-2b-q4.gguf","2B","Q4_K_M"),
        ModelInfo("phi3-mini-q4","Phi-3 Mini","Microsoft compact model. Great reasoning.","2.2 GB","4 GB RAM",
            "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
            "phi3-mini-q4.gguf","3.8B","Q4_K_M",isRecommended=true),
        ModelInfo("llama3.2-3b-q4","Llama 3.2 3B","Meta latest. Strong instruction following.","2.0 GB","5 GB RAM",
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "llama3.2-3b-q4.gguf","3B","Q4_K_M",isRecommended=true),
        ModelInfo("qwen2.5-7b-q4","Qwen 2.5 7B","Excellent at coding and math.","4.4 GB","7 GB RAM",
            "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "qwen2.5-7b-q4.gguf","7B","Q4_K_M"),
        ModelInfo("llama3.1-8b-q4","Llama 3.1 8B","Best quality for high-RAM phones.","4.9 GB","8 GB RAM",
            "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "llama3.1-8b-q4.gguf","8B","Q4_K_M",isRecommended=true),
        ModelInfo("mistral-7b-q5","Mistral 7B v0.3","Great for creative writing.","5.1 GB","8 GB RAM",
            "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
            "mistral-7b-q5.gguf","7B","Q5_K_M")
    )
}