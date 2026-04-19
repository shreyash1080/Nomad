package com.pocketllm.engine

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

interface TokenCallback {
    fun onToken(piece: String): Boolean
}

object LlamaEngine {
    private const val TAG = "LlamaEngine"

    @JvmStatic external fun loadModel(modelPath: String, nCtx: Int, nThreads: Int, useGpu: Boolean): Boolean
    @JvmStatic external fun cacheSystemPrompt(systemPrefix: String): Boolean  // NEW — KV prefix cache
    @JvmStatic external fun unloadModel()
    @JvmStatic external fun isModelLoaded(): Boolean
    @JvmStatic external fun stopGeneration()
    @JvmStatic external fun getModelInfo(): String
    @JvmStatic external fun generate(
        prompt: String, maxTokens: Int,
        temperature: Float, topP: Float,
        callback: TokenCallback
    ): String

    const val DEFAULT_CTX     = 1536   // 1536 > 2048 for faster KV alloc; plenty for chat
    const val DEFAULT_THREADS = 0      // 0 = auto P-core detect in C++
    const val DEFAULT_MAXTOK  = 300

    private const val SYSTEM_PROMPT =
        "You are a helpful AI assistant. Be concise and accurate."

    init { System.loadLibrary("pocketllm_jni") }

    // ── Load + immediately cache system prefix ────────────────────────────────
    suspend fun load(
        file: File,
        contextSize: Int = DEFAULT_CTX,
        threads: Int     = DEFAULT_THREADS,
        useGpu: Boolean  = true,
        onProgress: (Float) -> Unit = {}
    ): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            onProgress(0.05f)
            val ok = loadModel(file.absolutePath, contextSize, threads, useGpu)
            if (!ok) return@withContext Result.failure(RuntimeException("loadModel failed"))
            onProgress(0.8f)
            // Cache system prefix — every chat turn will skip decoding these tokens
            val prefix = buildSystemPrefix(SYSTEM_PROMPT)
            val cached = cacheSystemPrompt(prefix)
            if (!cached) Log.w(TAG, "Prefix cache failed — will re-decode each turn")
            onProgress(1f)
            Result.success(Unit)
        } catch (e: Exception) { Result.failure(e) }
    }

    fun generateFlow(
        prompt: String,
        maxTokens: Int     = DEFAULT_MAXTOK,
        temperature: Float = 0.7f,
        topP: Float        = 0.9f
    ): Flow<String> = flow {
        if (!isModelLoaded()) { emit("[ERROR: no model loaded]"); return@flow }
        val chan = Channel<String>(capacity = 512)
        val job = kotlinx.coroutines.GlobalScope.launch(Dispatchers.IO) {
            try {
                val cb = object : TokenCallback {
                    override fun onToken(piece: String): Boolean {
                        chan.trySend(piece)
                        return !chan.isClosedForSend
                    }
                }
                generate(prompt, maxTokens, temperature, topP, cb)
            } finally { chan.close() }
        }
        for (tok in chan) emit(tok)
        job.join()
    }.flowOn(Dispatchers.IO)

    // ── Prompt builders ───────────────────────────────────────────────────────

    /** System prefix only — must match EXACTLY what's passed to cacheSystemPrompt */
    fun buildSystemPrefix(systemPrompt: String = SYSTEM_PROMPT): String =
        "<|im_start|>system\n$systemPrompt<|im_end|>\n"

    /**
     * Full chat prompt. System prefix tokens are already in KV cache —
     * C++ will detect the match and skip re-decoding them.
     *
     * Supports Llama 3 and ChatML formats based on model name.
     */
    fun buildChatPrompt(
        modelName: String,
        systemPrompt: String = SYSTEM_PROMPT,
        history: List<Pair<String, String>>,
        userMessage: String,
        fileContext: String = ""
    ): String {
        val isLlama3 = modelName.contains("Llama-3", ignoreCase = true) ||
                modelName.contains("Llama 3", ignoreCase = true)

        val sysContent = buildString {
            append(systemPrompt)
            if (fileContext.isNotBlank()) {
                append("\n\n[File attached by user]\n")
                append(fileContext.take(3000))
            }
        }

        return if (isLlama3) {
            // Llama 3 / 3.1 / 3.2 format — uses different tokens than ChatML
            buildString {
                append("<|begin_of_text|>")
                append("<|start_header_id|>system<|end_header_id|>\n\n$sysContent<|eot_id|>")
                for ((u, a) in history.takeLast(4)) {
                    append("<|start_header_id|>user<|end_header_id|>\n\n$u<|eot_id|>")
                    append("<|start_header_id|>assistant<|end_header_id|>\n\n$a<|eot_id|>")
                }
                append("<|start_header_id|>user<|end_header_id|>\n\n$userMessage<|eot_id|>")
                append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            }
        } else {
            // ChatML — Qwen, Phi-3, Mistral, Gemma, most others
            // System prefix matches KV cache → C++ skips decoding it
            buildString {
                append(buildSystemPrefix(systemPrompt))  // ← MUST match cacheSystemPrompt exactly
                if (fileContext.isNotBlank()) {
                    append("<|im_start|>system\n[File content]:\n")
                    append(fileContext.take(3000))
                    append("<|im_end|>\n")
                }
                for ((u, a) in history.takeLast(4)) {
                    append("<|im_start|>user\n$u<|im_end|>\n")
                    append("<|im_start|>assistant\n$a<|im_end|>\n")
                }
                append("<|im_start|>user\n$userMessage<|im_end|>\n")
                append("<|im_start|>assistant\n")
            }
        }
    }
}