package com.eigen.engine

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

    const val DEFAULT_CTX     = 2048   // Increased from 1536 to prevent crashes with documents
    const val DEFAULT_THREADS = 0      // 0 = auto P-core detect in C++
    const val DEFAULT_MAXTOK  = 300
    const val DEFAULT_SYSTEM  = "You are a helpful AI assistant named Eigen. Be concise and accurate."

    init { System.loadLibrary("eigen_jni") }

    // ── Load + immediately cache system prefix ────────────────────────────────
    suspend fun load(
        file: File,
        contextSize: Int = DEFAULT_CTX,
        threads: Int     = DEFAULT_THREADS,
        useGpu: Boolean  = true,
        systemPrompt: String = DEFAULT_SYSTEM,
        onProgress: (Float) -> Unit = {}
    ): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            onProgress(0.05f)
            val ok = loadModel(file.absolutePath, contextSize, threads, useGpu)
            if (!ok) return@withContext Result.failure(RuntimeException("loadModel failed"))
            onProgress(0.8f)
            // Cache system prefix — every chat turn will skip decoding these tokens
            val prefix = buildSystemPrefix(systemPrompt)
            val cached = cacheSystemPrompt(prefix)
            if (!cached) Log.w(TAG, "Prefix cache failed — will re-decode each turn")
            onProgress(1f)
            Result.success(Unit)
        } catch (e: Exception) { Result.failure(e) }
    }

    data class InferenceMetrics(
        val tokensPerSecond: Float = 0f,
        val firstTokenLatencyMs: Long = 0,
        val totalTokens: Int = 0,
        val durationMs: Long = 0
    )

    fun generateFlow(
        prompt: String,
        maxTokens: Int     = DEFAULT_MAXTOK,
        temperature: Float = 0.7f,
        topP: Float        = 0.9f,
        smoothingDelayMs: Long = 20L
    ): Flow<String> = flow {
        if (!isModelLoaded()) { emit("[ERROR: no model loaded]"); return@flow }
        
        val chan = Channel<String>(capacity = Channel.UNLIMITED)
        val startTime = System.currentTimeMillis()
        var firstTokenTime = 0L
        var tokenCount = 0

        withContext(Dispatchers.IO) {
            val job = launch {
                try {
                    val cb = object : TokenCallback {
                        override fun onToken(piece: String): Boolean {
                            if (tokenCount == 0) firstTokenTime = System.currentTimeMillis()
                            tokenCount++
                            chan.trySend(piece)
                            return true
                        }
                    }
                    generate(prompt, maxTokens, temperature, topP, cb)
                } catch (e: Exception) {
                    Log.e(TAG, "Native generate failed: ${e.message}")
                    chan.trySend("[ERROR: Inference crash]")
                } finally {
                    chan.close()
                }
            }

            try {
                for (tok in chan) {
                    emit(tok)
                    if (smoothingDelayMs > 0) kotlinx.coroutines.delay(smoothingDelayMs)
                }
            } finally {
                job.cancel()
                job.join()
            }
        }
        
        val totalTime = System.currentTimeMillis() - startTime
        val tps = if (totalTime > 0) (tokenCount / (totalTime / 1000f)) else 0f
        Log.i(TAG, "Inference complete: $tokenCount tokens, $tps t/s, Latency: ${firstTokenTime - startTime}ms")
    }.flowOn(Dispatchers.IO)

    // ── Prompt builders ───────────────────────────────────────────────────────

    /** System prefix only — must match EXACTLY what's passed to cacheSystemPrompt */
    fun buildSystemPrefix(systemPrompt: String = DEFAULT_SYSTEM): String =
        "<|im_start|>system\n$systemPrompt<|im_end|>\n"

    /**
     * Full chat prompt. System prefix tokens are already in KV cache —
     * C++ will detect the match and skip re-decoding them.
     *
     * Supports Llama 3 and ChatML formats based on model name.
     */
    fun buildChatPrompt(
        modelName: String,
        systemPrompt: String = DEFAULT_SYSTEM,
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