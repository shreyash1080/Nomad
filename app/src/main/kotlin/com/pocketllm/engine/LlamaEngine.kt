package com.pocketllm.engine

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Kotlin callback interface called from C++ for each generated token.
 * Return true to continue generation, false to stop.
 */
interface TokenCallback {
    fun onToken(piece: String): Boolean
}

/**
 * Singleton engine that owns the llama.cpp native context.
 * All heavy work is dispatched to [Dispatchers.IO].
 */
object LlamaEngine {

    private const val TAG = "LlamaEngine"

    // ── Native methods declared here, implemented in llama_wrapper.cpp ──────
    @JvmStatic external fun loadModel(
        modelPath: String,
        nCtx: Int,
        nThreads: Int,
        useGpu: Boolean
    ): Boolean

    @JvmStatic external fun unloadModel()
    @JvmStatic external fun isModelLoaded(): Boolean
    @JvmStatic external fun stopGeneration()
    @JvmStatic external fun getModelInfo(): String

    /** Blocking generate — use [generateFlow] for streaming UI. */
    @JvmStatic external fun generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        callback: TokenCallback
    ): String

    // ── Load the shared library ──────────────────────────────────────────────
    init {
        System.loadLibrary("pocketllm_jni")
        Log.i(TAG, "pocketllm_jni loaded")
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /**
     * Loads a .gguf model file from [file].
     * [onProgress] receives a 0–1 float (best-effort; llama.cpp doesn't always report it).
     */
    suspend fun load(
        file: File,
        contextSize: Int   = 4096,
        threads: Int       = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
        useGpu: Boolean    = true,
        onProgress: (Float) -> Unit = {}
    ): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            Log.i(TAG, "Loading ${file.name}  ctx=$contextSize  threads=$threads  gpu=$useGpu")
            onProgress(0.1f)
            val ok = loadModel(file.absolutePath, contextSize, threads, useGpu)
            onProgress(1.0f)
            if (ok) Result.success(Unit)
            else    Result.failure(RuntimeException("loadModel returned false"))
        } catch (e: Exception) {
            Log.e(TAG, "load() exception: ${e.message}")
            Result.failure(e)
        }
    }

    /**
     * Streams generated text as a [Flow] of token strings.
     */
    fun generateFlow(
        prompt: String,
        maxTokens: Int   = 512,
        temperature: Float = 0.7f,
        topP: Float      = 0.9f
    ): Flow<String> = flow {
        if (!isModelLoaded()) {
            emit("[ERROR: no model loaded]")
            return@flow
        }

        val chan = kotlinx.coroutines.channels.Channel<String>(capacity = 256)
        
        // Use a local flag to handle cancellation
        var isCancelled = false

        val callback = object : TokenCallback {
            override fun onToken(piece: String): Boolean {
                if (isCancelled) return false
                val sent = chan.trySend(piece).isSuccess
                return sent
            }
        }

        // Run generation in a separate coroutine within the flow's scope
        kotlinx.coroutines.coroutineScope {
            val genJob = launch(Dispatchers.IO) {
                try {
                    generate(prompt, maxTokens, temperature, topP, callback)
                } finally {
                    chan.close()
                }
            }

            try {
                for (token in chan) {
                    emit(token)
                }
            } finally {
                isCancelled = true
                stopGeneration()
                genJob.join()
            }
        }
    }.flowOn(Dispatchers.IO)

    /** Builds a chat prompt using standard TinyLlama/ChatML-like format. */
    fun buildChatPrompt(
        systemPrompt: String,
        history: List<Pair<String, String>>,   // (user, assistant)
        userMessage: String
    ): String = buildString {
        // Optimization: Use a smaller history window for faster prompt processing
        append("<|system|>\n$systemPrompt</s>\n")
        // Only keep last 3-4 turns to keep prompt small and processing fast
        val recentHistory = history.takeLast(4)
        for ((user, asst) in recentHistory) {
            append("<|user|>\n$user</s>\n")
            append("<|assistant|>\n$asst</s>\n")
        }
        append("<|user|>\n$userMessage</s>\n")
        append("<|assistant|>\n")
    }
}
