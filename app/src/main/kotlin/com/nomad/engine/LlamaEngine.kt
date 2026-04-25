package com.nomad.engine

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

object LlamaEngine {
    private const val TAG = "LlamaEngine"

    interface TokenCallback {
        fun onToken(piece: String): Boolean
    }

    @JvmStatic external fun loadModel(modelPath: String, nCtx: Int, nThreads: Int, useGpu: Boolean): Boolean
    @JvmStatic external fun cacheSystemPrompt(systemPrefix: String): Boolean
    @JvmStatic external fun setThreads(nThreads: Int, nThreadsBatch: Int)
    @JvmStatic external fun unloadModel()
    @JvmStatic external fun isModelLoaded(): Boolean
    @JvmStatic external fun stopGeneration()
    @JvmStatic external fun getModelInfo(): String
    @JvmStatic external fun generate(
        prompt: String, maxTokens: Int,
        temperature: Float, topP: Float,
        callback: TokenCallback
    ): String

    const val DEFAULT_CTX     = 4096 // Increased default context for better long-form answers
    const val DEFAULT_THREADS = 0
    const val DEFAULT_MAXTOK  = 2500 // Increased to allow much longer responses
    const val DEFAULT_SYSTEM = """
You are Nomad, an on-device AI assistant optimized for speed, accuracy, and practical usefulness.

CORE BEHAVIOR:
- Be concise, direct, and highly actionable.
- Avoid unnecessary explanations unless asked.
- Prefer structured outputs (steps, bullets, code).

FILE SEARCH & LOCAL DATA (IMPORTANT):
- When a user asks "find [something]" or uses the search icon, they specifically want you to help locate files or data on their device.
- The system will provide search results from the local file system. 
- You MUST analyze these results and tell the user where their files are or summarize what was found.
- If the user is asking about "finding something in an array" (programming context), do NOT confuse this with local file search. Check if the conversation is about C++, Java, or logic.
- Only assume a file search intent if the "find" keyword is used in a non-programming context or if file metadata is provided in the context.

LOCAL DEVICE AWARENESS:
- You run fully offline on a mobile device.
- Assume limited memory and compute; keep answers efficient.

CODING & TECHNICAL TASKS:
- Provide production-ready code (clean, optimized).
- Handle Android, Kotlin, APIs, backend, and system design tasks effectively.

GOAL:
Deliver fast, practical, and high-quality assistance for daily tasks, coding, and problem solving.
"""
    init {
        try {
            System.loadLibrary("nomad_jni")
        } catch (_: UnsatisfiedLinkError) {
            // JVM unit tests exercise prompt builders without packaging native libs.
        }
    }

    // ── Load + immediately cache system prefix ──────────────────────────────
    suspend fun load(
        file: File,
        contextSize: Int     = DEFAULT_CTX,
        threads: Int         = DEFAULT_THREADS,
        useGpu: Boolean      = true,
        systemPrompt: String = DEFAULT_SYSTEM,
        onProgress: (Float) -> Unit = {}
    ): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            onProgress(0.05f)
            val ok = loadModel(file.absolutePath, contextSize, threads, useGpu)
            if (!ok) return@withContext Result.failure(RuntimeException("loadModel failed"))
            onProgress(0.8f)
            val prefix = buildSystemPrefix(systemPrompt)
            val cached = cacheSystemPrompt(prefix)
            if (!cached) Log.w(TAG, "Prefix cache failed — will re-decode each turn")
            onProgress(1f)
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Streams tokens from the native inference engine.
     *
     * BUG FIX (CRASH): The original code used flow { withContext(IO) { emit(tok) } }
     * which violates Kotlin's Flow invariant — you cannot call emit() from a different
     * coroutine context inside a flow{} builder. This caused an IllegalStateException
     * that crashed the app every time a message was sent.
     *
     * FIX: callbackFlow is purpose-built for converting callback APIs to flows.
     * trySend() is thread-safe and can be called from the JNI thread inside onToken().
     */
    fun generateFlow(
        prompt: String,
        maxTokens: Int     = DEFAULT_MAXTOK,
        temperature: Float = 0.7f,
        topP: Float        = 0.9f
    ): Flow<String> = callbackFlow {
        if (!isModelLoaded()) {
            trySend("[ERROR: no model loaded]")
            close()
            return@callbackFlow
        }

        val startTime = System.currentTimeMillis()
        var firstTokenTime = 0L
        var tokenCount = 0

        val cb = object : TokenCallback {
            override fun onToken(piece: String): Boolean {
                if (tokenCount == 0) firstTokenTime = System.currentTimeMillis()
                tokenCount++
                // trySend is non-blocking and thread-safe; called from JNI thread
                return trySend(piece).isSuccess
            }
        }

        val job = launch(Dispatchers.IO) {
            try {
                generate(prompt, maxTokens, temperature, topP, cb)
            } catch (e: Exception) {
                Log.e(TAG, "Native generate failed: ${e.message}", e)
                trySend("[ERROR: ${e.message}]")
            } finally {
                val totalMs = System.currentTimeMillis() - startTime
                val tps = if (totalMs > 0) tokenCount / (totalMs / 1000f) else 0f
                val latency = if (firstTokenTime > 0) firstTokenTime - startTime else -1L
                Log.i(TAG, "Inference done: $tokenCount tokens, ${"%.1f".format(tps)} t/s, latency=${latency}ms")
                close()
            }
        }

        // Called when downstream cancels (e.g. user presses Stop)
        awaitClose {
            stopGeneration()
            job.cancel()
        }
    }
        .buffer(Channel.UNLIMITED) // Prevent JNI thread stalling while UI updates
        .flowOn(Dispatchers.IO)

    // ── Prompt builders ─────────────────────────────────────────────────────

    /** ChatML system prefix — must match exactly what cacheSystemPrompt receives */
    fun buildSystemPrefix(systemPrompt: String = DEFAULT_SYSTEM): String =
        "<|im_start|>system\n$systemPrompt<|im_end|>\n"

    /**
     * Full chat prompt dispatched to the correct format per model family.
     *
     * BUG FIX (CRASH): Gemma 2 was receiving ChatML format prompts, which it does
     * not recognise. This caused the model to produce garbage and run to the end of
     * the context window, triggering a native OOM crash.
     *
     * Gemma 2 uses <start_of_turn>user / <end_of_turn> with the system prompt
     * embedded in the first user turn (it has no dedicated system role token).
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
        val isGemma  = modelName.contains("Gemma", ignoreCase = true)

        val sysContent = buildString {
            append(systemPrompt)
            if (fileContext.isNotBlank() && !isGemma) {
                // Gemma injects file context inline in buildGemmaPrompt
                append("\n\n[File attached by user]\n")
                append(fileContext.take(3000))
            }
        }

        return when {
            isLlama3 -> buildLlama3Prompt(sysContent, history, userMessage)
            isGemma  -> buildGemmaPrompt(systemPrompt, history, userMessage, fileContext)
            else     -> buildChatMLPrompt(sysContent, history, userMessage, fileContext, systemPrompt)
        }
    }

    /** Llama 3 / 3.1 / 3.2 */
    private fun buildLlama3Prompt(
        sysContent: String,
        history: List<Pair<String, String>>,
        userMessage: String
    ): String = buildString {
        append("<|begin_of_text|>")
        append("<|start_header_id|>system<|end_header_id|>\n\n$sysContent<|eot_id|>")
        for ((u, a) in history.takeLast(4)) {
            append("<|start_header_id|>user<|end_header_id|>\n\n$u<|eot_id|>")
            append("<|start_header_id|>assistant<|end_header_id|>\n\n$a<|eot_id|>")
        }
        append("<|start_header_id|>user<|end_header_id|>\n\n$userMessage<|eot_id|>")
        append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    }

    /**
     * Gemma 2 native format.
     * System prompt is folded into the first user turn (Gemma has no system role).
     * Stop token is <end_of_turn> — ensure C++ STOP_SEQS contains this (no pipes).
     */
    private fun buildGemmaPrompt(
        systemPrompt: String,
        history: List<Pair<String, String>>,
        userMessage: String,
        fileContext: String
    ): String = buildString {
        val recent = history.takeLast(4)

        for ((idx, pair) in recent.withIndex()) {
            val (u, a) = pair
            // System prompt prepended only to the very first user turn
            val userContent = if (idx == 0) "$systemPrompt\n\n$u" else u
            append("<start_of_turn>user\n$userContent<end_of_turn>\n")
            append("<start_of_turn>model\n$a<end_of_turn>\n")
        }

        // Build current user message
        val currentUserContent = buildString {
            if (recent.isEmpty()) {
                // No history: inject system prompt here
                append(systemPrompt)
                append("\n\n")
            }
            if (fileContext.isNotBlank()) {
                append("[File content]:\n")
                append(fileContext.take(3000))
                append("\n\n")
            }
            append(userMessage)
        }
        append("<start_of_turn>user\n$currentUserContent<end_of_turn>\n")
        append("<start_of_turn>model\n")
    }

    /** ChatML — Qwen, Phi-3, Mistral, most instruction-tuned models */
    private fun buildChatMLPrompt(
        sysContent: String,
        history: List<Pair<String, String>>,
        userMessage: String,
        fileContext: String,
        systemPrompt: String
    ): String = buildString {
        append(buildSystemPrefix(systemPrompt)) // matches KV prefix cache
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
