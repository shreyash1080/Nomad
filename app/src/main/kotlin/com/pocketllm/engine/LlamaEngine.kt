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
    @JvmStatic external fun unloadModel()
    @JvmStatic external fun isModelLoaded(): Boolean
    @JvmStatic external fun stopGeneration()
    @JvmStatic external fun getModelInfo(): String
    @JvmStatic external fun generate(prompt: String, maxTokens: Int, temperature: Float, topP: Float, callback: TokenCallback): String

    const val DEFAULT_CTX     = 2048
    const val DEFAULT_THREADS = 0
    const val DEFAULT_MAXTOK  = 512

    init { System.loadLibrary("pocketllm_jni") }

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
            onProgress(1f)
            if (ok) Result.success(Unit) else Result.failure(RuntimeException("loadModel failed"))
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
        
        // Common stop tokens across formats
        val stopTokens = listOf("<|im_end|>", "<|eot_id|>", "</s>", "<|end_of_text|>", "<|end|>")
        
        val job = kotlinx.coroutines.GlobalScope.launch(Dispatchers.IO) {
            try {
                val cb = object : TokenCallback {
                    private var fullResponse = StringBuilder()

                    override fun onToken(piece: String): Boolean {
                        fullResponse.append(piece)
                        val currentText = fullResponse.toString()
                        
                        // If we see a stop token, signal the engine to stop
                        for (stop in stopTokens) {
                            if (currentText.contains(stop)) {
                                return false
                            }
                        }
                        
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

    /**
     * Build a prompt — selects format based on model name.
     */
    fun buildChatPrompt(
        modelName: String,
        systemPrompt: String,
        history: List<Pair<String, String>>,
        userMessage: String,
        fileContext: String = ""
    ): String {
        val isLlama3 = modelName.contains("Llama-3", ignoreCase = true) || modelName.contains("Llama 3", ignoreCase = true)
        
        val sysContent = buildString {
            append(systemPrompt)
            if (fileContext.isNotBlank()) {
                append("\n\nThe user has uploaded a file. Content:\n---\n")
                append(fileContext.take(4000))
                append("\n---")
            }
        }

        return if (isLlama3) {
            // Llama 3 / 3.1 / 3.2 Format
            buildString {
                append("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n$sysContent<|eot_id|>")
                for ((u, a) in history.takeLast(4)) {
                    append("<|start_header_id|>user<|end_header_id|>\n\n$u<|eot_id|>")
                    append("<|start_header_id|>assistant<|end_header_id|>\n\n$a<|eot_id|>")
                }
                append("<|start_header_id|>user<|end_header_id|>\n\n$userMessage<|eot_id|>")
                append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            }
        } else {
            // ChatML Format (Qwen, DeepSeek, etc.)
            buildString {
                append("<|im_start|>system\n$sysContent<|im_end|>\n")
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