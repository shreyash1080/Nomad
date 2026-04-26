package com.nomad.mobilellm

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

/**
 * AirLLMEngine
 *
 * Drop-in engine that replaces llama.cpp for models > 8B.
 * Integrates with Nomad's existing ViewModel/Repository pattern.
 *
 * In your ChatViewModel or LlmRepository, replace:
 *   val engine = LlamaCppEngine(modelFile)
 * with:
 *   val engine = AirLLMEngine(context, modelId = "meta-llama/Llama-2-30b-hf")
 *
 * Both implement the same interface so no other code changes needed.
 */
class AirLLMEngine(
    private val context: Context,
    private val modelId: String,
    private val compression: String = "4bit",
    private val shardDir: String? = null,
    private val hfToken: String? = null,
) {
    companion object {
        private const val TAG = "AirLLMEngine"
    }

    private val client = MobileAirLLMClient()
    private var engineState = EngineState.UNLOADED

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    suspend fun initialize(): InitResult {
        Log.i(TAG, "Checking MobileAirLLM server…")
        
        // Auto-start server if not running
        if (!client.isServerRunning()) {
            Log.i(TAG, "Server not detected. Attempting auto-start...")
            MobileLLMServerManager.startServer(context)
            
            // Wait for server to heartbeat
            var serverStarted = false
            repeat(15) { // wait up to 15 seconds for server startup
                if (client.isServerRunning()) {
                    serverStarted = true
                    return@repeat
                }
                delay(1000L)
            }
            
            if (!serverStarted) {
                return InitResult.ServerNotRunning(
                    instructions = buildTermuxInstructions()
                )
            }
        }

        val status = client.getStatus()
        if (status.isReady && status.model == modelId) {
            engineState = EngineState.READY
            Log.i(TAG, "Engine already loaded: $modelId")
            return InitResult.Success
        }

        Log.i(TAG, "Loading model: $modelId (compression=$compression)")
        engineState = EngineState.LOADING

        val ok = client.loadModel(
            modelPath = modelId,
            compression = compression,
            shardDir = shardDir,
            hfToken = hfToken,
        )

        if (!ok) {
            engineState = EngineState.ERROR
            return InitResult.LoadFailed("Server rejected load request")
        }

        // Poll until ready (model splitting can take several minutes on first run)
        repeat(360) { // wait up to 6 minutes
            delay(1000L)
            val s = client.getStatus()
            if (s.isReady) {
                engineState = EngineState.READY
                return InitResult.Success
            }
            if (s.isError) {
                engineState = EngineState.ERROR
                return InitResult.LoadFailed(s.error ?: "Unknown load error")
            }
        }

        engineState = EngineState.ERROR
        return InitResult.LoadFailed("Timeout waiting for model to load")
    }

    // ── Inference ─────────────────────────────────────────────────────────────
    fun generateFlow(
        prompt: String,
        maxNewTokens: Int = 512,
        temperature: Float = 0.7f,
        topP: Float = 0.9f,
        topK: Int = 40,
        systemPrompt: String? = null,
        resetCache: Boolean = true,
    ): Flow<String> = flow {
        check(engineState == EngineState.READY) {
            "Engine not ready. Call initialize() first."
        }

        val fullPrompt = buildPrompt(systemPrompt, prompt)

        client.streamGenerate(
            prompt = fullPrompt,
            maxNewTokens = maxNewTokens,
            temperature = temperature,
            topP = topP,
            topK = topK,
            resetCache = resetCache,
        ).collect { event ->
            when (event) {
                is MobileAirLLMClient.StreamEvent.Token -> emit(event.text)
                is MobileAirLLMClient.StreamEvent.Done -> { /* done */ }
                is MobileAirLLMClient.StreamEvent.Error -> throw RuntimeException(event.message)
            }
        }
    }

    suspend fun generate(
        prompt: String,
        maxNewTokens: Int = 512,
        temperature: Float = 0.7f,
        systemPrompt: String? = null,
    ): String {
        val sb = StringBuilder()
        generateFlow(prompt, maxNewTokens, temperature, systemPrompt = systemPrompt)
            .collect { sb.append(it) }
        return sb.toString()
    }

    fun newConversation() {
        CoroutineScope(Dispatchers.IO).launch { client.resetCache() }
    }

    suspend fun unload() {
        client.unload()
        engineState = EngineState.UNLOADED
    }

    val isReady: Boolean get() = engineState == EngineState.READY

    // ── Prompt formatting ─────────────────────────────────────────────────────
    private fun buildPrompt(system: String?, user: String): String {
        // LLaMA 2 / 3 chat template
        return if (system != null) {
            "<s>[INST] <<SYS>>\n$system\n<</SYS>>\n\n$user [/INST]"
        } else {
            "<s>[INST] $user [/INST]"
        }
    }

    // ── Helper data ───────────────────────────────────────────────────────────
    private fun buildTermuxInstructions(): String = """
        MobileAirLLM server is not running.
        
        To start it:
        1. Open Termux app
        2. Run: ~/start_mobilellm.sh
        3. Return to Nomad
        
        First-time setup in Termux:
        1. Install MobileAirLLM:
           cd /sdcard/Download && chmod +x setup_termux.sh && ./setup_termux.sh
        2. Then run: mobilellm serve
    """.trimIndent()

    enum class EngineState { UNLOADED, LOADING, READY, ERROR }

    sealed class InitResult {
        object Success : InitResult()
        data class ServerNotRunning(val instructions: String) : InitResult()
        data class LoadFailed(val reason: String) : InitResult()
    }
}
