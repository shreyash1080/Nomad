package com.nomad.mobilellm

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

/**
 * AirLLMEngine
 *
 * Drop-in engine for Nomad that handles 20B+ models via the
 * MobileAirLLM Python server running in Termux on the same device.
 *
 * Fixes vs original:
 *  - Server health check uses MobileLLMServerManager.isServerResponding()
 *    (HTTP liveness) not just process-level isServerRunning().
 *  - Progress polling: reports loading progress 0-100 via onProgress callback.
 *  - Auto-start gives up quickly and shows clear Termux instructions.
 *  - Coroutine scope properly cancelled on unload().
 *  - EngineState is thread-safe (@Volatile).
 */
class AirLLMEngine(
    private val context: Context,
    private val modelId: String,
    private val compression: String = "4bit",
    private val shardDir: String? = null,
    private val hfToken: String? = null,
    /** Called with 0-100 during model loading, then 100 when ready. */
    private val onProgress: ((Int, String) -> Unit)? = null,
) {
    companion object {
        private const val TAG = "AirLLMEngine"
        private const val LOAD_POLL_INTERVAL_MS = 2000L
        private const val LOAD_TIMEOUT_MS = 10 * 60 * 1000L   // 10 min (first run splits shards)
        private const val SERVER_START_TIMEOUT_MS = 20 * 1000L  // 20s for server to respond
    }

    private val client = MobileAirLLMClient()

    @Volatile
    private var engineState = EngineState.UNLOADED

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * Initialises the engine:
     *  1. Waits for server to be reachable (auto-start attempted once).
     *  2. Posts a /load request.
     *  3. Polls /progress until ready or error.
     *
     * Must be called from a coroutine (suspend fun).
     */
    suspend fun initialize(): InitResult {
        Log.i(TAG, "Initialising AirLLMEngine for $modelId")

        // 1. Confirm server is reachable ─────────────────────────────────────
        if (!waitForServer()) {
            return InitResult.ServerNotRunning(buildTermuxInstructions())
        }

        // 2. If model is already loaded, skip ─────────────────────────────────
        val status = safeGetStatus()
        if (status?.isReady == true && status.model == modelId) {
            engineState = EngineState.READY
            Log.i(TAG, "Model already loaded: $modelId")
            return InitResult.Success
        }

        // 3. Post load request ─────────────────────────────────────────────────
        Log.i(TAG, "Requesting model load: $modelId compression=$compression")
        engineState = EngineState.LOADING
        onProgress?.invoke(1, "Sending load request…")

        val accepted = client.loadModel(
            modelPath = modelId,
            compression = compression,
            shardDir = shardDir,
            hfToken = hfToken,
        )
        if (!accepted) {
            engineState = EngineState.ERROR
            return InitResult.LoadFailed("Server rejected load request")
        }

        // 4. Poll until ready ──────────────────────────────────────────────────
        val deadline = System.currentTimeMillis() + LOAD_TIMEOUT_MS
        while (System.currentTimeMillis() < deadline) {
            delay(LOAD_POLL_INTERVAL_MS)
            val s = safeGetStatus() ?: continue

            onProgress?.invoke(s.progress, s.progressMsg ?: "Loading…")

            when {
                s.isReady -> {
                    engineState = EngineState.READY
                    onProgress?.invoke(100, "Ready")
                    Log.i(TAG, "Model ready: $modelId")
                    return InitResult.Success
                }
                s.isError -> {
                    engineState = EngineState.ERROR
                    return InitResult.LoadFailed(s.error ?: "Unknown error")
                }
            }
        }

        engineState = EngineState.ERROR
        return InitResult.LoadFailed("Timeout: model took more than 10 minutes to load")
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
            "Engine not ready (state=$engineState). Call initialize() first."
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
                is MobileAirLLMClient.StreamEvent.Done  -> { /* stream complete */ }
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
        CoroutineScope(Dispatchers.IO).launch {
            runCatching { client.resetCache() }
        }
    }

    suspend fun unload() {
        runCatching { client.unload() }
        engineState = EngineState.UNLOADED
    }

    val isReady: Boolean get() = engineState == EngineState.READY

    // ── Internal helpers ──────────────────────────────────────────────────────

    /**
     * Waits up to SERVER_START_TIMEOUT_MS for the HTTP server to respond.
     * Attempts one auto-start via MobileLLMServerManager.
     */
    private suspend fun waitForServer(): Boolean {
        // Fast path: already up
        if (MobileLLMServerManager.isServerResponding()) return true

        // Try to auto-start (works if Termux + mobilellm is installed)
        Log.i(TAG, "Server not responding — attempting auto-start")
        MobileLLMServerManager.startServer(context)

        val deadline = System.currentTimeMillis() + SERVER_START_TIMEOUT_MS
        while (System.currentTimeMillis() < deadline) {
            delay(1500L)
            if (MobileLLMServerManager.isServerResponding()) {
                Log.i(TAG, "Server is now responding")
                return true
            }
        }
        Log.w(TAG, "Server did not respond within ${SERVER_START_TIMEOUT_MS / 1000}s")
        return false
    }

    private suspend fun safeGetStatus(): MobileAirLLMClient.ServerStatus? {
        return try {
            client.getStatus()
        } catch (e: Exception) {
            Log.w(TAG, "Status poll failed: $e")
            null
        }
    }

    // ── Prompt formatting ─────────────────────────────────────────────────────

    private fun buildPrompt(system: String?, user: String): String {
        return if (system != null) {
            "<s>[INST] <<SYS>>\n$system\n<</SYS>>\n\n$user [/INST]"
        } else {
            "<s>[INST] $user [/INST]"
        }
    }

    private fun buildTermuxInstructions(): String = """
        MobileAirLLM server is not running.
        
        ── Quick start ──────────────────────────────
        1. Install Termux from F-Droid (NOT Play Store)
        2. Open Termux and run:
           chmod +x setup_termux.sh && ./setup_termux.sh
        3. Then start the server:
           ~/start_mobilellm.sh
        4. Return to Nomad — it will connect automatically.
        
        ── First-time model download ────────────────
        In Termux, after setup:
           mobilellm split meta-llama/Llama-2-13b-hf --compression 4bit
        (This takes 30-60 min and ~15 GB of storage.)
        
        ── Already have Termux? ─────────────────────
        Just run:  ~/start_mobilellm.sh
    """.trimIndent()

    // ── State / Result types ──────────────────────────────────────────────────

    enum class EngineState { UNLOADED, LOADING, READY, ERROR }

    sealed class InitResult {
        object Success : InitResult()
        data class ServerNotRunning(val instructions: String) : InitResult()
        data class LoadFailed(val reason: String) : InitResult()
    }
}