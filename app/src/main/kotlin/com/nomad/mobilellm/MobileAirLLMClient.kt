package com.nomad.mobilellm

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.URLEncoder
import java.util.concurrent.TimeUnit

/**
 * MobileAirLLMClient
 *
 * Kotlin client that connects the Nomad Android app to the Python
 * MobileAirLLM server running in Termux on the same device.
 *
 * Drop-in replacement for llama.cpp JNI calls for large models (>8B).
 *
 * Usage in your ViewModel / Repository:
 *   val client = MobileAirLLMClient()
 *   client.loadModel("meta-llama/Llama-2-30b-hf", compression = "4bit")
 *   client.streamGenerate(prompt).collect { token -> appendToChat(token) }
 */
class MobileAirLLMClient(
    private val host: String = "127.0.0.1",
    private val port: Int = 8765,
) {
    companion object {
        private const val TAG = "MobileAirLLM"
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
    }

    private val baseUrl = "http://$host:$port"

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(300, TimeUnit.SECONDS)   // long for SSE streams
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    // ── Connection check ──────────────────────────────────────────────────────
    suspend fun isServerRunning(): Boolean = withContext(Dispatchers.IO) {
        try {
            val req = Request.Builder().url("$baseUrl/health").get().build()
            httpClient.newCall(req).execute().use { it.isSuccessful }
        } catch (e: Exception) {
            false
        }
    }

    // ── Server status ─────────────────────────────────────────────────────────
    suspend fun getStatus(): ServerStatus = withContext(Dispatchers.IO) {
        val req = Request.Builder().url("$baseUrl/status").get().build()
        httpClient.newCall(req).execute().use { resp ->
            val json = JSONObject(resp.body?.string() ?: "{}")
            ServerStatus(
                state    = json.optString("state", "unknown"),
                model    = json.optString("model", null),
                error    = json.optString("error", null),
                progress = json.optInt("progress", 0),
            )
        }
    }

    // ── Load a model ──────────────────────────────────────────────────────────
    suspend fun loadModel(
        modelPath: String,
        compression: String = "4bit",
        shardDir: String? = null,
        maxRamGb: Float? = null,
        hfToken: String? = null,
    ): Boolean = withContext(Dispatchers.IO) {
        val body = JSONObject().apply {
            put("model_path", modelPath)
            put("compression", compression)
            shardDir?.let { put("shard_dir", it) }
            maxRamGb?.let { put("max_ram_gb", it) }
            hfToken?.let { put("hf_token", it) }
        }
        val req = Request.Builder()
            .url("$baseUrl/load")
            .post(body.toString().toRequestBody(JSON_MEDIA_TYPE))
            .build()

        httpClient.newCall(req).execute().use { resp ->
            Log.i(TAG, "Load response: ${resp.code} ${resp.body?.string()}")
            resp.code in 200..299
        }
    }

    // ── Stream generate (SSE) ─────────────────────────────────────────────────
    fun streamGenerate(
        prompt: String,
        maxNewTokens: Int = 256,
        temperature: Float = 0.7f,
        topP: Float = 0.9f,
        topK: Int = 40,
        resetCache: Boolean = true,
    ): Flow<StreamEvent> = flow {
        val encodedPrompt = URLEncoder.encode(prompt, "UTF-8")
        val url = "$baseUrl/stream" +
                "?prompt=$encodedPrompt" +
                "&max_new_tokens=$maxNewTokens" +
                "&temperature=$temperature" +
                "&top_p=$topP" +
                "&top_k=$topK" +
                "&reset_cache=$resetCache"

        val req = Request.Builder().url(url).get()
            .addHeader("Accept", "text/event-stream")
            .build()

        httpClient.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) {
                emit(StreamEvent.Error("HTTP ${resp.code}"))
                return@use
            }
            val reader = BufferedReader(InputStreamReader(resp.body!!.byteStream()))
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                val ln = line ?: continue
                if (ln.startsWith("data: ")) {
                    val data = ln.removePrefix("data: ").trim()
                    if (data == "[DONE]") {
                        emit(StreamEvent.Done)
                        break
                    }
                    try {
                        val json = JSONObject(data)
                        if (json.has("error")) {
                            emit(StreamEvent.Error(json.getString("error")))
                            break
                        }
                        emit(StreamEvent.Token(json.getString("token")))
                    } catch (e: Exception) {
                        Log.w(TAG, "Parse error: $data")
                    }
                }
            }
        }
    }.flowOn(Dispatchers.IO)

    // ── Non-streaming generate ────────────────────────────────────────────────
    suspend fun generate(
        prompt: String,
        maxNewTokens: Int = 256,
        temperature: Float = 0.7f,
    ): String = withContext(Dispatchers.IO) {
        val body = JSONObject().apply {
            put("prompt", prompt)
            put("max_new_tokens", maxNewTokens)
            put("temperature", temperature)
        }
        val req = Request.Builder()
            .url("$baseUrl/generate")
            .post(body.toString().toRequestBody(JSON_MEDIA_TYPE))
            .build()

        httpClient.newCall(req).execute().use { resp ->
            val json = JSONObject(resp.body?.string() ?: "{}")
            json.optString("text", "")
        }
    }

    // ── Reset KV cache (new conversation) ─────────────────────────────────────
    suspend fun resetCache(): Boolean = withContext(Dispatchers.IO) {
        val req = Request.Builder()
            .url("$baseUrl/reset")
            .post("{}".toRequestBody(JSON_MEDIA_TYPE))
            .build()
        httpClient.newCall(req).execute().use { it.isSuccessful }
    }

    // ── Unload model ──────────────────────────────────────────────────────────
    suspend fun unload(): Boolean = withContext(Dispatchers.IO) {
        val req = Request.Builder()
            .url("$baseUrl/unload")
            .post("{}".toRequestBody(JSON_MEDIA_TYPE))
            .build()
        httpClient.newCall(req).execute().use { it.isSuccessful }
    }

    // ── Data classes ──────────────────────────────────────────────────────────
    data class ServerStatus(
        val state: String,
        val model: String?,
        val error: String?,
        val progress: Int,
    ) {
        val isReady: Boolean get() = state == "ready"
        val isLoading: Boolean get() = state == "loading"
        val isError: Boolean get() = state == "error"
    }

    sealed class StreamEvent {
        data class Token(val text: String) : StreamEvent()
        object Done : StreamEvent()
        data class Error(val message: String) : StreamEvent()
    }
}
