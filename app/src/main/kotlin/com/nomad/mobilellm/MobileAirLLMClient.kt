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
 * Kotlin HTTP client for the MobileAirLLM Python server on localhost.
 *
 * Fixes vs original:
 *  - ServerStatus exposes progressMsg (maps to server's progress_msg field).
 *  - getStatus() no longer throws on null body — returns safe default.
 *  - streamGenerate builds URL safely; long prompts sent as POST body instead
 *    of query string to avoid 8KB URL limit.
 *  - connectTimeout reduced to 3s (was 10s) — fail fast if server is down.
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
        .connectTimeout(3, TimeUnit.SECONDS)    // fail fast — server either up or not
        .readTimeout(600, TimeUnit.SECONDS)     // long for SSE + big model inference
        .writeTimeout(30, TimeUnit.SECONDS)
        .retryOnConnectionFailure(false)        // we handle retries ourselves
        .build()

    // ── Connection check ──────────────────────────────────────────────────────
    suspend fun isServerRunning(): Boolean = withContext(Dispatchers.IO) {
        try {
            val req = Request.Builder().url("$baseUrl/health").get().build()
            httpClient.newCall(req).execute().use { it.isSuccessful }
        } catch (_: Exception) {
            false
        }
    }

    // ── Status ────────────────────────────────────────────────────────────────
    suspend fun getStatus(): ServerStatus = withContext(Dispatchers.IO) {
        try {
            val req = Request.Builder().url("$baseUrl/status").get().build()
            httpClient.newCall(req).execute().use { resp ->
                val body = resp.body?.string() ?: "{}"
                val json = JSONObject(body)
                ServerStatus(
                    state       = json.optString("state", "unknown"),
                    model       = json.optString("model").takeIf { it.isNotEmpty() },
                    error       = json.optString("error").takeIf { it.isNotEmpty() },
                    progress    = json.optInt("progress", 0),
                    progressMsg = json.optString("progress_msg").takeIf { it.isNotEmpty() },
                )
            }
        } catch (e: Exception) {
            Log.w(TAG, "getStatus failed: $e")
            ServerStatus(state = "unknown", model = null, error = null, progress = 0, progressMsg = null)
        }
    }

    // ── Load model ────────────────────────────────────────────────────────────
    suspend fun loadModel(
        modelPath: String,
        compression: String = "4bit",
        shardDir: String? = null,
        maxRamGb: Float? = null,
        hfToken: String? = null,
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val body = JSONObject().apply {
                put("model_path", modelPath)
                put("compression", compression)
                shardDir?.let  { put("shard_dir", it) }
                maxRamGb?.let  { put("max_ram_gb", it) }
                hfToken?.let   { put("hf_token", it) }
            }
            val req = Request.Builder()
                .url("$baseUrl/load")
                .post(body.toString().toRequestBody(JSON_MEDIA_TYPE))
                .build()
            httpClient.newCall(req).execute().use { resp ->
                Log.i(TAG, "Load response: ${resp.code}")
                resp.code in 200..299
            }
        } catch (e: Exception) {
            Log.e(TAG, "loadModel failed: $e")
            false
        }
    }

    // ── Stream generate (SSE via POST to avoid URL length limit) ─────────────
    fun streamGenerate(
        prompt: String,
        maxNewTokens: Int = 256,
        temperature: Float = 0.7f,
        topP: Float = 0.9f,
        topK: Int = 40,
        resetCache: Boolean = true,
    ): Flow<StreamEvent> = flow {
        // Send prompt in POST body — avoids 8KB GET URL limit for long prompts
        val body = JSONObject().apply {
            put("prompt", prompt)
            put("max_new_tokens", maxNewTokens)
            put("temperature", temperature)
            put("top_p", topP)
            put("top_k", topK)
            put("reset_cache", resetCache)
        }

        val req = Request.Builder()
            .url("$baseUrl/stream")
            .post(body.toString().toRequestBody(JSON_MEDIA_TYPE))
            .addHeader("Accept", "text/event-stream")
            .build()

        try {
            httpClient.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) {
                    emit(StreamEvent.Error("HTTP ${resp.code}"))
                    return@use
                }
                val reader = BufferedReader(InputStreamReader(resp.body!!.byteStream()))
                for (line in reader.lineSequence()) {
                    if (!line.startsWith("data: ")) continue
                    val data = line.removePrefix("data: ").trim()
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
                        Log.w(TAG, "SSE parse error: $data")
                    }
                }
            }
        } catch (e: Exception) {
            emit(StreamEvent.Error(e.message ?: "Connection error"))
        }
    }.flowOn(Dispatchers.IO)

    // ── Non-streaming generate ────────────────────────────────────────────────
    suspend fun generate(
        prompt: String,
        maxNewTokens: Int = 256,
        temperature: Float = 0.7f,
    ): String = withContext(Dispatchers.IO) {
        try {
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
                JSONObject(resp.body?.string() ?: "{}").optString("text", "")
            }
        } catch (e: Exception) {
            Log.e(TAG, "generate failed: $e")
            ""
        }
    }

    // ── Reset / Unload ────────────────────────────────────────────────────────
    suspend fun resetCache(): Boolean = withContext(Dispatchers.IO) {
        try {
            val req = Request.Builder()
                .url("$baseUrl/reset")
                .post("{}".toRequestBody(JSON_MEDIA_TYPE))
                .build()
            httpClient.newCall(req).execute().use { it.isSuccessful }
        } catch (_: Exception) { false }
    }

    suspend fun unload(): Boolean = withContext(Dispatchers.IO) {
        try {
            val req = Request.Builder()
                .url("$baseUrl/unload")
                .post("{}".toRequestBody(JSON_MEDIA_TYPE))
                .build()
            httpClient.newCall(req).execute().use { it.isSuccessful }
        } catch (_: Exception) { false }
    }

    // ── Data classes ──────────────────────────────────────────────────────────
    data class ServerStatus(
        val state: String,
        val model: String?,
        val error: String?,
        val progress: Int,
        val progressMsg: String?,
    ) {
        val isReady: Boolean   get() = state == "ready"
        val isLoading: Boolean get() = state == "loading"
        val isError: Boolean   get() = state == "error"
    }

    sealed class StreamEvent {
        data class Token(val text: String) : StreamEvent()
        object Done : StreamEvent()
        data class Error(val message: String) : StreamEvent()
    }
}