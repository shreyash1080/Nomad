package com.pocketllm.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.pocketllm.data.*
import com.pocketllm.engine.LlamaEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.io.File

// ── UI state ─────────────────────────────────────────────────────────────────

data class ChatUiState(
    val messages: List<ChatMessage>   = emptyList(),
    val isGenerating: Boolean         = false,
    val loadedModel: ModelInfo?       = null,
    val loadingModel: Boolean         = false,
    val loadProgress: Float           = 0f,
    val error: String?                = null
)

data class ModelsUiState(
    val models: List<ModelInfo>                       = ModelCatalog.models,
    val downloadStates: Map<String, DownloadState>    = emptyMap(),
    val loadedModelId: String?                        = null
)

// ─────────────────────────────────────────────────────────────────────────────

class ChatViewModel(app: Application) : AndroidViewModel(app) {

    val modelManager = ModelManager(app)

    private val _chatState  = MutableStateFlow(ChatUiState())
    val chatState: StateFlow<ChatUiState> = _chatState.asStateFlow()

    private val _modelsState = MutableStateFlow(ModelsUiState())
    val modelsState: StateFlow<ModelsUiState> = _modelsState.asStateFlow()

    private var genJob: Job? = null
    private val downloadJobs = mutableMapOf<String, Job>()
    private val conversationHistory = mutableListOf<Pair<String, String>>()

    // System prompt
    var systemPrompt: String = "You are a helpful AI assistant running locally on this device. " +
            "Be concise, accurate, and friendly."

    // ── Model management ─────────────────────────────────────────────────────

    fun loadModel(model: ModelInfo) {
        // RAM Check
        val availableRamGb = getAvailableRamGb()
        val requiredRamGb = model.ramRequired.replace(" GB RAM", "").toDoubleOrNull() ?: 0.0
        
        if (availableRamGb < requiredRamGb) {
            _chatState.update { it.copy(error = "Insufficient RAM: This model requires ${model.ramRequired}, but only ${String.format("%.1f", availableRamGb)} GB is available.") }
            return
        }

        viewModelScope.launch {
            _chatState.update { it.copy(loadingModel = true, loadProgress = 0f, error = null) }

            val file = modelManager.getLocalFile(model)
            if (!file.exists()) {
                _chatState.update { it.copy(loadingModel = false, error = "Model file not found. Please download it first.") }
                return@launch
            }

            LlamaEngine.load(
                file       = file,
                contextSize = 4096,
                useGpu      = true,
                onProgress  = { p -> _chatState.update { it.copy(loadProgress = p) } }
            ).fold(
                onSuccess = {
                    _chatState.update { it.copy(loadingModel = false, loadedModel = model, loadProgress = 1f) }
                    _modelsState.update { it.copy(loadedModelId = model.id) }
                    addSystemMessage("✅ ${model.name} loaded and ready!")
                },
                onFailure = { e ->
                    _chatState.update { it.copy(loadingModel = false, error = "Failed to load: ${e.message}") }
                }
            )
        }
    }

    fun unloadModel() {
        viewModelScope.launch(Dispatchers.IO) {
            LlamaEngine.unloadModel()
            _chatState.update { it.copy(loadedModel = null) }
            _modelsState.update { it.copy(loadedModelId = null) }
        }
    }

    // ── Download ─────────────────────────────────────────────────────────────

    fun downloadModel(model: ModelInfo) {
        // Cancel existing job for this model if any
        downloadJobs[model.id]?.cancel()

        val job = viewModelScope.launch {
            modelManager.download(model).collect { state ->
                _modelsState.update {
                    it.copy(downloadStates = it.downloadStates + (model.id to state))
                }
                if (state is DownloadState.Done) {
                    downloadJobs.remove(model.id)
                    // autoload after download
                    loadModel(model)
                }
                if (state is DownloadState.Error) {
                    downloadJobs.remove(model.id)
                }
            }
        }
        downloadJobs[model.id] = job
    }

    fun cancelDownload(model: ModelInfo) {
        downloadJobs[model.id]?.cancel()
        downloadJobs.remove(model.id)
        _modelsState.update {
            it.copy(downloadStates = it.downloadStates - model.id)
        }
    }

    fun deleteModel(model: ModelInfo) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                // If the model is currently loaded, unload it first
                if (_chatState.value.loadedModel?.id == model.id) {
                    LlamaEngine.unloadModel()
                    _chatState.update { it.copy(loadedModel = null) }
                }

                // Ensure any active download for this model is stopped
                downloadJobs[model.id]?.cancel()
                downloadJobs.remove(model.id)

                val success = modelManager.delete(model)
                if (success) {
                    _modelsState.update {
                        it.copy(
                            downloadStates = it.downloadStates - model.id,
                            loadedModelId  = if (it.loadedModelId == model.id) null else it.loadedModelId
                        )
                    }
                } else {
                    _chatState.update { it.copy(error = "Failed to delete ${model.name}. File might be in use or doesn't exist.") }
                }
            } catch (e: Exception) {
                _chatState.update { it.copy(error = "Error during deletion: ${e.message}") }
            }
        }
    }

    // ── Chat ─────────────────────────────────────────────────────────────────

    fun sendMessage(userText: String) {
        if (userText.isBlank() || _chatState.value.isGenerating) return
        if (!LlamaEngine.isModelLoaded()) {
            addSystemMessage("⚠️ No model loaded. Go to Models tab to download and load one.")
            return
        }

        val userMsg = ChatMessage(role = Role.USER, content = userText.trim())
        // Generate a stable ID for the assistant message so LazyColumn keys don't collide during stream
        val assistantId = java.util.UUID.randomUUID().toString()
        val assistantMsg = ChatMessage(id = assistantId, role = Role.ASSISTANT, content = "", isStreaming = true)

        _chatState.update {
            it.copy(
                messages     = it.messages + userMsg + assistantMsg,
                isGenerating = true,
                error        = null
            )
        }

        genJob = viewModelScope.launch {
            val startTime = System.currentTimeMillis()
            var firstTokenTime = -1L

            val prompt = LlamaEngine.buildChatPrompt(
                systemPrompt = systemPrompt,
                history      = conversationHistory,
                userMessage  = userText.trim()
            )

            val sb = StringBuilder()

            LlamaEngine.generateFlow(
                prompt      = prompt,
                maxTokens   = 512,
                temperature = 0.7f,
                topP        = 0.9f
            ).collect { token ->
                if (firstTokenTime == -1L) {
                    firstTokenTime = System.currentTimeMillis()
                    val latency = firstTokenTime - startTime
                    android.util.Log.i("ChatPerformance", "Time to first token: ${latency}ms")
                    if (latency > 10000) {
                        android.util.Log.w("ChatPerformance", "LATENCY WARNING: > 10s!")
                    }
                }

                sb.append(token)
                // Update the last (streaming) message
                _chatState.update { state ->
                    val updated = state.messages.toMutableList()
                    val lastIdx = updated.lastIndex
                    if (lastIdx >= 0 && updated[lastIdx].role == Role.ASSISTANT) {
                        updated[lastIdx] = updated[lastIdx].copy(content = sb.toString())
                    }
                    state.copy(messages = updated)
                }
            }

            // Mark streaming done
            val finalText = sb.toString()
            conversationHistory.add(Pair(userText, finalText))

            _chatState.update { state ->
                val updated = state.messages.toMutableList()
                val lastIdx = updated.lastIndex
                if (lastIdx >= 0 && updated[lastIdx].role == Role.ASSISTANT) {
                    updated[lastIdx] = updated[lastIdx].copy(
                        content     = finalText,
                        isStreaming = false
                    )
                }
                state.copy(messages = updated, isGenerating = false)
            }
        }
    }

    fun stopGeneration() {
        LlamaEngine.stopGeneration()
        genJob?.cancel()
        _chatState.update { it.copy(isGenerating = false) }
    }

    fun clearChat() {
        conversationHistory.clear()
        _chatState.update { it.copy(messages = emptyList()) }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private fun getAvailableRamGb(): Double {
        return try {
            val activityManager = getApplication<Application>().getSystemService(android.content.Context.ACTIVITY_SERVICE) as android.app.ActivityManager
            val memoryInfo = android.app.ActivityManager.MemoryInfo()
            activityManager.getMemoryInfo(memoryInfo)
            memoryInfo.totalMem.toDouble() / (1024 * 1024 * 1024)
        } catch (e: Exception) {
            4.0 // Fallback to a safe middle ground if check fails
        }
    }

    private fun addSystemMessage(text: String) {
        _chatState.update {
            it.copy(messages = it.messages + ChatMessage(id = java.util.UUID.randomUUID().toString(), role = Role.SYSTEM, content = text))
        }
    }
}
