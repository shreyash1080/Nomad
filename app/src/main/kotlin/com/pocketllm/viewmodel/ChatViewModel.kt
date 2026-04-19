package com.pocketllm.viewmodel

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.pocketllm.data.*
import com.pocketllm.engine.LlamaEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

data class ChatUiState(
    val messages: List<ChatMessage>   = emptyList(),
    val isGenerating: Boolean         = false,
    val loadedModel: ModelInfo?       = null,
    val loadingModel: Boolean         = false,
    val loadProgress: Float           = 0f,
    val error: String?                = null,
    val tokensPerSec: Float           = 0f,
    val pendingAttachment: Attachment? = null,
    val isProcessingFile: Boolean     = false
)

data class ModelsUiState(
    val models: List<ModelInfo>                    = ModelCatalog.models,
    val downloadStates: Map<String, DownloadState> = emptyMap(),
    val loadedModelId: String?                     = null,
    val totalDeviceRamGb: Double                   = 0.0
)

class ChatViewModel(app: Application) : AndroidViewModel(app) {

    val modelManager = ModelManager(app)
    private val _chat   = MutableStateFlow(ChatUiState())
    val chatState: StateFlow<ChatUiState> = _chat.asStateFlow()
    private val _models = MutableStateFlow(ModelsUiState())
    val modelsState: StateFlow<ModelsUiState> = _models.asStateFlow()

    private var genJob: Job? = null
    private val history = ArrayDeque<Pair<String, String>>(8)

    var systemPrompt = "You are a helpful AI assistant. Answer clearly and concisely."

    init {
        detectRamAndAutoLoad()
    }

    private fun detectRamAndAutoLoad() {
        val activityManager = getApplication<Application>().getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        val totalRamGb = memoryInfo.totalMem / 1_073_741_824.0
        
        _models.update { it.copy(totalDeviceRamGb = totalRamGb) }

        // Auto-load best downloaded model
        viewModelScope.launch(Dispatchers.IO) {
            val downloaded = modelManager.listDownloaded()
            if (downloaded.isNotEmpty()) {
                val bestModel = ModelCatalog.models
                    .filter { model -> downloaded.any { it.name == model.filename } }
                    .filter { it.ramRequirementGb <= totalRamGb * 0.85 }
                    .maxByOrNull { it.paramCountValue }
                
                bestModel?.let { loadModel(it) }
            }
        }
    }

    // ── File attachment ───────────────────────────────────────────────────────

    fun attachFile(uri: Uri) {
        viewModelScope.launch {
            _chat.update { it.copy(isProcessingFile = true, error = null) }
            try {
                val attachment = FileProcessor.process(getApplication(), uri)
                _chat.update { it.copy(pendingAttachment = attachment, isProcessingFile = false) }
            } catch (e: Exception) {
                _chat.update { it.copy(isProcessingFile = false, error = "Could not read file: ${e.message}") }
            }
        }
    }

    fun clearAttachment() {
        _chat.update { it.copy(pendingAttachment = null) }
    }

    // ── Model management ─────────────────────────────────────────────────────

    fun loadModel(model: ModelInfo) {
        viewModelScope.launch {
            _chat.update { it.copy(loadingModel = true, loadProgress = 0f, error = null) }
            val file = modelManager.getLocalFile(model)
            if (!file.exists()) {
                _chat.update { it.copy(loadingModel = false, error = "File not found — download first.") }
                return@launch
            }
            LlamaEngine.load(file, LlamaEngine.DEFAULT_CTX, LlamaEngine.DEFAULT_THREADS, true) { p ->
                _chat.update { it.copy(loadProgress = p) }
            }.fold(
                onSuccess = {
                    _chat.update { it.copy(loadingModel = false, loadedModel = model, loadProgress = 1f) }
                    _models.update { it.copy(loadedModelId = model.id) }
                    addSysMsg("✅ ${model.name} ready!")
                },
                onFailure = { e ->
                    _chat.update { it.copy(loadingModel = false, error = "Load failed: ${e.message}") }
                }
            )
        }
    }

    fun unloadModel() {
        viewModelScope.launch(Dispatchers.IO) {
            LlamaEngine.unloadModel()
            _chat.update { it.copy(loadedModel = null, tokensPerSec = 0f) }
            _models.update { it.copy(loadedModelId = null) }
        }
    }

    private val downloadJobs = mutableMapOf<String, Job>()

    fun downloadModel(model: ModelInfo) {
        val job = viewModelScope.launch {
            modelManager.download(model).collect { state ->
                _models.update { it.copy(downloadStates = it.downloadStates + (model.id to state)) }
                if (state is DownloadState.Done) {
                    downloadJobs.remove(model.id)
                    loadModel(model)
                }
            }
        }
        downloadJobs[model.id] = job
    }

    fun cancelDownload(modelId: String) {
        downloadJobs[modelId]?.cancel()
        downloadJobs.remove(modelId)
        _models.update { it.copy(downloadStates = it.downloadStates - modelId) }
    }

    fun deleteModel(model: ModelInfo) {
        viewModelScope.launch(Dispatchers.IO) {
            if (_chat.value.loadedModel?.id == model.id) {
                LlamaEngine.unloadModel()
                _chat.update { it.copy(loadedModel = null) }
            }
            modelManager.delete(model)
            _models.update {
                it.copy(
                    downloadStates = it.downloadStates - model.id,
                    loadedModelId = if (it.loadedModelId == model.id) null else it.loadedModelId
                )
            }
        }
    }

    // ── Chat ──────────────────────────────────────────────────────────────────

    fun sendMessage(userText: String) {
        if (userText.isBlank() || _chat.value.isGenerating) return
        if (!LlamaEngine.isModelLoaded()) {
            addSysMsg("⚠️ No model loaded. Go to Models tab first.")
            return
        }

        val attachment = _chat.value.pendingAttachment
        val displayText = if (attachment != null) "$userText\n📎 ${attachment.fileName}" else userText
        val fileContext = attachment?.extractedText ?: ""

        val userMsg = ChatMessage(role = Role.USER, content = displayText, attachment = attachment)
        val asstMsg = ChatMessage(role = Role.ASSISTANT, content = "...", isStreaming = true)

        _chat.update {
            it.copy(
                messages          = it.messages + userMsg + asstMsg,
                isGenerating      = true,
                pendingAttachment = null,
                error             = null
            )
        }

        genJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                val prompt = LlamaEngine.buildChatPrompt(
                    modelName     = _chat.value.loadedModel?.name ?: "",
                    systemPrompt  = systemPrompt,
                    history       = history.toList(),
                    userMessage   = userText.trim(),
                    fileContext   = fileContext
                )

                val sb         = StringBuilder()
                val t0         = System.currentTimeMillis()
                var tokenCount = 0

                LlamaEngine.generateFlow(
                    prompt      = prompt,
                    maxTokens   = LlamaEngine.DEFAULT_MAXTOK,
                    temperature = 0.7f,
                    topP        = 0.9f
                ).collect { token ->
                    sb.append(token)
                    tokenCount++
                    _chat.update { state ->
                        val msgs = state.messages.toMutableList()
                        val idx  = msgs.lastIndex
                        if (idx >= 0 && msgs[idx].role == Role.ASSISTANT)
                            msgs[idx] = msgs[idx].copy(content = sb.toString())
                        state.copy(messages = msgs)
                    }
                }

                val tps = tokenCount * 1000f / (System.currentTimeMillis() - t0).coerceAtLeast(1)
                val finalText = sb.toString().trim()

                if (history.size >= 8) history.removeFirst()
                history.addLast(Pair(userText.trim(), finalText))

                _chat.update { state ->
                    val msgs = state.messages.toMutableList()
                    val idx  = msgs.lastIndex
                    if (idx >= 0 && msgs[idx].role == Role.ASSISTANT)
                        msgs[idx] = msgs[idx].copy(content = if (finalText.isEmpty()) "(No response)" else finalText, isStreaming = false)
                    state.copy(isGenerating = false, tokensPerSec = tps)
                }
            } catch (e: Exception) {
                _chat.update { state ->
                    val msgs = state.messages.toMutableList()
                    val idx  = msgs.lastIndex
                    if (idx >= 0 && msgs[idx].role == Role.ASSISTANT)
                        msgs[idx] = msgs[idx].copy(content = "Error during generation: ${e.message}", isStreaming = false)
                    state.copy(isGenerating = false, error = e.message)
                }
            }
        }
    }

    fun stopGeneration() {
        LlamaEngine.stopGeneration()
        genJob?.cancel()
        _chat.update { it.copy(isGenerating = false) }
    }

    fun clearChat() {
        history.clear()
        _chat.update { it.copy(messages = emptyList(), pendingAttachment = null, tokensPerSec = 0f) }
    }

    private fun addSysMsg(t: String) {
        _chat.update { it.copy(messages = it.messages + ChatMessage(role = Role.SYSTEM, content = t)) }
    }
}
