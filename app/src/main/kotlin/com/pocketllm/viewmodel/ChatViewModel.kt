package com.pocketllm.viewmodel

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.PerformanceHintManager
import android.os.Process
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.pocketllm.data.*
import com.pocketllm.engine.LlamaEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class ChatUiState(
    val messages: List<ChatMessage>    = emptyList(),
    val isGenerating: Boolean          = false,
    val loadedModel: ModelInfo?        = null,
    val loadingModel: Boolean          = false,
    val loadProgress: Float            = 0f,
    val error: String?                 = null,
    val tokensPerSec: Float            = 0f,
    val pendingAttachment: Attachment? = null,
    val isProcessingFile: Boolean      = false
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
    private val downloadJobs = mutableMapOf<String, Job>()

    // ── Android PerformanceHintManager (API 31+) ──────────────────────────────
    // Tells the power governor to sustain high CPU clocks during inference.
    // Without this, Android throttles after ~2s — a major cause of slowdowns.
    private var perfHintSession: Any? = null

    init {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            try {
                val mgr = app.getSystemService(PerformanceHintManager::class.java)
                if (mgr != null) {
                    val tids = intArrayOf(Process.myTid())
                    perfHintSession = mgr.createHintSession(tids, 100_000_000L)
                }
            } catch (_: Exception) {}
        }
        detectRamAndAutoLoad()
    }

    @Suppress("UNCHECKED_CAST")
    private fun boostCpuPerf() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            try {
                (perfHintSession as? android.os.PerformanceHintManager.Session)
                    ?.reportActualWorkDuration(50_000_000L)
            } catch (_: Exception) {}
        }
    }

    private fun detectRamAndAutoLoad() {
        val am   = getApplication<Application>().getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo()
        am.getMemoryInfo(info)
        val ramGb = info.totalMem / 1_073_741_824.0
        _models.update { it.copy(totalDeviceRamGb = ramGb) }

        // Auto-load best already-downloaded model
        viewModelScope.launch(Dispatchers.IO) {
            val downloaded = modelManager.listDownloaded()
            if (downloaded.isNotEmpty()) {
                val best = ModelCatalog.models
                    .filter { m -> downloaded.any { it.name == m.filename } }
                    .filter { m -> m.ramRequirementGb <= ramGb * 0.85 }
                    .maxByOrNull { it.paramCountValue }
                best?.let { loadModel(it) }
            }
        }
    }

    // ── File attachment ───────────────────────────────────────────────────────
    fun attachFile(uri: Uri) {
        viewModelScope.launch {
            _chat.update { it.copy(isProcessingFile = true, error = null) }
            try {
                val att = FileProcessor.process(getApplication(), uri)
                _chat.update { it.copy(pendingAttachment = att, isProcessingFile = false) }
            } catch (e: Exception) {
                _chat.update { it.copy(isProcessingFile = false,
                    error = "Can't read file: ${e.message}") }
            }
        }
    }

    fun clearAttachment() = _chat.update { it.copy(pendingAttachment = null) }

    // ── Model management ─────────────────────────────────────────────────────
    fun loadModel(model: ModelInfo) {
        viewModelScope.launch {
            _chat.update { it.copy(loadingModel = true, loadProgress = 0f, error = null) }
            val file = modelManager.getLocalFile(model)
            if (!file.exists()) {
                _chat.update { it.copy(loadingModel = false,
                    error = "File not found — download first.") }
                return@launch
            }
            LlamaEngine.load(file, LlamaEngine.DEFAULT_CTX, LlamaEngine.DEFAULT_THREADS, true) { p ->
                _chat.update { it.copy(loadProgress = p) }
            }.fold(
                onSuccess = {
                    _chat.update { it.copy(loadingModel = false, loadedModel = model, loadProgress = 1f) }
                    _models.update { it.copy(loadedModelId = model.id) }
                    addSysMsg("✅ ${model.name} ready!  (system prefix cached — fast responses)")
                },
                onFailure = { e ->
                    _chat.update { it.copy(loadingModel = false,
                        error = "Load failed: ${e.message}") }
                }
            )
        }
    }

    fun unloadModel() {
        viewModelScope.launch(Dispatchers.IO) {
            LlamaEngine.unloadModel()
            _chat.update { it.copy(loadedModel = null, tokensPerSec = 0f) }
            _models.update { it.copy(loadedModelId = null) }
            history.clear()
        }
    }

    fun downloadModel(model: ModelInfo) {
        val job = viewModelScope.launch {
            modelManager.download(model).collect { state ->
                _models.update {
                    it.copy(downloadStates = it.downloadStates + (model.id to state))
                }
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
                    loadedModelId  = if (it.loadedModelId == model.id) null else it.loadedModelId
                )
            }
        }
    }

    // ── Send message ──────────────────────────────────────────────────────────
    fun sendMessage(userText: String) {
        if (userText.isBlank() || _chat.value.isGenerating) return
        if (!LlamaEngine.isModelLoaded()) {
            addSysMsg("⚠️ No model loaded — go to Models tab.")
            return
        }

        val attachment  = _chat.value.pendingAttachment
        val displayText = if (attachment != null) "$userText\n📎 ${attachment.fileName}" else userText

        // ── File context with strong instructions ─────────────────────────────
        // Prevents hallucination: model is told explicitly what it has and hasn't seen.
        val fileCtx = if (attachment != null) {
            when (attachment.type) {
                AttachmentType.IMAGE -> buildString {
                    // Image: model cannot see pixels — be honest about this
                    append("[IMAGE ATTACHED: ${attachment.fileName}]\n")
                    append("The user attached an image. You CANNOT see the actual image — ")
                    append("this model does not support vision. Honestly tell the user you ")
                    append("cannot analyze the image content and suggest they describe it.")
                }
                else -> {
                    val text = attachment.extractedText ?: ""
                    if (text.isBlank() || text.startsWith("Could not")) {
                        // Empty / failed extraction — be honest
                        "[FILE: ${attachment.fileName}]\nContent could not be extracted. " +
                                "Tell the user you were unable to read this file."
                    } else {
                        // Real extracted content — include it fully
                        buildString {
                            append("[FILE: ${attachment.fileName} — ${FileProcessor.formatSize(attachment.fileSizeBytes)}]\n")
                            append("FULL CONTENT BELOW — answer questions about THIS content only:\n\n")
                            append(text.take(3500))
                            if (text.length > 3500) append("\n[... file truncated at 3500 chars]")
                        }
                    }
                }
            }
        } else ""

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
                boostCpuPerf()  // request sustained high CPU clocks

                val prompt = LlamaEngine.buildChatPrompt(
                    modelName    = _chat.value.loadedModel?.name ?: "",
                    history      = history.toList(),
                    userMessage  = userText.trim(),
                    fileContext  = fileCtx
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

                val tps       = tokenCount * 1000f /
                        (System.currentTimeMillis() - t0).coerceAtLeast(1)
                val finalText = sb.toString().trim()

                if (history.size >= 8) history.removeFirst()
                history.addLast(Pair(userText.trim(), finalText))

                _chat.update { state ->
                    val msgs = state.messages.toMutableList()
                    val idx  = msgs.lastIndex
                    if (idx >= 0 && msgs[idx].role == Role.ASSISTANT)
                        msgs[idx] = msgs[idx].copy(
                            content     = finalText.ifEmpty { "(No response)" },
                            isStreaming = false
                        )
                    state.copy(isGenerating = false, tokensPerSec = tps)
                }

            } catch (e: Exception) {
                _chat.update { state ->
                    val msgs = state.messages.toMutableList()
                    val idx  = msgs.lastIndex
                    if (idx >= 0 && msgs[idx].role == Role.ASSISTANT)
                        msgs[idx] = msgs[idx].copy(
                            content = "Error: ${e.message}", isStreaming = false)
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
        // Re-establish prefix cache after clear so next message is still fast
        viewModelScope.launch(Dispatchers.IO) {
            val prefix = LlamaEngine.buildSystemPrefix()
            LlamaEngine.cacheSystemPrompt(prefix)
        }
    }

    private fun addSysMsg(t: String) {
        _chat.update { it.copy(messages = it.messages +
                ChatMessage(role = Role.SYSTEM, content = t)) }
    }

    override fun onCleared() {
        super.onCleared()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            try {
                (perfHintSession as? android.os.PerformanceHintManager.Session)?.close()
            } catch (_: Exception) {}
        }
    }
}