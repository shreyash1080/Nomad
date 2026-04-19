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
import java.util.UUID

data class ChatUiState(
    val messages: List<ChatMessage>    = emptyList(),
    val isGenerating: Boolean          = false,
    val loadedModel: ModelInfo?        = null,
    val loadingModel: Boolean          = false,
    val loadProgress: Float            = 0f,
    val error: String?                 = null,
    val tokensPerSec: Float            = 0f,
    val pendingAttachment: Attachment? = null,
    val isProcessingFile: Boolean      = false,
    val userName: String               = "sac",
    val systemPrompt: String           = LlamaEngine.DEFAULT_SYSTEM,
    val useGpu: Boolean                = true,
    val saveChat: Boolean              = true,
    val temperature: Float             = 0.7f,
    val maxTokens: Int                 = 300,
    val isThinkingMode: Boolean        = false,
    val preferredLanguage: String      = "English",
    val chatHistory: List<ChatSession> = emptyList(),
    val currentSessionId: String?      = null,
    val isFirstLaunch: Boolean         = true
)

data class ModelsUiState(
    val models: List<ModelInfo>                    = ModelCatalog.models,
    val downloadStates: Map<String, DownloadState> = emptyMap(),
    val loadedModelId: String?                     = null,
    val totalDeviceRamGb: Double                   = 0.0
)

class ChatViewModel(app: Application) : AndroidViewModel(app) {

    val modelManager = ModelManager(app)
    private val historyManager = ChatHistoryManager(app)

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
    private var perfHintSession: android.os.PerformanceHintManager.Session? = null

    init {
        val prefs = app.getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
        val isFirst = prefs.getBoolean("is_first_launch", true)
        _chat.update { it.copy(isFirstLaunch = isFirst) }

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
        loadChatHistory()
    }

    @Suppress("UNCHECKED_CAST")
    private fun boostCpuPerf() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            try {
                perfHintSession?.reportActualWorkDuration(50_000_000L)
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
            LlamaEngine.load(
                file = file,
                systemPrompt = _chat.value.systemPrompt,
                useGpu = _chat.value.useGpu,
                onProgress = { p -> _chat.update { it.copy(loadProgress = p) } }
            ).fold(
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

    fun deleteMessage(id: Long) {
        _chat.update { it.copy(messages = it.messages.filter { m -> m.id != id }) }
        if (_chat.value.messages.isEmpty()) clearChat()
    }

    fun editMessage(id: Long, newContent: String) {
        _chat.update { state ->
            state.copy(messages = state.messages.map { 
                if (it.id == id) it.copy(content = newContent) else it 
            })
        }
    }

    fun setLanguage(lang: String) {
        _chat.update { it.copy(preferredLanguage = lang) }
    }

    fun resetFirstLaunch() {
        _chat.update { it.copy(isFirstLaunch = true) }
    }

    fun setFirstLaunchSeen() {
        _chat.update { it.copy(isFirstLaunch = false) }
        val prefs = getApplication<Application>().getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
        prefs.edit().putBoolean("is_first_launch", false).apply()
    }

    // ── Send message ──────────────────────────────────────────────────────────
    fun sendMessage(userText: String) {
        val attachment = _chat.value.pendingAttachment
        if ((userText.isBlank() && attachment == null) || _chat.value.isGenerating) return
        if (!LlamaEngine.isModelLoaded()) {
            addSysMsg("⚠️ No model loaded — go to Models tab.")
            return
        }

        val displayText = if (attachment != null) {
            if (userText.isNotBlank()) "$userText\n📎 ${attachment.fileName}"
            else "Analyze this file: ${attachment.fileName}"
        } else userText

        // ── File context with strong instructions ─────────────────────────────
        // Prevents hallucination: model is told explicitly what it has and hasn't seen.
        val fileCtx = if (attachment != null) {
            when (attachment.type) {
                AttachmentType.IMAGE -> buildString {
                    val text = attachment.extractedText ?: ""
                    if (text.isNotBlank() && !text.contains("No text detected")) {
                        append("[IMAGE: ${attachment.fileName}]\n")
                        append("The user attached an image. Here is the text extracted from it via OCR:\n\n")
                        append(text)
                        if (userText.isBlank()) {
                            append("\n\nAction: The user uploaded this image without a message. Analyze/Summarize the text above.")
                        } else {
                            append("\n\nUser Question: $userText")
                        }
                    } else {
                        append("[IMAGE ATTACHED: ${attachment.fileName}]\n")
                        append("The user attached an image, but NO text was detected. ")
                        append("You CANNOT see pixels. Tell the user you can't read this specific image and ask for a description.")
                    }
                }
                else -> {
                    val text = attachment.extractedText ?: ""
                    if (text.isBlank() || text.startsWith("Could not")) {
                        "[FILE: ${attachment.fileName}]\nContent could not be extracted."
                    } else {
                        buildString {
                            append("[FILE: ${attachment.fileName}]\n")
                            append("CONTENT:\n$text\n\n")
                            if (userText.isBlank()) {
                                append("Action: The user uploaded this file without a message. Summarize it.")
                            } else {
                                append("User Question: $userText")
                            }
                        }
                    }
                }
            }
        } else null

        val userMsg = ChatMessage(
            id      = System.nanoTime(),
            role    = Role.USER,
            content = displayText,
            attachment = attachment
        )
        val asstMsg = ChatMessage(
            id      = System.nanoTime() + 1,
            role    = Role.ASSISTANT,
            content = "...",
            isStreaming = true
        )

        _chat.update { it.copy(
            messages = it.messages + userMsg + asstMsg,
            isGenerating = true,
            pendingAttachment = null
        ) }

        genJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                boostCpuPerf()  // request sustained high CPU clocks

                val prompt = LlamaEngine.buildChatPrompt(
                    modelName    = _chat.value.loadedModel?.name ?: "",
                    systemPrompt = if (_chat.value.preferredLanguage == "English") _chat.value.systemPrompt 
                                   else "${_chat.value.systemPrompt} You MUST respond entirely in ${_chat.value.preferredLanguage}.",
                    history      = if (_chat.value.saveChat) history.toList() else emptyList(),
                    userMessage  = userText.ifBlank { displayText },
                    fileContext  = fileCtx ?: ""
                )

                val sb         = StringBuilder()
                val thoughtSb  = StringBuilder()
                var isThinking = false
                val t0         = System.currentTimeMillis()
                var tokenCount = 0

                LlamaEngine.generateFlow(
                    prompt      = prompt,
                    maxTokens   = if (_chat.value.isThinkingMode) _chat.value.maxTokens * 2 else _chat.value.maxTokens,
                    temperature = if (_chat.value.isThinkingMode) 0.6f else _chat.value.temperature,
                    topP        = 0.9f
                ).collect { token ->
                    var processedToken = token
                    if (token.contains("<think>")) {
                        isThinking = true
                        processedToken = token.replace("<think>", "")
                    }
                    
                    if (isThinking) {
                        if (processedToken.contains("</think>")) {
                            isThinking = false
                            val parts = processedToken.split("</think>")
                            thoughtSb.append(parts[0])
                            if (parts.size > 1) {
                                sb.append(parts[1])
                            }
                        } else {
                            thoughtSb.append(processedToken)
                        }
                    } else {
                        sb.append(processedToken)
                    }

                    tokenCount++
                    _chat.update { state ->
                        val msgs = state.messages.toMutableList()
                        val idx  = msgs.lastIndex
                        if (idx >= 0 && msgs[idx].role == Role.ASSISTANT)
                            msgs[idx] = msgs[idx].copy(
                                content = sb.toString(),
                                thought = thoughtSb.toString().ifBlank { null }
                            )
                        state.copy(messages = msgs)
                    }
                }

                val finalText = sb.toString().trim()
                val dt = (System.currentTimeMillis() - t0) / 1000.0
                val tps = if (dt > 0) tokenCount / dt else 0.0

                if (history.size >= 8) history.removeFirst()
                history.addLast(Pair(displayText, finalText))

                _chat.update { state ->
                    val msgs = state.messages.toMutableList()
                    val idx  = msgs.lastIndex
                    if (idx >= 0 && msgs[idx].role == Role.ASSISTANT)
                        msgs[idx] = msgs[idx].copy(
                            content     = finalText.ifEmpty { "(No response)" },
                            isStreaming = false
                        )
                    state.copy(messages = msgs, isGenerating = false, tokensPerSec = tps.toFloat())
                }

                if (_chat.value.saveChat) {
                    saveCurrentChat()
                }

            } catch (e: Exception) {
                val isCancel = e is kotlinx.coroutines.CancellationException || e.message?.contains("Job was cancelled") == true
                _chat.update { state ->
                    val msgs = state.messages.toMutableList()
                    val idx  = msgs.lastIndex
                    if (idx >= 0 && msgs[idx].role == Role.ASSISTANT) {
                        val currentText = msgs[idx].content
                        msgs[idx] = msgs[idx].copy(
                            content = if (isCancel) {
                                if (currentText == "...") "Response stopped by user." 
                                else "$currentText\n\n[Response stopped by user]"
                            } else "Error: ${e.message}", 
                            isStreaming = false
                        )
                    }
                    state.copy(isGenerating = false, error = if (isCancel) null else e.message)
                }
            }
        }
    }

    fun stopGeneration() {
        LlamaEngine.stopGeneration()
        genJob?.cancel()
        _chat.update { it.copy(isGenerating = false) }
    }

    fun updateUserName(name: String) {
        _chat.update { it.copy(userName = name) }
    }

    fun updateSystemPrompt(prompt: String) {
        _chat.update { it.copy(systemPrompt = prompt) }
        viewModelScope.launch(Dispatchers.IO) {
            LlamaEngine.cacheSystemPrompt(prompt)
        }
    }

    fun clearChat() {
        stopGeneration()
        history.clear()
        _chat.update { it.copy(messages = emptyList(), error = null, currentSessionId = null) }
    }

    private fun loadChatHistory() {
        viewModelScope.launch {
            val history = historyManager.getAllChats()
            _chat.update { it.copy(chatHistory = history) }
        }
    }

    private fun saveCurrentChat() {
        viewModelScope.launch {
            val messages = _chat.value.messages
            if (messages.isEmpty()) return@launch

            val title = messages.firstOrNull { it.role == Role.USER }?.content?.take(30) ?: "New Chat"
            val sessionId = _chat.value.currentSessionId ?: UUID.randomUUID().toString()

            val session = ChatSession(
                id = sessionId,
                title = title,
                messages = messages
            )
            historyManager.saveChat(session)
            _chat.update { it.copy(currentSessionId = sessionId) }
            loadChatHistory()
        }
    }

    fun loadChatSession(session: ChatSession) {
        stopGeneration()
        history.clear()
        // Rebuild context history for LLM
        val chatPairs = session.messages
            .filter { it.role != Role.SYSTEM }
            .chunked(2)
            .mapNotNull { 
                if (it.size == 2) Pair(it[0].content, it[1].content) else null 
            }
        history.addAll(chatPairs.takeLast(8))

        _chat.update { it.copy(
            messages = session.messages,
            currentSessionId = session.id
        ) }
    }

    fun deleteChatSession(id: String) {
        viewModelScope.launch {
            historyManager.deleteChat(id)
            if (_chat.value.currentSessionId == id) {
                clearChat()
            }
            loadChatHistory()
        }
    }

    fun updateUseGpu(enabled: Boolean) {
        _chat.update { it.copy(useGpu = enabled) }
    }

    fun updateSaveChat(enabled: Boolean) {
        _chat.update { it.copy(saveChat = enabled) }
    }

    fun updateTemperature(temp: Float) {
        _chat.update { it.copy(temperature = temp) }
    }

    fun updateMaxTokens(tokens: Int) {
        _chat.update { it.copy(maxTokens = tokens) }
    }

    fun updateThinkingMode(enabled: Boolean) {
        _chat.update { it.copy(isThinkingMode = enabled) }
        val mode = if (enabled) "THINKING MODE: ON (Deep reasoning enabled)" else "THINKING MODE: OFF"
        addSysMsg(mode)
    }

    private fun addSysMsg(t: String) {
        _chat.update { it.copy(messages = it.messages +
                ChatMessage(role = Role.SYSTEM, content = t)) }
    }

    override fun onCleared() {
        super.onCleared()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            try {
                perfHintSession?.close()
            } catch (_: Exception) {}
        }
    }
}
