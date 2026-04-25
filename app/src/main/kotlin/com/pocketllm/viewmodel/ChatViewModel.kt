package com.eigen.viewmodel

import android.app.Application
import android.content.Context
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.eigen.data.*
import com.eigen.engine.LlamaEngine
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import java.util.ArrayDeque

enum class PerformanceMode { HIGH, BALANCED, POWER_SAVER }

data class ChatUiState(
    val messages: List<ChatMessage> = emptyList(),
    val isGenerating: Boolean = false,
    val loadedModel: ModelInfo? = null,
    val loadingModel: Boolean = false,
    val loadProgress: Float = 0f,
    val error: String? = null,
    val tokensPerSec: Float = 0f,
    val firstTokenLatencyMs: Long = 0,
    val deviceTemp: Float = 0f,
    val pendingAttachment: Attachment? = null,
    val isProcessingFile: Boolean = false,
    val userName: String = "User",
    val systemPrompt: String = "You are Eigen, a helpful and efficient local AI assistant.",
    val useGpu: Boolean = true,
    val saveChat: Boolean = true,
    val temperature: Float = 0.7f,
    val maxTokens: Int = 512,
    val performanceMode: PerformanceMode = PerformanceMode.BALANCED,
    val isThinkingMode: Boolean = true,
    val preferredLanguage: String = "English",
    val chatHistory: List<ChatSession> = emptyList(),
    val currentSessionId: String? = null,
    val isFirstLaunch: Boolean = true,
    val localFileHelperEnabled: Boolean = false,
    val showPermissionRationale: Boolean = false,
    val contextLength: Int = 2048
)

data class ModelsUiState(
    val models: List<ModelInfo> = emptyList(),
    val downloadStates: Map<String, DownloadState> = emptyMap(),
    val loadedModelId: String? = null,
    val totalDeviceRamGb: Double = 0.0
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    val modelManager = ModelManager(application)
    val historyManager = ChatHistoryManager(application)

    private val _chat = MutableStateFlow(ChatUiState())
    val chatState: StateFlow<ChatUiState> = _chat
    private val _models = MutableStateFlow(ModelsUiState())
    val modelsState: StateFlow<ModelsUiState> = _models

    private var genJob: Job? = null
    private val history = ArrayDeque<Pair<String, String>>()
    private val downloadJobs = mutableMapOf<String, Job>()

    private val perfHintSession: Any? = null

    init {
        loadChatHistory()
        detectRamAndAutoLoad()
        
        val prefs = application.getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
        val isFirstLaunch = prefs.getBoolean("is_first_launch", true)
        _chat.update { it.copy(
            userName = prefs.getString("user_name", "User") ?: "User",
            systemPrompt = prefs.getString("system_prompt", "You are Eigen, a helpful and efficient local AI assistant.") ?: "You are Eigen, a helpful and efficient local AI assistant.",
            isFirstLaunch = isFirstLaunch,
            useGpu = prefs.getBoolean("use_gpu", true),
            saveChat = prefs.getBoolean("save_chat", true),
            temperature = prefs.getFloat("temperature", 0.7f),
            maxTokens = prefs.getInt("max_tokens", 512),
            performanceMode = PerformanceMode.valueOf(prefs.getString("perf_mode", "BALANCED") ?: "BALANCED"),
            preferredLanguage = prefs.getString("language", "English") ?: "English",
            localFileHelperEnabled = prefs.getBoolean("local_file_helper", false)
        ) }
        startPerformanceMonitoring()
    }

    private fun startPerformanceMonitoring() {
        viewModelScope.launch {
            while (isActive) {
                // In a real app, you'd use SensorManager for temperature
                val currentTemp = 35f + (Math.random() * 5).toFloat() 
                _chat.update { it.copy(deviceTemp = currentTemp) }
                delay(5000)
            }
        }
    }

    private fun boostCpuPerf() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
            try {
                // Implementation using PowerManager or PerformanceHintManager would go here
            } catch (e: Exception) {}
        }
    }

    private fun detectRamAndAutoLoad() {
        viewModelScope.launch {
            val totalRam = modelManager.getTotalRAM()
            _models.update { it.copy(
                models = modelManager.getAvailableModels(),
                totalDeviceRamGb = totalRam
            ) }

            // Auto-load last model if available
            val lastModelId = getApplication<Application>()
                .getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
                .getString("last_model_id", null)
            
            if (lastModelId != null) {
                val model = _models.value.models.find { it.id == lastModelId }
                if (model != null && modelManager.getLocalFile(model).exists()) {
                    loadModel(model)
                }
            }
        }
    }

    fun attachFile(uri: Uri) {
        viewModelScope.launch {
            _chat.update { it.copy(isProcessingFile = true) }
            try {
                val attachment = FileProcessor.processUri(getApplication(), uri)
                _chat.update { it.copy(pendingAttachment = attachment, isProcessingFile = false) }
            } catch (e: Exception) {
                _chat.update { it.copy(error = "File error: ${e.message}", isProcessingFile = false) }
            }
        }
    }

    fun clearAttachment() {
        _chat.update { it.copy(pendingAttachment = null) }
    }

    fun loadModel(model: ModelInfo) {
        viewModelScope.launch {
            val stats = modelManager.getMemoryStats()
            // Be more lenient with RAM requirements (allow up to 20% over physical RAM due to swap/ZRAM)
            if (stats.physicalRamGb * 1.2 < model.ramRequirementGb) {
                _chat.update { it.copy(error = "Warning: ${model.name} requires ${model.ramRequired}, but you have ${"%.1f".format(stats.physicalRamGb)} GB. It might crash.") }
            }

            _chat.update { it.copy(loadingModel = true, loadProgress = 0.1f) }
            try {
                val file = modelManager.getLocalFile(model)
                LlamaEngine.load(
                    file, 
                    useGpu = _chat.value.useGpu,
                    contextSize = _chat.value.contextLength
                ) { progress ->
                    _chat.update { it.copy(loadProgress = 0.1f + (progress * 0.9f)) }
                }
                _chat.update { it.copy(loadedModel = model, loadingModel = false) }
                _models.update { it.copy(loadedModelId = model.id) }
                
                getApplication<Application>()
                    .getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
                    .edit().putString("last_model_id", model.id).apply()

            } catch (e: Exception) {
                _chat.update { it.copy(error = "Load failed: ${e.message}", loadingModel = false) }
            }
        }
    }

    fun unloadModel() {
        viewModelScope.launch {
            LlamaEngine.unloadModel()
            _chat.update { it.copy(loadedModel = null) }
            _models.update { it.copy(loadedModelId = null) }
        }
    }

    fun downloadModel(model: ModelInfo) {
        val job = viewModelScope.launch {
            modelManager.downloadModel(model).collect { state ->
                _models.update { it.copy(downloadStates = it.downloadStates + (model.id to state)) }
                if (state is DownloadState.Success) {
                    _models.update { it.copy(models = modelManager.getAvailableModels()) }
                    // Automatically load the model after successful download
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
        viewModelScope.launch {
            if (_chat.value.loadedModel?.id == model.id) {
                unloadModel()
            }
            modelManager.getLocalFile(model).delete()
            _models.update { it.copy(
                models = modelManager.getAvailableModels(),
                downloadStates = it.downloadStates - model.id
            ) }
        }
    }

    fun deleteMessage(id: Long) {
        _chat.update { state ->
            state.copy(messages = state.messages.filter { it.id != id })
        }
    }

    fun editMessage(id: Long, newContent: String) {
        val currentMsgs = _chat.value.messages
        val index = currentMsgs.indexOfFirst { it.id == id }
        if (index == -1) return

        val messageToEdit = currentMsgs[index]
        
        if (messageToEdit.role == Role.USER) {
            stopGeneration()
            val newMessages = currentMsgs.take(index + 1).toMutableList()
            newMessages[index] = messageToEdit.copy(content = newContent)
            
            _chat.update { state ->
                state.copy(messages = newMessages)
            }
            rebuildHistory()
            sendMessage(newContent, isRetry = true)
        } else {
            _chat.update { state ->
                state.copy(messages = state.messages.map { 
                    if (it.id == id) it.copy(content = newContent) else it 
                })
            }
        }
    }

    private fun rebuildHistory() {
        history.clear()
        val msgs = _chat.value.messages
        for (i in 0 until msgs.size - 1) {
            if (msgs[i].role == Role.USER && i + 1 < msgs.size && msgs[i+1].role == Role.ASSISTANT) {
                history.addLast(Pair(msgs[i].content, msgs[i+1].content))
            }
        }
    }

    fun setLanguage(lang: String) {
        _chat.update { it.copy(preferredLanguage = lang) }
        getApplication<Application>()
            .getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
            .edit().putString("language", lang).apply()
    }

    fun resetFirstLaunch() {
        _chat.update { it.copy(isFirstLaunch = true) }
    }

    fun setFirstLaunchSeen() {
        _chat.update { it.copy(isFirstLaunch = false) }
        val prefs = getApplication<Application>().getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
        prefs.edit().putBoolean("is_first_launch", false).apply()
    }

    fun sendMessage(userText: String, isRetry: Boolean = false) {
        val currentAttachment = if (isRetry) null else _chat.value.pendingAttachment
        if (!isRetry && (userText.isBlank() && currentAttachment == null)) return
        if (_chat.value.isGenerating) return
        
        if (!LlamaEngine.isModelLoaded()) {
            addSysMsg("⚠️ No model loaded — go to Models tab.")
            return
        }

        val attachmentToUse = if (isRetry) {
             _chat.value.messages.lastOrNull { it.role == Role.USER }?.attachment
        } else currentAttachment

        val displayText = if (attachmentToUse != null) {
            if (userText.isNotBlank()) "$userText\n📎 ${attachmentToUse.fileName}"
            else "Analyze this file: ${attachmentToUse.fileName}"
        } else userText

        var fileCtx: String? = null
        if (attachmentToUse != null) {
            fileCtx = when (attachmentToUse.type) {
                AttachmentType.IMAGE -> buildString {
                    val text = attachmentToUse.extractedText ?: ""
                    if (text.isNotBlank() && !text.contains("No text detected")) {
                        append("[IMAGE: ${attachmentToUse.fileName}]\n")
                        append("The user attached an image. Here is the text extracted from it via OCR:\n\n")
                        append(text)
                        if (userText.isBlank()) {
                            append("\n\nAction: The user uploaded this image without a message. Analyze/Summarize the text above.")
                        } else {
                            append("\n\nUser Question: $userText")
                        }
                    } else {
                        append("[IMAGE ATTACHED: ${attachmentToUse.fileName}]\n")
                        append("The user attached an image, but NO text was detected. ")
                        append("You CANNOT see pixels. Tell the user you can't read this specific image and ask for a description.")
                    }
                }
                else -> {
                    val text = attachmentToUse.extractedText ?: ""
                    if (text.isBlank() || text.startsWith("Could not")) {
                        "[FILE: ${attachmentToUse.fileName}]\nContent could not be extracted."
                    } else {
                        buildString {
                            append("[FILE: ${attachmentToUse.fileName}]\n")
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
        } else if (_chat.value.localFileHelperEnabled) {
            // Check for file search intent
            val query = userText.lowercase()
            if (query.contains("find file") || query.contains("search for") || 
                query.contains("recent files") || query.contains("show me files")) {
                
                val searchTerm = query
                    .replace("find file", "")
                    .replace("search for", "")
                    .replace("recent files", "")
                    .replace("show me files", "")
                    .replace("named", "")
                    .trim()

                val foundFiles = if (searchTerm.isBlank() || query.contains("recent")) {
                    LocalFileHelper.getRecentFiles(getApplication())
                } else {
                    LocalFileHelper.searchFiles(getApplication(), searchTerm)
                }

                if (foundFiles.isNotEmpty()) {
                    fileCtx = buildString {
                        append("[LOCAL FILE SYSTEM INFO]\n")
                        append("I found the following files on your device matching your request:\n")
                        foundFiles.forEach { 
                            append("- ${it.name} (${FileProcessor.formatSize(it.size)})\n")
                        }
                        append("\nNote: I can only see metadata. If you want me to read a file's content, please attach it specifically.")
                    }
                } else {
                    fileCtx = "[LOCAL FILE SYSTEM INFO]\nNo files matching '$searchTerm' were found on the device."
                }
            }
        }

        if (!isRetry) {
            val userMsg = ChatMessage(
                id      = System.nanoTime(),
                role    = Role.USER,
                content = displayText,
                attachment = attachmentToUse
            )
            _chat.update { state ->
                state.copy(
                    messages = state.messages + userMsg,
                    pendingAttachment = null
                )
            }
        }

        val asstMsg = ChatMessage(
            id      = System.nanoTime() + 1,
            role    = Role.ASSISTANT,
            content = "...",
            isStreaming = true
        )

        _chat.update { it.copy(
            messages = it.messages + asstMsg,
            isGenerating = true
        ) }

        genJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                boostCpuPerf()

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
                var tokenCount = 0
                val startTs = System.currentTimeMillis()
                var firstTokenTs = 0L

                // Adaptive parameters based on Performance Mode
                val smoothingDelay = when (_chat.value.performanceMode) {
                    PerformanceMode.HIGH -> 5L
                    PerformanceMode.BALANCED -> 20L
                    PerformanceMode.POWER_SAVER -> 50L
                }

                val threads = when (_chat.value.performanceMode) {
                    PerformanceMode.HIGH -> 0 // Auto (usually uses all P-cores)
                    PerformanceMode.BALANCED -> 4
                    PerformanceMode.POWER_SAVER -> 2
                }

                LlamaEngine.generateFlow(
                    prompt      = prompt,
                    maxTokens   = if (_chat.value.isThinkingMode) _chat.value.maxTokens * 2 else _chat.value.maxTokens,
                    temperature = if (_chat.value.isThinkingMode) 0.6f else _chat.value.temperature,
                    topP        = 0.9f,
                    smoothingDelayMs = smoothingDelay
                ).collect { token ->
                    if (tokenCount == 0) {
                        firstTokenTs = System.currentTimeMillis()
                        _chat.update { it.copy(firstTokenLatencyMs = firstTokenTs - startTs) }
                    }
                    tokenCount++
                    
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

                val finalText = sb.toString()
                val totalTime = System.currentTimeMillis() - startTs
                val tps = if (totalTime > 0) (tokenCount / (totalTime / 1000f)) else 0f

                _chat.update { state ->
                    val msgs = state.messages.toMutableList()
                    val idx  = msgs.lastIndex
                    if (idx >= 0 && msgs[idx].role == Role.ASSISTANT)
                        msgs[idx] = msgs[idx].copy(isStreaming = false)
                    state.copy(
                        messages = msgs, 
                        isGenerating = false,
                        tokensPerSec = tps
                    )
                }

                if (_chat.value.saveChat) {
                    history.addLast(Pair(userText.ifBlank { displayText }, finalText))
                    if (history.size > 10) history.removeFirst()
                    saveCurrentChat()
                }

            } catch (e: CancellationException) {
                // Stopped by user
            } catch (e: Exception) {
                val isCancel = e is CancellationException || e.message?.contains("cancel", true) == true
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
    }

    fun updateUserName(name: String) {
        _chat.update { it.copy(userName = name) }
        getApplication<Application>()
            .getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
            .edit().putString("user_name", name).apply()
    }

    fun updateSystemPrompt(prompt: String) {
        _chat.update { it.copy(systemPrompt = prompt) }
        getApplication<Application>()
            .getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
            .edit().putString("system_prompt", prompt).apply()
    }

    fun clearChat() {
        stopGeneration()
        _chat.update { it.copy(messages = emptyList(), currentSessionId = null) }
        history.clear()
    }

    private fun loadChatHistory() {
        viewModelScope.launch {
            val sessions = historyManager.getAllChats()
            _chat.update { it.copy(chatHistory = sessions) }
        }
    }

    private fun saveCurrentChat() {
        viewModelScope.launch {
            val state = _chat.value
            if (state.messages.isEmpty()) return@launch
            
            val firstUserMsg = state.messages.firstOrNull { it.role == Role.USER }?.content ?: "New Chat"
            val title = if (firstUserMsg.length > 30) firstUserMsg.take(27) + "..." else firstUserMsg
            
            val sessionId = state.currentSessionId ?: System.currentTimeMillis().toString()
            val session = ChatSession(
                id = sessionId,
                title = title,
                timestamp = System.currentTimeMillis(),
                messages = state.messages
            )
            
            historyManager.saveChat(session)
            _chat.update { it.copy(currentSessionId = sessionId) }
            loadChatHistory()
        }
    }

    fun loadChatSession(session: ChatSession) {
        stopGeneration()
        _chat.update { it.copy(
            messages = session.messages,
            currentSessionId = session.id
        ) }
        
        history.clear()
        val msgs = session.messages
        for (i in 0 until msgs.size - 1) {
            if (msgs[i].role == Role.USER && i + 1 < msgs.size && msgs[i+1].role == Role.ASSISTANT) {
                history.addLast(Pair(msgs[i].content, msgs[i+1].content))
            }
        }
    }

    fun deleteChatSession(sessionId: String) {
        viewModelScope.launch {
            historyManager.deleteChat(sessionId)
            if (_chat.value.currentSessionId == sessionId) {
                clearChat()
            }
            loadChatHistory()
        }
    }

    fun updateUseGpu(use: Boolean) {
        _chat.update { it.copy(useGpu = use) }
        getApplication<Application>()
            .getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean("use_gpu", use).apply()
    }

    fun updateSaveChat(save: Boolean) {
        _chat.update { it.copy(saveChat = save) }
        getApplication<Application>()
            .getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean("save_chat", save).apply()
    }

    fun updatePerformanceMode(mode: PerformanceMode) {
        _chat.update { it.copy(performanceMode = mode) }
        getApplication<Application>()
            .getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
            .edit().putString("perf_mode", mode.name).apply()
        
        // If model is loaded, we might need to reload or update threads (not always possible without reload)
        // For now, we update it for the next generation call
    }

    fun updateTemperature(temp: Float) {
        _chat.update { it.copy(temperature = temp) }
        getApplication<Application>()
            .getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
            .edit().putFloat("temperature", temp).apply()
    }

    fun updateMaxTokens(tokens: Int) {
        _chat.update { it.copy(maxTokens = tokens) }
        getApplication<Application>()
            .getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
            .edit().putInt("max_tokens", tokens).apply()
    }

    fun updateContextLength(length: Int) {
        _chat.update { it.copy(contextLength = length) }
        getApplication<Application>()
            .getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
            .edit().putInt("context_length", length).apply()
    }

    fun updateThinkingMode(enabled: Boolean) {
        _chat.update { it.copy(isThinkingMode = enabled) }
    }

    fun updateLocalFileHelper(enabled: Boolean) {
        if (enabled) {
            _chat.update { it.copy(showPermissionRationale = true) }
        } else {
            _chat.update { it.copy(localFileHelperEnabled = false) }
            getApplication<Application>()
                .getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
                .edit().putBoolean("local_file_helper", false).apply()
        }
    }

    fun dismissPermissionRationale() {
        _chat.update { it.copy(showPermissionRationale = false) }
    }

    fun onPermissionResult(granted: Boolean) {
        _chat.update { it.copy(
            localFileHelperEnabled = granted,
            showPermissionRationale = false
        ) }
        getApplication<Application>()
            .getSharedPreferences("eigen_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean("local_file_helper", granted).apply()
    }

    fun addSysMsg(content: String) {
        _chat.update { it.copy(messages = it.messages + ChatMessage(System.nanoTime(), Role.SYSTEM, content)) }
    }

    override fun onCleared() {
        super.onCleared()
        LlamaEngine.unloadModel()
    }
}
