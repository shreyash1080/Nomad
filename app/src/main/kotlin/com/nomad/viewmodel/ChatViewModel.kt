package com.nomad.viewmodel

import android.Manifest
import android.app.Application
import android.content.Context
import android.net.Uri
import android.os.Build
import android.content.pm.PackageManager
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.core.content.ContextCompat
import androidx.work.*
import com.nomad.data.*
import com.nomad.engine.LlamaEngine
import com.nomad.mobilellm.AirLLMEngine
import com.nomad.mobilellm.MobileAirLLMClient
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.ArrayDeque

enum class PerformanceMode { HIGH, BALANCED, POWER_SAVER }
enum class ResponseMode { FAST, BALANCED, THINKING }

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
    val userName: String = "Joe",
    val systemPrompt: String = """
        You are Nomad, a highly intelligent and versatile on-device AI assistant.
        Your goal is to provide exceptional assistance across various domains including coding, writing, analysis, and general knowledge.
        
        GUIDELINES:
        1. Accuracy First: Provide factually correct information. If unsure, state your limitations.
        2. Technical Excellence: When writing code, ensure it is modern, idiomatic, and follows best practices for the specific language or framework.
        3. Contextual Awareness: Pay close attention to previous messages in the conversation to maintain consistency.
        4. Structured Responses: Use Markdown (headers, lists, code blocks) to make your answers easy to read.
        5. Adaptive Tone: Maintain a professional yet helpful and approachable persona.
        6. On-Device Nature: You are running locally on the user's mobile device. Be efficient with resources while delivering high-quality output.
        7. Web & Files: Use the specific tools provided (Web Search, File Analysis) to augment your knowledge when relevant.
    """.trimIndent(),
    val useGpu: Boolean = true,
    val saveChat: Boolean = true,
    val temperature: Float = 0.7f,
    val maxTokens: Int = 1600,
    val performanceMode: PerformanceMode = PerformanceMode.BALANCED,
    val responseMode: ResponseMode = ResponseMode.BALANCED,
    val preferredLanguage: String = "English",
    val chatHistory: List<ChatSession> = emptyList(),
    val currentSessionId: String? = null,
    val isFirstLaunch: Boolean = true,
    val hasAcceptedTerms: Boolean = false,
    val localFileHelperEnabled: Boolean = false,
    val webSearchEnabled: Boolean = true,
    val isSearchingWeb: Boolean = false,
    val showPermissionRationale: Boolean = false,
    val contextLength: Int = 4096,
    val loadStatus: String = "Loading model..."
)

data class ModelsUiState(
    val models: List<ModelInfo> = emptyList(),
    val downloadStates: Map<String, DownloadState> = emptyMap(),
    val loadedModelId: String? = null,
    val totalDeviceRamGb: Double = 0.0,
    val physicalRamGb: Double = 0.0,
    val availableStorageGb: Double = 0.0,
    val recommendedModelId: String? = null
)

private data class PerformanceProfile(
    val generationThreads: Int,
    val promptThreads: Int,
    val historyPairs: Int,
    val uiUpdateIntervalMs: Long,
    val maxTokens: Int,
    val temperature: Float,
    val responseMode: ResponseMode,
    val useExplicitThinking: Boolean
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    val modelManager = ModelManager(application)
    val historyManager = ChatHistoryManager(application)

    private val _chat = MutableStateFlow(ChatUiState())
    val chatState: StateFlow<ChatUiState> = _chat
    private val _models = MutableStateFlow(ModelsUiState())
    val modelsState: StateFlow<ModelsUiState> = _models

    private var airLLMEngine: AirLLMEngine? = null
    private var genJob: Job? = null
    private val history = ArrayDeque<Pair<String, String>>()
    private val downloadJobs = mutableMapOf<String, Job>()
    private val modelLoadMutex = Mutex()

    private var initialContextLength: Int = 4096

    init {
        loadChatHistory()
        detectRamAndAutoLoad()

        val prefs = application.getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
        val isFirstLaunch = prefs.getBoolean("is_first_launch", true)
        val savedContext = prefs.getInt("context_length", 4096)
        initialContextLength = savedContext

        _chat.update {
            it.copy(
                userName = prefs.getString("user_name", "Joe") ?: "Joe",
                systemPrompt = prefs.getString(
                    "system_prompt",
                    """
                        You are Nomad, a highly intelligent and versatile on-device AI assistant.
                        Your goal is to provide exceptional assistance across various domains including coding, writing, analysis, and general knowledge.
                        
                        GUIDELINES:
                        1. Accuracy First: Provide factually correct information. If unsure, state your limitations.
                        2. Technical Excellence: When writing code, ensure it is modern, idiomatic, and follows best practices for the specific language or framework.
                        3. Contextual Awareness: Pay close attention to previous messages in the conversation to maintain consistency.
                        4. Structured Responses: Use Markdown (headers, lists, code blocks) to make your answers easy to read.
                        5. Adaptive Tone: Maintain a professional yet helpful and approachable persona.
                        6. On-Device Nature: You are running locally on the user's mobile device. Be efficient with resources while delivering high-quality output.
                        7. Web & Files: Use the specific tools provided (Web Search, File Analysis) to augment your knowledge when relevant.
                    """.trimIndent()
                ) ?: "",
                isFirstLaunch = isFirstLaunch,
                useGpu = prefs.getBoolean("use_gpu", true),
                saveChat = prefs.getBoolean("save_chat", true),
                temperature = prefs.getFloat("temperature", 0.7f),
                maxTokens = prefs.getInt("max_tokens", 1600),
                performanceMode = PerformanceMode.valueOf(
                    prefs.getString("perf_mode", "BALANCED") ?: "BALANCED"
                ),
                responseMode = ResponseMode.valueOf(
                    prefs.getString("response_mode", "BALANCED") ?: "BALANCED"
                ),
                preferredLanguage = prefs.getString("language", "English") ?: "English",
                localFileHelperEnabled = prefs.getBoolean("local_file_helper", false),
                webSearchEnabled = prefs.getBoolean("web_search_enabled", true),
                hasAcceptedTerms = prefs.getBoolean("has_accepted_terms", false),
                contextLength = prefs.getInt("context_length", 4096)
            )
        }
        startPerformanceMonitoring()
    }

    private fun startPerformanceMonitoring() {
        viewModelScope.launch {
            while (isActive) {
                val currentTemp = 35f + (Math.random() * 5).toFloat()
                _chat.update { it.copy(deviceTemp = currentTemp) }
                delay(5000)
            }
        }
    }

    private fun boostCpuPerf() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            try {
                // PowerManager / PerformanceHintManager integration goes here
            } catch (_: Exception) {
            }
        }
    }

    private fun detectRamAndAutoLoad() {
        viewModelScope.launch {
            val stats = modelManager.getMemoryStats()
            val availableModels = modelManager.getAvailableModels()
            val recommendedModel = modelManager.recommendBestModel(availableModels)
            _models.update {
                it.copy(
                    models = availableModels,
                    totalDeviceRamGb = stats.effectiveRamBudgetGb,
                    physicalRamGb = stats.physicalRamGb,
                    availableStorageGb = stats.availableStorageGb,
                    recommendedModelId = recommendedModel.id
                )
            }
            val lastModelId = getApplication<Application>()
                .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
                .getString("last_model_id", null)
            if (lastModelId != null) {
                val model = _models.value.models.find { it.id == lastModelId }
                if (model != null && modelManager.getLocalFile(model).exists()) loadModel(model)
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
            modelLoadMutex.withLock {
                val stats = modelManager.getMemoryStats()
                val currentLoadedModel = _chat.value.loadedModel
                val isSwitchingModel =
                    currentLoadedModel?.id != null && currentLoadedModel.id != model.id

                val isAirLLM = model.id.endsWith("-air")

                if (!isAirLLM && stats.effectiveRamBudgetGb * 1.05 < model.ramRequirementGb) {
                    _chat.update {
                        it.copy(
                            error = "Warning: ${model.name} needs about ${model.ramRequired}, " +
                                    "while this device budget is about ${"%.1f".format(stats.effectiveRamBudgetGb)} GB."
                        )
                    }
                }

                val tunedState = applyRecommendedSettingsForModel(model, stats)
                _chat.update {
                    it.copy(
                        loadingModel = true,
                        loadProgress = if (isSwitchingModel) 0.06f else 0.12f,
                        loadStatus = if (isSwitchingModel) "Switching to ${model.name}..."
                        else "Loading ${model.name}..."
                    )
                }

                try {
                    stopGenerationAndWait()

                    if (currentLoadedModel != null) {
                        if (currentLoadedModel.id.endsWith("-air")) {
                            airLLMEngine?.unload()
                            airLLMEngine = null
                        } else {
                            LlamaEngine.unloadModel()
                        }
                    }

                    if (isAirLLM) {
                        _chat.update { it.copy(loadStatus = "Connecting to MobileAirLLM...") }
                        
                        val engine = AirLLMEngine(
                            context = getApplication(),
                            modelId = model.downloadUrl,
                            compression = model.quantization
                        )
                        airLLMEngine = engine
                        
                        // Attempt to load the model through the engine
                        val initResult = engine.initialize()
                        if (initResult is AirLLMEngine.InitResult.Success) {
                            val file = modelManager.getLocalFile(model)
                            val client = MobileAirLLMClient()
                            val loadSuccess = client.loadModel(
                                modelPath = file.absolutePath,
                                compression = model.quantization
                            )
                            if (!loadSuccess) throw RuntimeException("AirLLM failed to load shards.")
                        } else if (initResult is AirLLMEngine.InitResult.ServerNotRunning) {
                            throw RuntimeException("MobileAirLLM server not running. Please start it in Termux.")
                        } else if (initResult is AirLLMEngine.InitResult.LoadFailed) {
                            throw RuntimeException("AirLLM load failed: ${(initResult as AirLLMEngine.InitResult.LoadFailed).reason}")
                        }
                    } else {
                        val file = modelManager.getLocalFile(model)
                        val loadResult = LlamaEngine.load(
                            file,
                            useGpu = tunedState.useGpu,
                            contextSize = tunedState.contextLength,
                            threads = generationThreadsFor(tunedState.performanceMode),
                            systemPrompt = tunedState.systemPrompt
                        ) { progress ->
                            _chat.update {
                                it.copy(
                                    loadProgress = 0.2f + (progress * 0.75f),
                                    loadStatus = "Optimizing ${model.name} for this device..."
                                )
                            }
                        }
                        loadResult.getOrThrow()
                        LlamaEngine.setThreads(
                            generationThreadsFor(tunedState.performanceMode),
                            promptThreadsFor(tunedState.performanceMode)
                        )
                    }

                    _chat.update {
                        it.copy(
                            loadedModel = model,
                            loadingModel = false,
                            loadProgress = 1f,
                            loadStatus = "${model.name} ready"
                        )
                    }
                    _models.update { it.copy(loadedModelId = model.id) }

                    getApplication<Application>()
                        .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
                        .edit().putString("last_model_id", model.id).apply()

                    addSysMsg("Nomad tuned ${model.name} for this device.")

                    delay(120)
                    _chat.update { it.copy(loadProgress = 0f, loadStatus = "Loading model...") }
                } catch (e: Exception) {
                    _chat.update {
                        it.copy(
                            error = "Load failed: ${e.message}",
                            loadingModel = false,
                            loadProgress = 0f,
                            loadStatus = "Loading model..."
                        )
                    }
                }
            }
        }
    }

    fun unloadModel() {
        viewModelScope.launch {
            modelLoadMutex.withLock {
                stopGenerationAndWait()
                LlamaEngine.unloadModel()
                _chat.update {
                    it.copy(
                        loadedModel = null, loadingModel = false,
                        loadProgress = 0f, loadStatus = "Loading model..."
                    )
                }
                _models.update { it.copy(loadedModelId = null) }
            }
        }
    }

    fun downloadModel(model: ModelInfo) {
        val workManager = WorkManager.getInstance(getApplication())
        val data = workDataOf("model_id" to model.id)
        
        val downloadRequest = OneTimeWorkRequestBuilder<DownloadWorker>()
            .setInputData(data)
            .setExpedited(OutOfQuotaPolicy.RUN_AS_NON_EXPEDITED_WORK_REQUEST)
            .addTag("download_${model.id}")
            .build()
            
        workManager.enqueueUniqueWork(
            "download_${model.id}",
            ExistingWorkPolicy.REPLACE,
            downloadRequest
        )

        // Observe progress from WorkManager
        viewModelScope.launch {
            workManager.getWorkInfoByIdFlow(downloadRequest.id).collect { workInfo ->
                if (workInfo != null) {
                    val progress = workInfo.progress.getInt("progress", 0)
                    val error = workInfo.outputData.getString("error")
                    
                    _models.update { state ->
                        val newStates = state.downloadStates.toMutableMap()
                        when (workInfo.state) {
                            WorkInfo.State.RUNNING -> {
                                newStates[model.id] = DownloadState.Progress(progress, 0, model.fileSizeBytes)
                            }
                            WorkInfo.State.SUCCEEDED -> {
                                newStates[model.id] = DownloadState.Success
                                // Reload models to show READY
                                _models.update { it.copy(models = modelManager.getAvailableModels()) }
                            }
                            WorkInfo.State.FAILED -> {
                                newStates[model.id] = DownloadState.Error(error ?: "Unknown error")
                            }
                            else -> {}
                        }
                        state.copy(downloadStates = newStates)
                    }
                }
            }
        }
    }

    fun cancelDownload(modelId: String) {
        downloadJobs[modelId]?.cancel()
        downloadJobs.remove(modelId)
        _models.update { it.copy(downloadStates = it.downloadStates - modelId) }
    }

    fun deleteModel(model: ModelInfo) {
        viewModelScope.launch {
            if (_chat.value.loadedModel?.id == model.id) unloadModel()
            modelManager.getLocalFile(model).delete()
            _models.update {
                it.copy(
                    models = modelManager.getAvailableModels(),
                    downloadStates = it.downloadStates - model.id
                )
            }
        }
    }

    fun deleteMessage(id: Long) {
        _chat.update { state -> state.copy(messages = state.messages.filter { it.id != id }) }
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
            _chat.update { state -> state.copy(messages = newMessages) }
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
            if (msgs[i].role == Role.USER && i + 1 < msgs.size && msgs[i + 1].role == Role.ASSISTANT) {
                history.addLast(Pair(msgs[i].content, msgs[i + 1].content))
            }
        }
    }

    fun setLanguage(lang: String) {
        _chat.update { it.copy(preferredLanguage = lang) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putString("language", lang).apply()
    }

    fun resetFirstLaunch() {
        _chat.update { it.copy(isFirstLaunch = true) }
    }

    fun setFirstLaunchSeen() {
        _chat.update { it.copy(isFirstLaunch = false) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean("is_first_launch", false).apply()
    }

    fun acceptTerms() {
        _chat.update { it.copy(hasAcceptedTerms = true) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean("has_accepted_terms", true).apply()
    }

    // ── Message sending — FIXED routing order ───────────────────────────────
    /**
     * FIXED routing logic:
     * 1. Web search is checked FIRST if enabled — "search ..." goes here
     * 2. Local file search is checked SECOND if enabled — "find ..." goes here
     * 3. Removed overlap: "search" is no longer in LocalFileHelper's finder verbs
     *
     * Previously the routing was reversed (file checked before web), and "search"
     * existed in both helpers causing "search weather in Mumbai" to trigger file search.
     */
    fun sendMessage(userText: String, isRetry: Boolean = false) {
        val stateSnapshot = _chat.value
        val currentAttachment = if (isRetry) null else stateSnapshot.pendingAttachment
        if (!isRetry && (userText.isBlank() && currentAttachment == null)) return
        if (stateSnapshot.isGenerating) return
        if (stateSnapshot.loadingModel) {
            addSysMsg(
                "Wait for ${stateSnapshot.loadedModel?.name ?: "the model"} " +
                        "to finish loading before sending a message."
            )
            return
        }

        val attachmentToUse = if (isRetry)
            stateSnapshot.messages.lastOrNull { it.role == Role.USER }?.attachment
        else currentAttachment

        val displayText = if (attachmentToUse != null) {
            if (userText.isNotBlank()) userText
            else "📎 ${attachmentToUse.fileName}"
        } else {
            // Remove "search " or "lookup " from the displayed message
            if (WebSearchHelper.isWebSearchIntent(userText)) {
                WebSearchHelper.cleanQuery(userText)
            } else userText
        }

        // ── Intent routing (only for plain text, no attachment) ────────────
        if (attachmentToUse == null) {
            // PRIORITY 1 — Web Search (checked FIRST to avoid "search" being stolen by file helper)
            if (_chat.value.webSearchEnabled && WebSearchHelper.isWebSearchIntent(userText)) {
                handleWebSearchRequest(userText)
                return
            }

            // PRIORITY 2 — Local File Search
            if (_chat.value.localFileHelperEnabled) {
                val fileIntent = LocalFileHelper.parseIntent(userText)
                if (fileIntent != null) {
                    handleLocalFileHelperRequest(
                        userText = userText,
                        displayText = displayText,
                        intent = fileIntent
                    )
                    return
                }
            }
        }

        if (!LlamaEngine.isModelLoaded() && airLLMEngine?.isReady != true) {
            addSysMsg("⚠️ No model loaded — go to Models tab.")
            return
        }

        var fileCtx: String? = null
        if (attachmentToUse != null) {
            fileCtx = when (attachmentToUse.type) {
                AttachmentType.IMAGE -> buildString {
                    val text = attachmentToUse.extractedText ?: ""
                    if (text.isNotBlank() && !text.contains("No text detected")) {
                        append("[IMAGE: ${attachmentToUse.fileName}]\n")
                        append("The user attached an image. Here is the text extracted from it via OCR:\n\n")
                        append(text)
                        if (userText.isBlank()) append("\n\nAction: The user uploaded this image without a message. Analyze/Summarize the text above.")
                        else append("\n\nUser Question: $userText")
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
                    } else buildString {
                        append("[FILE: ${attachmentToUse.fileName}]\n")
                        append("CONTENT:\n$text\n\n")
                        if (userText.isBlank()) append("Action: The user uploaded this file without a message. Summarize it.")
                        else append("User Question: $userText")
                    }
                }
            }
        }

        if (!isRetry) appendUserMessage(displayText, attachmentToUse)

        val asstMsg = ChatMessage(
            id = System.nanoTime() + 1,
            role = Role.ASSISTANT,
            content = "...",
            isStreaming = true
        )
        _chat.update { it.copy(messages = it.messages + asstMsg, isGenerating = true, error = null) }

        generateResponse(userText, displayText, attachmentToUse, fileCtx)
    }

    private fun generateResponse(
        userText: String,
        displayText: String,
        attachmentToUse: Attachment?,
        fileCtx: String?,
        isInternalPrompt: Boolean = false,
        internalPrompt: String? = null
    ) {
        genJob = viewModelScope.launch(Dispatchers.Default) {
            try {
                boostCpuPerf()
                val liveState = _chat.value
                val isAirLLM = liveState.loadedModel?.id?.endsWith("-air") == true
                val perfProfile = buildPerformanceProfile(liveState, userText, attachmentToUse)
                
                if (!isAirLLM) {
                    LlamaEngine.setThreads(perfProfile.generationThreads, perfProfile.promptThreads)
                }

                val prompt = if (isInternalPrompt && internalPrompt != null) {
                    LlamaEngine.buildChatPrompt(
                        modelName = liveState.loadedModel?.name ?: "",
                        systemPrompt = buildSystemPromptForTurn(liveState, perfProfile),
                        history = emptyList(),
                        userMessage = internalPrompt,
                        fileContext = ""
                    )
                } else {
                    LlamaEngine.buildChatPrompt(
                        modelName = liveState.loadedModel?.name ?: "",
                        systemPrompt = buildSystemPromptForTurn(liveState, perfProfile),
                        history = if (liveState.saveChat) history.toList()
                            .takeLast(perfProfile.historyPairs) else emptyList(),
                        userMessage = userText.ifBlank { displayText },
                        fileContext = fileCtx ?: ""
                    )
                }

                val sb = StringBuilder()
                val thoughtSb = StringBuilder()
                var isThinking = false
                var tokenCount = 0
                val startTs = System.currentTimeMillis()
                var firstTokenTs = 0L
                var lastUiUpdateTs = 0L

                val flow = if (isAirLLM) {
                    airLLMEngine?.generateFlow(
                        prompt = prompt,
                        maxNewTokens = perfProfile.maxTokens,
                        temperature = perfProfile.temperature
                    ) ?: emptyFlow()
                } else {
                    LlamaEngine.generateFlow(
                        prompt = prompt,
                        maxTokens = perfProfile.maxTokens,
                        temperature = perfProfile.temperature,
                        topP = 0.95f
                    )
                }

                flow.collect { token ->
                    if (tokenCount == 0) {
                        firstTokenTs = System.currentTimeMillis()
                        _chat.update { it.copy(firstTokenLatencyMs = firstTokenTs - startTs) }
                    }
                    tokenCount++
                    val nowTs = System.currentTimeMillis()
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
                            if (parts.size > 1) sb.append(parts[1])
                        } else {
                            thoughtSb.append(processedToken)
                        }
                    } else {
                        sb.append(processedToken)
                    }

                    val shouldUpdateUi = tokenCount <= 2 ||
                            nowTs - lastUiUpdateTs >= perfProfile.uiUpdateIntervalMs ||
                            token.contains('\n') || token.contains('.') ||
                            token.contains('!') || token.contains('?')

                    if (shouldUpdateUi) {
                        updateAssistantMessage(
                            content = sb.toString(),
                            thought = thoughtSb.toString().ifBlank { null },
                            isStreaming = true
                        )
                        lastUiUpdateTs = nowTs
                    }
                }

                val finalText = sb.toString()
                val totalTime = System.currentTimeMillis() - startTs
                val tps = if (totalTime > 0) (tokenCount / (totalTime / 1000f)) else 0f

                // Fast update: no typing effect for web results or when explicitly requested
                updateAssistantMessage(
                    content = if (sb.toString().isBlank()) "No response generated." else sb.toString(),
                    thought = thoughtSb.toString().ifBlank { null },
                    isStreaming = false,
                    tokensPerSec = tps
                )

                if (_chat.value.saveChat) {
                    history.addLast(Pair(userText.ifBlank { displayText }, finalText))
                    if (history.size > 8) history.removeFirst()
                    saveCurrentChat()
                }

            } catch (_: CancellationException) {
                // Stopped by user — normal exit
                _chat.update { state ->
                    val msgs = state.messages.toMutableList()
                    val idx = msgs.lastIndex
                    if (idx >= 0 && msgs[idx].role == Role.ASSISTANT) {
                        msgs[idx] = msgs[idx].copy(
                            content = if (msgs[idx].content == "...") "Stopped." else msgs[idx].content,
                            isStreaming = false
                        )
                    }
                    state.copy(isGenerating = false)
                }
            } catch (e: Exception) {
                val isCancel =
                    e is CancellationException || e.message?.contains("cancel", true) == true
                _chat.update { state ->
                    val msgs = state.messages.toMutableList()
                    val idx = msgs.lastIndex
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
        airLLMEngine?.let {
             // MobileAirLLMClient doesn't have a stop method exposed in AirLLMEngine yet, 
             // but we can cancel the job.
        }
        genJob?.cancel()
    }

    private suspend fun stopGenerationAndWait() {
        LlamaEngine.stopGeneration()
        // airLLMEngine stop would go here if available
        genJob?.cancelAndJoin()
        genJob = null
        _chat.update { it.copy(isGenerating = false) }
    }

    fun updateUserName(name: String) {
        _chat.update { it.copy(userName = name) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putString("user_name", name).apply()
    }

    fun updateSystemPrompt(prompt: String) {
        _chat.update { it.copy(systemPrompt = prompt) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putString("system_prompt", prompt).apply()
        if (LlamaEngine.isModelLoaded()) {
            viewModelScope.launch(Dispatchers.IO) {
                LlamaEngine.cacheSystemPrompt(LlamaEngine.buildSystemPrefix(prompt))
            }
        }
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
        _chat.update { it.copy(messages = session.messages, currentSessionId = session.id) }
        history.clear()
        val msgs = session.messages
        for (i in 0 until msgs.size - 1) {
            if (msgs[i].role == Role.USER && i + 1 < msgs.size && msgs[i + 1].role == Role.ASSISTANT) {
                history.addLast(Pair(msgs[i].content, msgs[i + 1].content))
            }
        }
    }

    fun deleteChatSession(sessionId: String) {
        viewModelScope.launch {
            historyManager.deleteChat(sessionId)
            if (_chat.value.currentSessionId == sessionId) clearChat()
            loadChatHistory()
        }
    }

    fun updateUseGpu(use: Boolean) {
        _chat.update { it.copy(useGpu = use) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean("use_gpu", use).apply()
        reloadLoadedModelIfNeeded()
    }

    fun updateSaveChat(save: Boolean) {
        _chat.update { it.copy(saveChat = save) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean("save_chat", save).apply()
    }

    fun updatePerformanceMode(mode: PerformanceMode) {
        val current = _chat.value
        _chat.update { it.copy(performanceMode = mode) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putString("perf_mode", mode.name).apply()
        
        // Dynamic update without reload if possible
        if (LlamaEngine.isModelLoaded()) {
            LlamaEngine.setThreads(generationThreadsFor(mode), promptThreadsFor(mode))
        }
    }

    fun updateTemperature(temp: Float) {
        _chat.update { it.copy(temperature = temp) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putFloat("temperature", temp).apply()
    }

    fun updateMaxTokens(tokens: Int) {
        _chat.update { it.copy(maxTokens = tokens) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putInt("max_tokens", tokens).apply()
    }

    fun updateContextLength(length: Int) {
        if (_chat.value.contextLength == length) return
        
        _chat.update { it.copy(contextLength = length) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putInt("context_length", length).apply()
    }

    /**
     * Call this when the user finishes adjusting settings (e.g. onDismiss of SettingsSheet)
     * to avoid reloading multiple times during slider movement.
     */
    fun applySettingsAndReloadIfNeeded() {
        val currentContext = _chat.value.contextLength
        
        // Only reload if context length actually changed from when settings were opened
        if (LlamaEngine.isModelLoaded() && currentContext != initialContextLength) {
             reloadLoadedModelIfNeeded()
             initialContextLength = currentContext
        }
    }

    /**
     * Call this when opening the settings menu to track the baseline context length
     */
    fun onSettingsOpened() {
        initialContextLength = _chat.value.contextLength
    }

    fun updateResponseMode(mode: ResponseMode) {
        _chat.update { it.copy(responseMode = mode) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putString("response_mode", mode.name).apply()
    }

    fun updateThinkingMode(enabled: Boolean) {
        updateResponseMode(if (enabled) ResponseMode.THINKING else ResponseMode.BALANCED)
    }

    /**
     * FIXED: Local File Helper and Web Search are mutually exclusive at the settings level,
     * but the check order (web first, file second) in sendMessage ensures correct dispatch
     * even if both were somehow enabled.
     */
    fun updateLocalFileHelper(enabled: Boolean) {
        if (enabled) {
            _chat.update {
                it.copy(
                    error = "Local Insight (Beta) is currently disabled for further optimization. Please use Web Search for now.",
                    localFileHelperEnabled = false
                )
            }
            return
        }
        _chat.update { it.copy(localFileHelperEnabled = false) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean("local_file_helper", false).apply()
    }

    fun dismissPermissionRationale() {
        _chat.update { it.copy(showPermissionRationale = false) }
    }

    fun updateWebSearchEnabled(enabled: Boolean) {
        _chat.update { it.copy(webSearchEnabled = enabled) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean("web_search_enabled", enabled).apply()
        if (enabled) {
            // Disable file helper when enabling web search
            if (_chat.value.localFileHelperEnabled) {
                _chat.update { it.copy(localFileHelperEnabled = false) }
                getApplication<Application>()
                    .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
                    .edit().putBoolean("local_file_helper", false).apply()
            }
        }
    }

    fun onPermissionResult(granted: Boolean) {
        _chat.update { it.copy(localFileHelperEnabled = granted, showPermissionRationale = false) }
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit().putBoolean("local_file_helper", granted).apply()
    }

    fun addSysMsg(content: String) {
        _chat.update {
            it.copy(messages = it.messages + ChatMessage(System.nanoTime(), Role.SYSTEM, content))
        }
    }

    private fun appendUserMessage(content: String, attachment: Attachment?) {
        val userMsg = ChatMessage(
            id = System.nanoTime(),
            role = Role.USER,
            content = content,
            attachment = attachment
        )
        _chat.update { state ->
            state.copy(messages = state.messages + userMsg, pendingAttachment = null, error = null)
        }
    }

    private fun updateAssistantMessage(
        content: String,
        thought: String? = null,
        isStreaming: Boolean,
        tokensPerSec: Float? = null,
        searchResults: String? = null
    ) {
        _chat.update { state ->
            val msgs = state.messages.toMutableList()
            val idx = msgs.lastIndex
            if (idx >= 0 && msgs[idx].role == Role.ASSISTANT) {
                msgs[idx] = msgs[idx].copy(
                    content = content,
                    thought = thought,
                    isStreaming = isStreaming,
                    searchResults = searchResults ?: msgs[idx].searchResults
                )
            }
            state.copy(
                messages = msgs,
                isGenerating = isStreaming,
                tokensPerSec = tokensPerSec ?: state.tokensPerSec
            )
        }
    }

    // ── Local File Handler ──────────────────────────────────────────────────

    private fun handleLocalFileHelperRequest(
        userText: String,
        displayText: String,
        intent: FileSearchIntent
    ) {
        appendUserMessage(displayText, attachment = null)

        if (!_chat.value.localFileHelperEnabled && !hasLocalFileHelperPermission()) {
            _chat.update {
                it.copy(
                    messages = it.messages + ChatMessage(
                        id = System.nanoTime(),
                        role = Role.ASSISTANT,
                        content = "Turn on Local Insight in Settings to search files on your device."
                    )
                )
            }
            return
        }

        val placeholder = ChatMessage(
            id = System.nanoTime() + 1,
            role = Role.ASSISTANT,
            content = "Searching your device...",
            isStreaming = true
        )
        _chat.update {
            it.copy(messages = it.messages + placeholder, isGenerating = true, error = null)
        }

        viewModelScope.launch(Dispatchers.IO) {
            val results = if (intent.recentOnly) {
                LocalFileHelper.getRecentFiles(
                    context = getApplication(),
                    category = intent.category,
                    limit = 8
                )
            } else {
                LocalFileHelper.searchFiles(
                    context = getApplication(),
                    query = intent.searchTerm,
                    category = intent.category,
                    limit = 8
                )
            }

            val response = buildLocalFileResponse(intent, results)
            updateAssistantMessage(content = response, isStreaming = false, tokensPerSec = 0f)

            if (_chat.value.saveChat) {
                history.addLast(Pair(userText, response))
                if (history.size > 8) history.removeFirst()
                saveCurrentChat()
            }
        }
    }

    // ── Web Search Handler ──────────────────────────────────────────────────

    private fun handleWebSearchRequest(userText: String) {
        val query = WebSearchHelper.cleanQuery(userText)
        appendUserMessage(userText, attachment = null)

        val placeholder = ChatMessage(
            id = System.nanoTime() + 1,
            role = Role.ASSISTANT,
            content = "...",
            isStreaming = true
        )
        _chat.update {
            it.copy(
                messages = it.messages + placeholder,
                isGenerating = true,
                isSearchingWeb = true,
                error = null
            )
        }

        viewModelScope.launch(Dispatchers.IO) {
            val searchResult = WebSearchHelper.performSearch(query)
            _chat.update { it.copy(isSearchingWeb = false) }

            if (LlamaEngine.isModelLoaded() &&
                !searchResult.contains("failed", ignoreCase = true) &&
                !searchResult.contains("Could not reach", ignoreCase = true) &&
                !searchResult.contains("No clear results", ignoreCase = true)
            ) {
                val prompt = """
                    You are Nomad. A web search was performed for: "$query"
                    
                    Search results:
                    $searchResult
                    
                    Based ONLY on the search results above, give a concise and accurate answer to the user's request.
                    Do not add information beyond what the search results say.
                    If results don't fully answer the question, say so clearly.
                    User asked: $userText
                """.trimIndent()

                updateAssistantMessage(
                    content = "...",
                    isStreaming = true,
                    searchResults = searchResult
                )

                generateResponse(
                    userText = userText,
                    displayText = userText,
                    attachmentToUse = null,
                    fileCtx = null,
                    isInternalPrompt = true,
                    internalPrompt = prompt
                )
            } else {
                // No model or search failed — show raw results directly
                val response = if (searchResult.contains("failed", ignoreCase = true) ||
                    searchResult.contains("No clear", ignoreCase = true)
                ) {
                    searchResult
                } else {
                    "**Web results for \"$query\":**\n\n$searchResult"
                }

                updateAssistantMessage(
                    content = response,
                    isStreaming = false,
                    tokensPerSec = 0f,
                    searchResults = searchResult.takeUnless {
                        it.contains("failed", ignoreCase = true)
                    }
                )

                if (_chat.value.saveChat) {
                    history.addLast(Pair(userText, response))
                    if (history.size > 8) history.removeFirst()
                    saveCurrentChat()
                }
                _chat.update { it.copy(isGenerating = false) }
            }
        }
    }

    // ── File Result Builder — FIXED to show path ────────────────────────────
    /**
     * FIXED: Shows the actual file path/location so users know where their files are.
     * Read-only display — no delete or modify actions exposed.
     */
    private fun buildLocalFileResponse(
        intent: FileSearchIntent,
        results: List<FileMetadata>
    ): String {
        val categoryLabel = when (intent.category) {
            FileSearchCategory.ALL -> "files"
            FileSearchCategory.IMAGE -> "images"
            FileSearchCategory.DOCUMENT -> "documents"
            FileSearchCategory.AUDIO -> "audio files"
            FileSearchCategory.VIDEO -> "videos"
        }

        if (results.isEmpty()) {
            return if (intent.recentOnly) {
                "I couldn't find any recent $categoryLabel on this device."
            } else {
                "I couldn't find any $categoryLabel matching \"${intent.searchTerm}\".\n\n" +
                        "Try using a more general term, or check that storage permission is granted in Settings."
            }
        }

        return buildString {
            if (intent.recentOnly) {
                append("Here are your most recent $categoryLabel:\n\n")
            } else {
                append("Found ${results.size} $categoryLabel for \"${intent.searchTerm}\":\n\n")
            }

            results.forEachIndexed { index, file ->
                append("${index + 1}. **${file.name}**\n")
                // Show human-readable path
                append("   📂 ${file.path}\n")
                // Mime type if meaningful
                file.mimeType
                    ?.takeIf { it.isNotBlank() && it != "application/octet-stream" }
                    ?.let { append("   📋 $it\n") }
                // File size
                append("   💾 ${FileProcessor.formatSize(file.size)}\n")
                if (index < results.lastIndex) append("\n")
            }

            append("\nTip: Tap the 📎 button to attach a file and ask me to read or summarize it.")
        }
    }

    // ── Performance helpers ──────────────────────────────────────────────────

    private fun buildPerformanceProfile(
        state: ChatUiState,
        userText: String,
        attachment: Attachment?
    ): PerformanceProfile {
        val shortPrompt = attachment == null && userText.length <= 140 && !userText.contains('\n')
        val generationThreads = generationThreadsFor(state.performanceMode)
        val promptThreads = promptThreadsFor(state.performanceMode)
        val responseMode = state.responseMode
        val responseMaxTokens = when (responseMode) {
            ResponseMode.FAST -> if (shortPrompt) minOf(state.maxTokens, 512) else minOf(state.maxTokens, 1024)
            ResponseMode.BALANCED -> state.maxTokens
            ResponseMode.THINKING -> state.maxTokens
        }
        val responseTemperature = when (responseMode) {
            ResponseMode.FAST -> minOf(state.temperature, 0.55f)
            ResponseMode.BALANCED -> state.temperature
            ResponseMode.THINKING -> minOf(state.temperature, 0.5f)
        }
        val explicitThinking =
            responseMode == ResponseMode.THINKING && (state.loadedModel?.supportsThinking == true)

        return when (state.performanceMode) {
            PerformanceMode.HIGH -> PerformanceProfile(
                generationThreads = generationThreads, promptThreads = promptThreads,
                historyPairs = when (responseMode) {
                    ResponseMode.FAST -> if (attachment == null) 2 else 3
                    ResponseMode.BALANCED -> if (attachment == null) 3 else 4
                    ResponseMode.THINKING -> 4
                },
                uiUpdateIntervalMs = 24L, maxTokens = responseMaxTokens,
                temperature = responseTemperature, responseMode = responseMode,
                useExplicitThinking = explicitThinking
            )
            PerformanceMode.BALANCED -> PerformanceProfile(
                generationThreads = generationThreads, promptThreads = promptThreads,
                historyPairs = when (responseMode) {
                    ResponseMode.FAST -> 2; ResponseMode.BALANCED -> 4; ResponseMode.THINKING -> 5
                },
                uiUpdateIntervalMs = 40L, maxTokens = responseMaxTokens,
                temperature = responseTemperature, responseMode = responseMode,
                useExplicitThinking = explicitThinking
            )
            PerformanceMode.POWER_SAVER -> PerformanceProfile(
                generationThreads = generationThreads, promptThreads = promptThreads,
                historyPairs = if (responseMode == ResponseMode.THINKING) 3 else 2,
                uiUpdateIntervalMs = 72L,
                maxTokens = minOf(responseMaxTokens, if (responseMode == ResponseMode.THINKING) 256 else responseMaxTokens),
                temperature = responseTemperature, responseMode = responseMode,
                useExplicitThinking = explicitThinking
            )
        }
    }

    private fun applyRecommendedSettingsForModel(
        model: ModelInfo,
        stats: ModelManager.MemoryStats
    ): ChatUiState {
        val current = _chat.value
        val tightBudget = model.ramRequirementGb >= stats.effectiveRamBudgetGb * 0.9
        
        // Default to BALANCED unless budget is extremely tight
        val recommendedResponseMode = if (tightBudget) ResponseMode.FAST else ResponseMode.BALANCED
        
        val recommendedPerformanceMode = when {
            model.paramCountValue >= 7.0 || model.ramRequirementGb >= stats.effectiveRamBudgetGb * 0.8 -> PerformanceMode.HIGH
            model.paramCountValue <= 2.2 -> PerformanceMode.BALANCED
            else -> PerformanceMode.BALANCED
        }
        val recommendedContext = when {
            tightBudget -> 2048
            model.paramCountValue >= 7.0 -> 3072
            model.paramCountValue >= 3.0 -> 3072
            else -> 4096
        }
        
        // Default max tokens to 1600 as requested
        val recommendedMaxTokens = 1600

        val recommendedTemperature = when {
            model.name.contains("Phi-3", ignoreCase = true) -> 0.55f
            model.paramCountValue <= 2.2 -> 0.65f
            else -> 0.7f
        }
        val tuned = current.copy(
            useGpu = current.useGpu,
            temperature = recommendedTemperature,
            maxTokens = recommendedMaxTokens,
            contextLength = recommendedContext,
            performanceMode = recommendedPerformanceMode,
            responseMode = recommendedResponseMode
        )
        _chat.value = tuned
        persistRuntimeSettings(tuned)
        return tuned
    }

    private fun persistRuntimeSettings(state: ChatUiState) {
        getApplication<Application>()
            .getSharedPreferences("nomad_prefs", Context.MODE_PRIVATE)
            .edit()
            .putBoolean("use_gpu", state.useGpu)
            .putFloat("temperature", state.temperature)
            .putInt("max_tokens", state.maxTokens)
            .putString("perf_mode", state.performanceMode.name)
            .putString("response_mode", state.responseMode.name)
            .putInt("context_length", state.contextLength)
            .apply()
    }

    private fun buildSystemPromptForTurn(
        state: ChatUiState,
        profile: PerformanceProfile
    ): String {
        val languagePrompt = if (state.preferredLanguage == "English") state.systemPrompt
        else "${state.systemPrompt} You MUST respond entirely in ${state.preferredLanguage}."

        val modeDirective = when (profile.responseMode) {
            ResponseMode.FAST -> "Give the answer quickly, directly, and in the fewest helpful words."
            ResponseMode.BALANCED -> "Give a clear answer with practical detail, but avoid unnecessary length."
            ResponseMode.THINKING -> "Reason carefully before answering. For complex questions, work step by step, but keep the final answer concise."
        }
        return "$languagePrompt $modeDirective"
    }

    private fun generationThreadsFor(mode: PerformanceMode): Int {
        val available = Runtime.getRuntime().availableProcessors()
        return when (mode) {
            PerformanceMode.HIGH -> available.coerceIn(4, 8)
            PerformanceMode.BALANCED -> (available - 1).coerceIn(3, 6)
            PerformanceMode.POWER_SAVER -> 2
        }
    }

    private fun promptThreadsFor(mode: PerformanceMode): Int {
        val available = Runtime.getRuntime().availableProcessors()
        return when (mode) {
            PerformanceMode.HIGH -> available
            PerformanceMode.BALANCED -> available
            PerformanceMode.POWER_SAVER -> maxOf(2, available / 2)
        }
    }

    private fun reloadLoadedModelIfNeeded() {
        _chat.value.loadedModel?.let { loadModel(it) }
    }

    private fun hasLocalFileHelperPermission(): Boolean {
        val context = getApplication<Application>()
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            listOf(
                Manifest.permission.READ_MEDIA_IMAGES,
                Manifest.permission.READ_MEDIA_VIDEO,
                Manifest.permission.READ_MEDIA_AUDIO
            ).any { perm ->
                ContextCompat.checkSelfPermission(context, perm) == PackageManager.PERMISSION_GRANTED
            }
        } else {
            ContextCompat.checkSelfPermission(
                context, Manifest.permission.READ_EXTERNAL_STORAGE
            ) == PackageManager.PERMISSION_GRANTED
        }
    }

    override fun onCleared() {
        super.onCleared()
        LlamaEngine.unloadModel()
    }
}