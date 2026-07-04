package com.nomad.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nomad.engine.FileMatch
import com.nomad.engine.FileSearchEngine
import com.nomad.engine.LlamaInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed interface FileSearchUiState {
    data object Idle : FileSearchUiState

    data class Scanning(
        val scanned: Int,
        val total: Int,
        val phase: String = "Scanning files…",
        val isIndexing: Boolean = false
    ) : FileSearchUiState

    data class Results(
        val matches: List<FileMatch>,
        val query: String,
        val totalScanned: Int
    ) : FileSearchUiState

    data class Error(val message: String) : FileSearchUiState
}

class FileSearchViewModel(application: Application) : AndroidViewModel(application) {

    private val engine = FileSearchEngine(application)

    private val _uiState = MutableStateFlow<FileSearchUiState>(FileSearchUiState.Idle)
    val uiState: StateFlow<FileSearchUiState> = _uiState.asStateFlow()

    private val _description = MutableStateFlow("")
    val description: StateFlow<String> = _description.asStateFlow()

    var llmInference: LlamaInference? = null

    private var searchJob: Job? = null

    fun updateDescription(text: String) {
        _description.value = text
    }

    fun startSearch() {
        val query = _description.value.trim()
        if (query.isBlank()) return

        searchJob?.cancel()
        searchJob = viewModelScope.launch(Dispatchers.IO) {
            val db = com.nomad.data.FileIndexDatabase.getInstance(getApplication())
            val indexCount = db.dao().count()

            if (indexCount == 0) {
                // If index is empty, fall back to slow scan but also trigger background indexing
                com.nomad.data.scheduleIndexing(getApplication())
                _uiState.value = FileSearchUiState.Scanning(0, 0, "Building index for the first time…", isIndexing = true)
                // We'll proceed with a legacy scan for immediate results
                runLegacySearch(query)
            } else {
                _uiState.value = FileSearchUiState.Scanning(0, 0, "Searching index…")
                runCatching {
                    val results = engine.search(
                        description = query,
                        db = db,
                        llmInference = llmInference,
                        onProgress = { phase ->
                            _uiState.value = FileSearchUiState.Scanning(0, 0, phase)
                        }
                    )
                    _uiState.value = FileSearchUiState.Results(
                        matches = results,
                        query = query,
                        totalScanned = indexCount
                    )
                }.onFailure { e ->
                    _uiState.value = FileSearchUiState.Error(e.message ?: "Unknown error")
                }
            }
        }
    }

    private suspend fun runLegacySearch(query: String) {
        runCatching {
            val results = engine.search(
                description = query,
                llmInference = llmInference,
                onProgress = { scanned, total ->
                    _uiState.value = FileSearchUiState.Scanning(
                        scanned = scanned,
                        total = total,
                        phase = if (total == 0) "Discovering files…" else "Legacy Scan: $scanned / $total",
                        isIndexing = true
                    )
                }
            )
            _uiState.value = FileSearchUiState.Results(
                matches = results,
                query = query,
                totalScanned = ((_uiState.value as? FileSearchUiState.Scanning)?.total ?: 0)
            )
        }.onFailure { e ->
            _uiState.value = FileSearchUiState.Error(e.message ?: "Search failed: ${e.message}")
        }
    }

    fun cancelSearch() {
        searchJob?.cancel()
        _uiState.value = FileSearchUiState.Idle
    }

    fun reset() {
        searchJob?.cancel()
        _description.value = ""
        _uiState.value = FileSearchUiState.Idle
    }

    fun openFile(match: FileMatch, context: android.content.Context) {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val uri = androidx.core.content.FileProvider.getUriForFile(
                    context,
                    "${context.packageName}.fileprovider",
                    match.file
                )
                val intent = android.content.Intent(android.content.Intent.ACTION_VIEW).apply {
                    setDataAndType(uri, resolveMimeType(match))
                    flags = android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION or
                            android.content.Intent.FLAG_ACTIVITY_NEW_TASK
                }
                context.startActivity(intent)
            }
        }
    }

    fun loadIntoChat(match: FileMatch): String {
        val preview = match.extractedText.take(500).ifBlank { "(binary file – no text preview)" }
        return buildString {
            appendLine("📎 **${match.file.name}**")
            appendLine("Type: ${match.mimeCategory}  |  Size: ${match.file.length().toHumanSize()}")
            appendLine()
            appendLine("**Content preview:**")
            appendLine(preview)
        }
    }

    private fun resolveMimeType(match: FileMatch): String = when (match.mimeCategory) {
        "image"        -> "image/*"
        "pdf"          -> "application/pdf"
        "document"     -> "application/msword"
        "spreadsheet"  -> "application/vnd.ms-excel"
        "text"         -> "text/plain"
        else           -> "*/*"
    }

    private fun Long.toHumanSize(): String = when {
        this < 1_024              -> "$this B"
        this < 1_048_576          -> "${this / 1_024} KB"
        this < 1_073_741_824      -> "${this / 1_048_576} MB"
        else                      -> "${this / 1_073_741_824} GB"
    }
}
