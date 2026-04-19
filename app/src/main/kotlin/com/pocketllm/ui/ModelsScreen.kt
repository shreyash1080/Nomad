package com.pocketllm.ui

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.pocketllm.data.DownloadState
import com.pocketllm.data.ModelInfo
import com.pocketllm.viewmodel.ChatViewModel
import com.pocketllm.viewmodel.ModelsUiState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelsScreen(viewModel: ChatViewModel) {
    val state     by viewModel.modelsState.collectAsState()
    val chatState by viewModel.chatState.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Models") })
        }
    ) { padding ->
        LazyColumn(
            modifier        = Modifier
                .fillMaxSize()
                .padding(padding),
            contentPadding  = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Section: RAM guide
            item {
                RamGuideCard()
                Spacer(Modifier.height(8.dp))
                Text(
                    "Available Models",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
            }

            items(state.models, key = { it.id }) { model ->
                ModelCard(
                    model       = model,
                    isLoaded    = state.loadedModelId == model.id,
                    isDownloaded = viewModel.modelManager.isDownloaded(model),
                    downloadState = state.downloadStates[model.id],
                    onDownload  = { viewModel.downloadModel(model) },
                    onCancelDownload = { viewModel.cancelDownload(model) },
                    onLoad      = { viewModel.loadModel(model) },
                    onUnload    = { viewModel.unloadModel() },
                    onDelete    = { viewModel.deleteModel(model) },
                    isLoadingAny = chatState.loadingModel
                )
            }
        }
    }
}

@Composable
private fun RamGuideCard() {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        )
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Memory, contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(8.dp))
                Text("RAM Guide", style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.height(8.dp))
            RamRow("8 GB phone",  "Phi-3 Mini, Gemma 2 2B, TinyLlama")
            RamRow("12 GB phone", "+ Llama 3.2 3B")
            RamRow("16 GB phone", "+ Qwen 2.5 7B, Mistral 7B")
            RamRow("24 GB phone", "+ Llama 3.1 8B  (near-desktop quality)")
        }
    }
}

@Composable
private fun RamRow(ram: String, models: String) {
    Row(Modifier.padding(vertical = 2.dp)) {
        Text("$ram: ", style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold)
        Text(models, style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun ModelCard(
    model: ModelInfo,
    isLoaded: Boolean,
    isDownloaded: Boolean,
    downloadState: DownloadState?,
    onDownload: () -> Unit,
    onCancelDownload: () -> Unit,
    onLoad: () -> Unit,
    onUnload: () -> Unit,
    onDelete: () -> Unit,
    isLoadingAny: Boolean
) {
    val isDownloading = downloadState is DownloadState.Progress ||
                        downloadState is DownloadState.Connecting

    Card(
        modifier = Modifier.fillMaxWidth(),
        border   = if (isLoaded)
            BorderStroke(2.dp, MaterialTheme.colorScheme.primary)
        else null
    ) {
        Column(Modifier.padding(16.dp)) {
            // Header row
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(model.name, style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold)
                        if (model.isRecommended) {
                            Spacer(Modifier.width(6.dp))
                            AssistChip(
                                onClick = {},
                                label   = { Text("⭐ Recommended", style = MaterialTheme.typography.labelSmall) },
                                modifier = Modifier.height(24.dp)
                            )
                        }
                        if (isLoaded) {
                            Spacer(Modifier.width(6.dp))
                            AssistChip(
                                onClick = {},
                                label   = { Text("● Loaded", style = MaterialTheme.typography.labelSmall) },
                                colors  = AssistChipDefaults.assistChipColors(
                                    containerColor = MaterialTheme.colorScheme.primaryContainer
                                ),
                                modifier = Modifier.height(24.dp)
                            )
                        }
                    }
                    Spacer(Modifier.height(2.dp))
                    Text(model.description,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }

            Spacer(Modifier.height(8.dp))

            // Chips: size, params, quant, RAM
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                InfoChip(Icons.Default.Storage,    model.sizeLabel)
                InfoChip(Icons.Default.Psychology, model.paramCount)
                InfoChip(Icons.Default.Compress,   model.quantization)
                InfoChip(Icons.Default.Memory,     model.ramRequired)
            }

            // Download progress
            if (downloadState != null) {
                val progress = (downloadState as? DownloadState.Progress)
                Column(Modifier.padding(top = 8.dp)) {
                    if (downloadState is DownloadState.Connecting) {
                        // Using a simple text-based indicator to avoid the broken M3 LinearProgressIndicator
                        Text("Connecting to server...", 
                             style = MaterialTheme.typography.labelSmall,
                             color = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.height(4.dp))
                        Box(Modifier.fillMaxWidth().height(4.dp).background(MaterialTheme.colorScheme.surfaceVariant))
                    } else if (progress != null) {
                        // Custom progress bar to avoid the NoSuchMethodError in current M3 version
                        val pct = (progress.percent / 100f).coerceIn(0f, 1f)
                        Column {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(8.dp)
                                    .background(
                                        MaterialTheme.colorScheme.surfaceVariant,
                                        shape = MaterialTheme.shapes.extraSmall
                                    )
                            ) {
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth(pct)
                                        .fillMaxHeight()
                                        .background(
                                            MaterialTheme.colorScheme.primary,
                                            shape = MaterialTheme.shapes.extraSmall
                                        )
                                )
                            }
                            Spacer(Modifier.height(4.dp))
                            Text(
                                "${progress.percent}%  (${formatBytes(progress.bytesDownloaded)} / ${formatBytes(progress.totalBytes)})",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    } else if (downloadState is DownloadState.Error) {
                        Text("Error: ${downloadState.message}", 
                             color = MaterialTheme.colorScheme.error,
                             style = MaterialTheme.typography.labelSmall)
                    }
                }
            }

            Spacer(Modifier.height(12.dp))

            // Action buttons
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                when {
                    isLoaded -> {
                        OutlinedButton(onClick = onUnload, modifier = Modifier.weight(1f)) {
                            Icon(Icons.Default.Eject, null, Modifier.size(16.dp))
                            Spacer(Modifier.width(4.dp))
                            Text("Unload")
                        }
                        OutlinedButton(onClick = onDelete, modifier = Modifier.weight(1f),
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = MaterialTheme.colorScheme.error)) {
                            Icon(Icons.Default.Delete, null, Modifier.size(16.dp))
                            Spacer(Modifier.width(4.dp))
                            Text("Delete")
                        }
                    }
                    isDownloaded -> {
                        Button(onClick = onLoad,
                            enabled  = !isLoadingAny,
                            modifier = Modifier.weight(1f)) {
                            Icon(Icons.Default.PlayArrow, null, Modifier.size(16.dp))
                            Spacer(Modifier.width(4.dp))
                            Text("Load")
                        }
                        OutlinedButton(onClick = onDelete, modifier = Modifier.weight(1f),
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = MaterialTheme.colorScheme.error)) {
                            Icon(Icons.Default.Delete, null, Modifier.size(16.dp))
                            Spacer(Modifier.width(4.dp))
                            Text("Delete")
                        }
                    }
                    isDownloading -> {
                        OutlinedButton(onClick = onCancelDownload, modifier = Modifier.weight(1f)) {
                            SimpleSpinner(Modifier.size(16.dp))
                            Spacer(Modifier.width(8.dp))
                            Text("Cancel Download")
                        }
                    }
                    else -> {
                        Button(onClick = onDownload, modifier = Modifier.weight(1f)) {
                            Icon(Icons.Default.Download, null, Modifier.size(16.dp))
                            Spacer(Modifier.width(4.dp))
                            Text("Download  ${model.sizeLabel}")
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun InfoChip(icon: androidx.compose.ui.graphics.vector.ImageVector, label: String) {
    SuggestionChip(
        onClick = {},
        label   = { Text(label, style = MaterialTheme.typography.labelSmall) },
        icon    = { Icon(icon, null, Modifier.size(12.dp)) },
        modifier = Modifier.height(28.dp)
    )
}

@Composable
private fun SimpleSpinner(modifier: Modifier = Modifier) {
    val infiniteTransition = rememberInfiniteTransition(label = "spinner")
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "rotation"
    )

    val color = MaterialTheme.colorScheme.primary
    Canvas(modifier = modifier) {
        val strokeWidth = 2.dp.toPx()
        drawArc(
            color = color,
            startAngle = rotation,
            sweepAngle = 270f,
            useCenter = false,
            style = androidx.compose.ui.graphics.drawscope.Stroke(width = strokeWidth)
        )
    }
}

private fun formatBytes(bytes: Long): String = when {
    bytes < 1024             -> "${bytes} B"
    bytes < 1_048_576        -> "${"%.1f".format(bytes / 1024.0)} KB"
    bytes < 1_073_741_824    -> "${"%.1f".format(bytes / 1_048_576.0)} MB"
    else                     -> "${"%.2f".format(bytes / 1_073_741_824.0)} GB"
}
