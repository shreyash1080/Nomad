package com.pocketllm.ui

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
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
            CenterAlignedTopAppBar(
                title = { Text("Model Hub", fontWeight = FontWeight.Bold) },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { padding ->
        LazyColumn(
            modifier        = Modifier
                .fillMaxSize()
                .padding(padding)
                .background(MaterialTheme.colorScheme.surface),
            contentPadding  = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            item {
                DeviceStatusCard(state.totalDeviceRamGb)
            }

            item {
                Text(
                    "Recommended for You",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.ExtraBold,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(top = 8.dp, bottom = 4.dp)
                )
            }

            items(state.models, key = { it.id }) { model ->
                val isSafe = model.ramRequirementGb <= state.totalDeviceRamGb * 0.9
                ModelCard(
                    model       = model,
                    isLoaded    = state.loadedModelId == model.id,
                    isDownloaded = viewModel.modelManager.isDownloaded(model),
                    downloadState = state.downloadStates[model.id],
                    onDownload  = { viewModel.downloadModel(model) },
                    onCancelDownload = { viewModel.cancelDownload(model.id) },
                    onLoad      = { viewModel.loadModel(model) },
                    onUnload    = { viewModel.unloadModel() },
                    onDelete    = { viewModel.deleteModel(model) },
                    isLoadingAny = chatState.loadingModel,
                    isSafe      = isSafe
                )
            }
        }
    }
}

@Composable
private fun DeviceStatusCard(totalRamGb: Double) {
    Card(
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.4f)
        ),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Row(
            Modifier.padding(20.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Surface(
                modifier = Modifier.size(48.dp),
                shape = CircleShape,
                color = MaterialTheme.colorScheme.primary
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(Icons.Default.Memory, null, tint = MaterialTheme.colorScheme.onPrimary)
                }
            }
            Spacer(Modifier.width(16.dp))
            Column {
                Text("Device Performance", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
                Text(
                    "Total RAM: ${"%.1f".format(totalRamGb)} GB",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    if (totalRamGb < 6) "Budget device - stick to Small models" 
                    else if (totalRamGb < 12) "Mid-range device - 3B models recommended"
                    else "High-end device - Large models supported",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
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
    isLoadingAny: Boolean,
    isSafe: Boolean
) {
    val isDownloading = downloadState is DownloadState.Progress ||
                        downloadState is DownloadState.Connecting

    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        shape    = RoundedCornerShape(20.dp),
        colors   = if (isLoaded) 
            CardDefaults.elevatedCardColors(containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f))
            else CardDefaults.elevatedCardColors()
    ) {
        Column(Modifier.padding(20.dp)) {
            Row(verticalAlignment = Alignment.Top) {
                Column(Modifier.weight(1f)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(model.name, style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.ExtraBold)
                        if (model.isRecommended) {
                            Spacer(Modifier.width(8.dp))
                            Surface(
                                color = MaterialTheme.colorScheme.tertiaryContainer,
                                shape = CircleShape
                            ) {
                                Text("⭐ BEST", 
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
                                    style = MaterialTheme.typography.labelSmall,
                                    fontWeight = FontWeight.Bold,
                                    color = MaterialTheme.colorScheme.onTertiaryContainer)
                            }
                        }
                    }
                    Spacer(Modifier.height(4.dp))
                    Text(model.description,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                
                if (!isSafe) {
                    Surface(
                        color = MaterialTheme.colorScheme.errorContainer,
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Icon(Icons.Default.Warning, "Incompatible", 
                             modifier = Modifier.padding(6.dp).size(20.dp),
                             tint = MaterialTheme.colorScheme.error)
                    }
                }
            }

            Spacer(Modifier.height(16.dp))

            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                InfoBadge(Icons.Default.Storage,    model.sizeLabel)
                InfoBadge(Icons.Default.Psychology, model.paramCount)
                InfoBadge(Icons.Default.Memory,     model.ramRequired)
            }

            if (downloadState != null) {
                DownloadProgressSection(downloadState, onCancelDownload)
            }

            Spacer(Modifier.height(20.dp))

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                if (isLoaded) {
                    Button(onClick = onUnload, modifier = Modifier.weight(1f), 
                           colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary)) {
                        Icon(Icons.Default.PowerSettingsNew, null, Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text("Unload")
                    }
                } else if (isDownloaded) {
                    Button(onClick = onLoad,
                        enabled  = !isLoadingAny && isSafe,
                        modifier = Modifier.weight(1f)) {
                        Icon(Icons.Default.Bolt, null, Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(if (isSafe) "Load Model" else "Too Heavy")
                    }
                } else if (isDownloading) {
                    // Handled by progress section
                } else {
                    Button(
                        onClick = onDownload, 
                        modifier = Modifier.weight(1f),
                        enabled = isSafe
                    ) {
                        Icon(Icons.Default.CloudDownload, null, Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(if (isSafe) "Get Model" else "High RAM Required")
                    }
                }

                if (isDownloaded || isLoaded) {
                    IconButton(
                        onClick = onDelete,
                        colors = IconButtonDefaults.iconButtonColors(contentColor = MaterialTheme.colorScheme.error)
                    ) {
                        Icon(Icons.Default.DeleteOutline, "Delete")
                    }
                }
            }
            
            if (!isSafe) {
                Text(
                    "⚠️ This model requires ${model.ramRequired} which exceeds your device's safety limit. Loading it might cause system instability or crashes.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(top = 12.dp)
                )
            }
        }
    }
}

@Composable
private fun DownloadProgressSection(state: DownloadState, onCancel: () -> Unit) {
    Column(Modifier.padding(top = 16.dp)) {
        val pct = if (state is DownloadState.Progress) state.percent / 100f else 0f
        
        Box(
            Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.surfaceVariant)
        ) {
            Box(
                Modifier
                    .fillMaxWidth(pct.coerceIn(0f, 1f))
                    .fillMaxHeight()
                    .background(MaterialTheme.colorScheme.primary)
            )
        }
        
        Row(
            Modifier.fillMaxWidth().padding(top = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                when (state) {
                    is DownloadState.Connecting -> "Connecting..."
                    is DownloadState.Progress -> "${state.percent}% (${formatBytes(state.bytesDownloaded)} / ${formatBytes(state.totalBytes)})"
                    is DownloadState.Error -> "Error: ${state.message}"
                    else -> ""
                },
                style = MaterialTheme.typography.labelMedium,
                color = if (state is DownloadState.Error) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
            )
            
            TextButton(onClick = onCancel, colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)) {
                Text("Cancel")
            }
        }
    }
}

@Composable
private fun InfoBadge(icon: androidx.compose.ui.graphics.vector.ImageVector, label: String) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        shape = RoundedCornerShape(8.dp)
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(icon, null, Modifier.size(14.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.width(6.dp))
            Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

private fun formatBytes(bytes: Long): String = when {
    bytes < 1024             -> "${bytes} B"
    bytes < 1_048_576        -> "${"%.1f".format(bytes / 1024.0)} KB"
    bytes < 1_073_741_824    -> "${"%.1f".format(bytes / 1_048_576.0)} MB"
    else                     -> "${"%.2f".format(bytes / 1_073_741_824.0)} GB"
}
