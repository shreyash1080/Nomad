package com.pocketllm.ui

import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Base64
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.pocketllm.data.Attachment
import com.pocketllm.data.AttachmentType
import com.pocketllm.data.ChatMessage
import com.pocketllm.data.Role
import com.pocketllm.data.FileProcessor
import com.pocketllm.viewmodel.ChatUiState
import com.pocketllm.viewmodel.ChatViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    val state     by viewModel.chatState.collectAsState()
    val listState = rememberLazyListState()
    var inputText by remember { mutableStateOf("") }

    val filePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? -> uri?.let { viewModel.attachFile(it) } }

    LaunchedEffect(state.messages.size) {
        if (state.messages.isNotEmpty())
            listState.animateScrollToItem(state.messages.lastIndex)
    }

    Scaffold(
        topBar = { ChatTopBar(state, onClear = { viewModel.clearChat() }) },
        bottomBar = {
            Surface(tonalElevation = 8.dp) {
                Column {
                    AnimatedVisibility(visible = state.pendingAttachment != null) {
                        state.pendingAttachment?.let { att ->
                            AttachmentChip(
                                attachment = att,
                                onRemove   = { viewModel.clearAttachment() }
                            )
                        }
                    }
                    if (state.isProcessingFile) {
                        Box(
                            Modifier
                                .fillMaxWidth()
                                .height(4.dp)
                                .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.2f))
                        ) {
                            Box(
                                Modifier
                                    .fillMaxWidth(0.2f)
                                    .fillMaxHeight()
                                    .background(MaterialTheme.colorScheme.primary)
                            )
                        }
                    }
                    ChatInputBar(
                        text         = inputText,
                        onTextChange = { inputText = it },
                        isGenerating = state.isGenerating,
                        onSend       = {
                            if (inputText.isNotBlank()) {
                                viewModel.sendMessage(inputText)
                                inputText = ""
                            }
                        },
                        onStop       = { viewModel.stopGeneration() },
                        onAttach     = { filePicker.launch("*/*") }
                    )
                }
            }
        }
    ) { padding ->
        Box(Modifier.fillMaxSize().padding(padding).background(
            Brush.verticalGradient(listOf(
                MaterialTheme.colorScheme.surface,
                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)
            ))
        )) {
            if (state.messages.isEmpty()) {
                WelcomeContent(state)
            } else {
                LazyColumn(
                    state           = listState,
                    modifier        = Modifier.fillMaxSize(),
                    contentPadding  = PaddingValues(vertical = 16.dp)
                ) {
                    items(state.messages, key = { it.id }) { msg ->
                        MessageBubble(msg)
                    }
                    if (state.tokensPerSec > 0 && !state.isGenerating) {
                        item {
                            Text(
                                "⚡ ${"%.1f".format(state.tokensPerSec)} tokens/sec",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.primary.copy(alpha = 0.7f),
                                modifier = Modifier.padding(horizontal = 24.dp, vertical = 4.dp)
                            )
                        }
                    }
                }
            }
            if (state.loadingModel) LoadingOverlay(state.loadProgress)
            
            state.error?.let { err ->
                Snackbar(
                    modifier = Modifier.align(Alignment.BottomCenter).padding(16.dp),
                    containerColor = MaterialTheme.colorScheme.errorContainer,
                    contentColor = MaterialTheme.colorScheme.onErrorContainer
                ) { Text(err) }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChatTopBar(state: ChatUiState, onClear: () -> Unit) {
    CenterAlignedTopAppBar(
        title = {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("PocketLLM", style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold))
                Text(
                    state.loadedModel?.name ?: "No model loaded",
                    style = MaterialTheme.typography.labelSmall,
                    color = if (state.loadedModel != null)
                        MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.error
                )
            }
        },
        actions = {
            if (state.messages.isNotEmpty())
                IconButton(onClick = onClear) { 
                    Icon(Icons.Default.DeleteSweep, "Clear Chat", tint = MaterialTheme.colorScheme.onSurfaceVariant) 
                }
        },
        colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.95f)
        )
    )
}

@Composable
private fun AttachmentChip(attachment: Attachment, onRemove: () -> Unit) {
    Card(
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
        modifier = Modifier.padding(12.dp).fillMaxWidth()
    ) {
        Row(
            Modifier.padding(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (attachment.type == AttachmentType.IMAGE && attachment.thumbnailBase64 != null) {
                val bytes = Base64.decode(attachment.thumbnailBase64, Base64.DEFAULT)
                val bmp   = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                if (bmp != null) {
                    Image(
                        bitmap       = bmp.asImageBitmap(),
                        contentDescription = null,
                        modifier     = Modifier.size(48.dp).clip(RoundedCornerShape(8.dp)),
                        contentScale = ContentScale.Crop
                    )
                    Spacer(Modifier.width(12.dp))
                }
            } else {
                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = MaterialTheme.colorScheme.primaryContainer,
                    modifier = Modifier.size(48.dp)
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            imageVector = when (attachment.type) {
                                AttachmentType.PDF   -> Icons.Default.PictureAsPdf
                                AttachmentType.EXCEL -> Icons.Default.TableChart
                                AttachmentType.WORD  -> Icons.Default.Description
                                AttachmentType.TEXT  -> Icons.Default.Article
                                else                 -> Icons.Default.AttachFile
                            },
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onPrimaryContainer,
                            modifier = Modifier.size(24.dp)
                        )
                    }
                }
                Spacer(Modifier.width(12.dp))
            }

            Column(Modifier.weight(1f)) {
                Text(attachment.fileName,
                    style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
                    maxLines = 1)
                Text(FileProcessor.formatSize(attachment.fileSizeBytes),
                    style = MaterialTheme.typography.labelSmall)
            }
            IconButton(onClick = onRemove) {
                Icon(Icons.Default.Cancel, "Remove")
            }
        }
    }
}

@Composable
private fun MessageBubble(msg: ChatMessage) {
    if (msg.role == Role.SYSTEM) {
        Box(
            Modifier.fillMaxWidth().padding(horizontal = 32.dp, vertical = 8.dp),
            contentAlignment = Alignment.Center
        ) {
            Surface(
                color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.5f),
                shape = CircleShape
            ) {
                Text(msg.content,
                    style     = MaterialTheme.typography.labelMedium,
                    modifier  = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
                    textAlign = TextAlign.Center)
            }
        }
        return
    }

    val isUser = msg.role == Role.USER
    Row(
        modifier            = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        if (!isUser) {
            Surface(
                modifier = Modifier.size(36.dp),
                shape = CircleShape,
                color = MaterialTheme.colorScheme.primaryContainer
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(Icons.Default.SmartToy, "AI", modifier = Modifier.size(20.dp), tint = MaterialTheme.colorScheme.onPrimaryContainer)
                }
            }
            Spacer(Modifier.width(10.dp))
        }

        Column(horizontalAlignment = if (isUser) Alignment.End else Alignment.Start) {
            msg.attachment?.let { att ->
                if (att.type == AttachmentType.IMAGE && att.thumbnailBase64 != null) {
                    val bytes = Base64.decode(att.thumbnailBase64, Base64.DEFAULT)
                    val bmp   = try { BitmapFactory.decodeByteArray(bytes, 0, bytes.size) } catch(e: Exception) { null }
                    if (bmp != null) {
                        Image(
                            bitmap            = bmp.asImageBitmap(),
                            contentDescription = "Attached image",
                            modifier          = Modifier
                                .widthIn(max = 240.dp)
                                .clip(RoundedCornerShape(16.dp))
                                .padding(bottom = 4.dp),
                            contentScale      = ContentScale.FillWidth
                        )
                    }
                }
            }

            Surface(
                shape = RoundedCornerShape(
                    topStart    = if (isUser) 20.dp else 4.dp,
                    topEnd      = if (isUser) 4.dp  else 20.dp,
                    bottomStart = 20.dp,
                    bottomEnd   = 20.dp
                ),
                color    = if (isUser) MaterialTheme.colorScheme.primary
                else MaterialTheme.colorScheme.surfaceVariant,
                tonalElevation = if (isUser) 2.dp else 0.dp,
                modifier = Modifier.widthIn(max = 320.dp)
            ) {
                Column(Modifier.padding(horizontal = 16.dp, vertical = 12.dp)) {
                    Text(
                        text       = msg.content,
                        color      = if (isUser) MaterialTheme.colorScheme.onPrimary
                        else MaterialTheme.colorScheme.onSurfaceVariant,
                        style      = MaterialTheme.typography.bodyLarge,
                        fontFamily = if (msg.content.contains("```")) FontFamily.Monospace
                        else FontFamily.Default
                    )
                    if (msg.isStreaming) {
                        Spacer(Modifier.height(8.dp))
                        Box(
                            Modifier
                                .width(40.dp)
                                .height(3.dp)
                                .clip(CircleShape)
                                .background(
                                    if (isUser) MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.4f)
                                    else MaterialTheme.colorScheme.primary.copy(alpha = 0.4f)
                                )
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ChatInputBar(
    text: String,
    onTextChange: (String) -> Unit,
    isGenerating: Boolean,
    onSend: () -> Unit,
    onStop: () -> Unit,
    onAttach: () -> Unit
) {
    Surface(tonalElevation = 2.dp) {
        Row(
            Modifier.fillMaxWidth().padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(
                onClick = onAttach,
                enabled = !isGenerating,
                colors = IconButtonDefaults.iconButtonColors(contentColor = MaterialTheme.colorScheme.primary)
            ) {
                Icon(Icons.Default.AddCircle, "Attach", modifier = Modifier.size(28.dp))
            }

            TextField(
                value         = text,
                onValueChange = onTextChange,
                placeholder   = { Text("Ask PocketLLM...") },
                modifier      = Modifier.weight(1f).clip(RoundedCornerShape(28.dp)),
                colors        = TextFieldDefaults.colors(
                    focusedIndicatorColor = Color.Transparent,
                    unfocusedIndicatorColor = Color.Transparent,
                    disabledIndicatorColor = Color.Transparent
                ),
                maxLines      = 6,
                enabled       = !isGenerating
            )
            
            Spacer(Modifier.width(8.dp))

            if (isGenerating) {
                FilledIconButton(
                    onClick  = onStop,
                    modifier = Modifier.size(48.dp),
                    colors = IconButtonDefaults.filledIconButtonColors(containerColor = MaterialTheme.colorScheme.error)
                ) {
                    Icon(Icons.Default.Stop, "Stop")
                }
            } else {
                FilledIconButton(
                    onClick  = onSend,
                    enabled  = text.isNotBlank(),
                    modifier = Modifier.size(48.dp),
                    colors = IconButtonDefaults.filledIconButtonColors(
                        containerColor = MaterialTheme.colorScheme.primary
                    )
                ) {
                    Icon(Icons.Default.ArrowUpward, "Send")
                }
            }
        }
    }
}

@Composable
private fun WelcomeContent(state: ChatUiState) {
    Column(
        Modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Surface(
            modifier = Modifier.size(100.dp),
            shape = CircleShape,
            color = MaterialTheme.colorScheme.primaryContainer
        ) {
            Box(contentAlignment = Alignment.Center) {
                Icon(Icons.Default.AutoAwesome, null,
                    modifier = Modifier.size(50.dp),
                    tint     = MaterialTheme.colorScheme.primary)
            }
        }
        Spacer(Modifier.height(24.dp))
        Text("PocketLLM",
            style = MaterialTheme.typography.displaySmall.copy(fontWeight = FontWeight.Black),
            color = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.height(8.dp))
        Text(
            if (state.loadedModel != null)
                "Private, Offline AI for your device.\nAttach files or just start typing."
            else
                "Let's get started. Head over to the Models tab to download a language model.",
            style     = MaterialTheme.typography.bodyLarge,
            textAlign = TextAlign.Center,
            color     = MaterialTheme.colorScheme.onSurfaceVariant
        )
        if (state.loadedModel != null) {
            Spacer(Modifier.height(32.dp))
            Text("TRY ATTACHING:", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.outline)
            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf("📄 PDF", "📊 Excel", "🖼️ Photo").forEach { label ->
                    AssistChip(onClick = {}, label = { Text(label) }, leadingIcon = { Icon(Icons.Default.FileUpload, null, Modifier.size(16.dp)) })
                }
            }
        }
    }
}

@Composable
private fun LoadingOverlay(progress: Float) {
    Box(
        Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.7f)),
        contentAlignment = Alignment.Center
    ) {
        Card(
            shape = RoundedCornerShape(28.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
        ) {
            Column(Modifier.padding(32.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Box(contentAlignment = Alignment.Center) {
                    Box(Modifier.size(64.dp)) {
                        // Custom Static Progress Circle
                        Canvas(modifier = Modifier.fillMaxSize()) {
                            val sw = 15f
                            drawArc(
                                color = Color.LightGray.copy(alpha = 0.3f),
                                startAngle = 0f,
                                sweepAngle = 360f,
                                useCenter = false,
                                style = androidx.compose.ui.graphics.drawscope.Stroke(width = sw)
                            )
                            drawArc(
                                color = Color(0xFF2196F3),
                                startAngle = -90f,
                                sweepAngle = 360f * progress,
                                useCenter = false,
                                style = androidx.compose.ui.graphics.drawscope.Stroke(width = sw, cap = androidx.compose.ui.graphics.StrokeCap.Round)
                            )
                        }
                    }
                }
                Spacer(Modifier.height(24.dp))
                Text("Initializing Engine...", style = MaterialTheme.typography.titleMedium)
                Text("${(progress * 100).toInt()}%", style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Bold)
            }
        }
    }
}
