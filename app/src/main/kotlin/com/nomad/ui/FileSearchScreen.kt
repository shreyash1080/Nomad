package com.nomad.ui

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.*
import androidx.compose.material.icons.automirrored.outlined.*
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nomad.engine.FileMatch
import com.nomad.engine.LlamaInference
import com.nomad.viewmodel.FileSearchUiState
import com.nomad.viewmodel.FileSearchViewModel
import java.text.SimpleDateFormat
import java.util.*

private val Black       = Color(0xFF000000)
private val Surface1    = Color(0xFF0D0D0D)
private val Surface2    = Color(0xFF1A1A1A)
private val Surface3    = Color(0xFF252525)
private val Accent      = Color(0xFF5E9AFF)
private val AccentDim   = Color(0xFF2A4070)
private val TextPrimary = Color(0xFFEEEEEE)
private val TextSecond  = Color(0xFF888888)
private val Success     = Color(0xFF4CAF50)
private val DividerCol  = Color(0xFF1F1F1F)

@Composable
fun FileSearchDialog(
    llmInference: LlamaInference?,
    onFileLoadedToChat: (String) -> Unit,
    onDismiss: () -> Unit
) {
    val vm: FileSearchViewModel = viewModel()

    LaunchedEffect(llmInference) { vm.llmInference = llmInference }

    Dialog(
        onDismissRequest = { vm.reset(); onDismiss() },
        properties = DialogProperties(usePlatformDefaultWidth = false, dismissOnBackPress = true)
    ) {
        FileSearchScreen(
            vm = vm,
            onFileLoadedToChat = { content ->
                onFileLoadedToChat(content)
                vm.reset()
                onDismiss()
            },
            onDismiss = { vm.reset(); onDismiss() }
        )
    }
}

@Composable
fun FileSearchScreen(
    vm: FileSearchViewModel,
    onFileLoadedToChat: (String) -> Unit,
    onDismiss: () -> Unit
) {
    val uiState by vm.uiState.collectAsState()
    val description by vm.description.collectAsState()
    val context = LocalContext.current
    val focusRequester = remember { FocusRequester() }

    LaunchedEffect(Unit) {
        runCatching { focusRequester.requestFocus() }
    }

    Box(
        Modifier
            .fillMaxSize()
            .background(Black)
    ) {
        Column(Modifier.fillMaxSize()) {
            SearchTopBar(
                onClose = onDismiss,
                canCancel = uiState is FileSearchUiState.Scanning
            ) { vm.cancelSearch() }

            AnimatedVisibility(
                visible = uiState !is FileSearchUiState.Scanning,
                enter = fadeIn() + expandVertically(),
                exit = fadeOut() + shrinkVertically()
            ) {
                DescriptionInput(
                    value = description,
                    onValueChange = vm::updateDescription,
                    onSearch = vm::startSearch,
                    focusRequester = focusRequester
                )
            }

            Box(
                Modifier
                    .fillMaxWidth()
                    .weight(1f)
            ) {
                when (val s = uiState) {
                    is FileSearchUiState.Idle -> SearchHintPlaceholder()
                    is FileSearchUiState.Scanning -> ScanningIndicator(state = s)
                    is FileSearchUiState.Results -> {
                        if (s.matches.isEmpty()) {
                            NoResultsView(query = s.query, totalScanned = s.totalScanned)
                        } else {
                            ResultsList(
                                matches = s.matches,
                                onOpen = { vm.openFile(it, context) },
                                onLoadToChat = { onFileLoadedToChat(vm.loadIntoChat(it)) }
                            )
                        }
                    }
                    is FileSearchUiState.Error -> ErrorView(message = s.message) { vm.reset() }
                }
            }

            AnimatedVisibility(
                visible = uiState is FileSearchUiState.Idle || uiState is FileSearchUiState.Results || uiState is FileSearchUiState.Error,
                enter = slideInVertically(initialOffsetY = { it }),
                exit = slideOutVertically(targetOffsetY = { it })
            ) {
                SearchButton(
                    enabled = description.isNotBlank(),
                    onClick = vm::startSearch
                )
            }
        }
    }
}

@Composable
private fun SearchTopBar(
    onClose: () -> Unit,
    canCancel: Boolean,
    onCancel: () -> Unit
) {
    Row(
        Modifier
            .fillMaxWidth()
            .background(Black)
            .padding(horizontal = 8.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        IconButton(onClick = onClose) {
            Icon(Icons.Filled.Close, "Close", tint = TextPrimary)
        }
        Text(
            text = "Find File in Local",
            style = MaterialTheme.typography.titleMedium.copy(
                color = TextPrimary,
                fontWeight = FontWeight.SemiBold,
                letterSpacing = 0.3.sp
            ),
            modifier = Modifier.weight(1f).padding(start = 4.dp)
        )
        if (canCancel) {
            TextButton(onClick = onCancel) {
                Text("Cancel", color = Accent)
            }
        }
    }
    HorizontalDivider(color = DividerCol, thickness = 0.5.dp)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun DescriptionInput(
    value: String,
    onValueChange: (String) -> Unit,
    onSearch: () -> Unit,
    focusRequester: FocusRequester
) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(Surface1)
            .padding(16.dp)
    ) {
        Text(
            "Describe the file you're looking for",
            fontSize = 11.sp,
            color = TextSecond,
            letterSpacing = 0.8.sp,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            modifier = Modifier
                .fillMaxWidth()
                .focusRequester(focusRequester),
            placeholder = {
                Text(
                    "e.g. \"the invoice PDF from last week\" or \"photo of mountains\"",
                    color = TextSecond,
                    fontSize = 14.sp
                )
            },
            minLines = 2,
            maxLines = 4,
            colors = OutlinedTextFieldDefaults.colors(
                focusedContainerColor   = Surface2,
                unfocusedContainerColor = Surface2,
                focusedBorderColor      = Accent,
                unfocusedBorderColor    = Surface3,
                focusedTextColor        = TextPrimary,
                unfocusedTextColor      = TextPrimary,
                cursorColor             = Accent
            ),
            shape = RoundedCornerShape(12.dp),
            textStyle = MaterialTheme.typography.bodyMedium.copy(color = TextPrimary)
        )

        Row(
            Modifier
                .fillMaxWidth()
                .padding(top = 10.dp)
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            listOf("Photo", "PDF", "Word doc", "Excel", "Recent", "Large file").forEach { hint ->
                SuggestionChip(
                    onClick = {
                        val updated = if (value.isBlank()) hint else "$value $hint"
                        onValueChange(updated)
                    },
                    label = { Text(hint, fontSize = 12.sp) },
                    colors = SuggestionChipDefaults.suggestionChipColors(
                        containerColor = Surface3,
                        labelColor = TextSecond
                    ),
                    border = BorderStroke(0.5.dp, Surface3)
                )
            }
        }
    }
    HorizontalDivider(color = DividerCol, thickness = 0.5.dp)
}

@Composable
private fun ScanningIndicator(state: FileSearchUiState.Scanning) {
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0.4f, targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(900, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "alpha"
    )

    Column(
        Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            Icons.AutoMirrored.Outlined.ManageSearch,
            contentDescription = null,
            modifier = Modifier.size(56.dp).alpha(alpha),
            tint = Accent
        )

        Spacer(Modifier.height(24.dp))

        Text(
            state.phase,
            color = TextPrimary,
            fontSize = 15.sp,
            fontWeight = FontWeight.Medium
        )

        if (state.total > 0) {
            Spacer(Modifier.height(16.dp))
            LinearProgressIndicator(
                progress = { state.scanned.toFloat() / state.total.toFloat() },
                modifier = Modifier
                    .fillMaxWidth(0.7f)
                    .height(3.dp)
                    .clip(RoundedCornerShape(2.dp)),
                color = Accent,
                trackColor = Surface3
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "${state.scanned} / ${state.total} files",
                color = TextSecond,
                fontSize = 12.sp
            )
        } else {
            Spacer(Modifier.height(16.dp))
            LinearProgressIndicator(
                modifier = Modifier
                    .fillMaxWidth(0.7f)
                    .height(3.dp)
                    .clip(RoundedCornerShape(2.dp)),
                color = Accent,
                trackColor = Surface3
            )
        }

        Spacer(Modifier.height(32.dp))
        Text(
            "100% offline • your files never leave this device",
            color = TextSecond,
            fontSize = 11.sp,
            letterSpacing = 0.4.sp
        )
    }
}

@Composable
private fun SearchHintPlaceholder() {
    Column(
        Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            Icons.Outlined.FolderOpen,
            contentDescription = null,
            modifier = Modifier.size(64.dp),
            tint = Surface3
        )
        Spacer(Modifier.height(16.dp))
        Text(
            "Describe any file on your device",
            color = TextSecond,
            fontSize = 15.sp,
            fontWeight = FontWeight.Medium
        )
        Spacer(Modifier.height(8.dp))
        Text(
            "AI reads filenames, content, images & docs\nto find exactly what you're thinking of.",
            color = Color(0xFF555555),
            fontSize = 13.sp,
            textAlign = androidx.compose.ui.text.style.TextAlign.Center,
            lineHeight = 20.sp
        )
        Spacer(Modifier.height(32.dp))
        SupportedTypesRow()
    }
}

@Composable
private fun SupportedTypesRow() {
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically
    ) {
        listOf(
            Triple(Icons.Outlined.Image, "Images", Color(0xFF7EB8FF)),
            Triple(Icons.Outlined.PictureAsPdf, "PDFs", Color(0xFFFF7070)),
            Triple(Icons.Outlined.Description, "Docs", Color(0xFF7ED4A0)),
            Triple(Icons.Outlined.TableChart, "Excel", Color(0xFF90EE90)),
            Triple(Icons.AutoMirrored.Outlined.TextSnippet, "Text", Color(0xFFFFD580))
        ).forEach { (icon, label, tint) ->
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.padding(horizontal = 10.dp)
            ) {
                Icon(icon, null, tint = tint, modifier = Modifier.size(22.dp))
                Spacer(Modifier.height(4.dp))
                Text(label, color = Color(0xFF555555), fontSize = 10.sp)
            }
        }
    }
}

@Composable
private fun ResultsList(
    matches: List<FileMatch>,
    onOpen: (FileMatch) -> Unit,
    onLoadToChat: (FileMatch) -> Unit
) {
    LazyColumn(
        Modifier.fillMaxSize(),
        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        item {
            Text(
                "${matches.size} result${if (matches.size == 1) "" else "s"} found",
                color = TextSecond,
                fontSize = 12.sp,
                modifier = Modifier.padding(start = 4.dp, top = 4.dp, bottom = 4.dp)
            )
        }
        items(matches, key = { it.file.absolutePath }) { match ->
            FileResultCard(
                match = match,
                onOpen = { onOpen(match) },
                onLoadToChat = { onLoadToChat(match) }
            )
        }
    }
}

@Composable
private fun FileResultCard(
    match: FileMatch,
    onOpen: () -> Unit,
    onLoadToChat: () -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    val dateStr = remember(match.file.lastModified()) {
        SimpleDateFormat("MMM d, yyyy", Locale.getDefault()).format(Date(match.file.lastModified()))
    }
    val confidencePct = (match.score * 100).toInt()

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { expanded = !expanded },
        colors = CardDefaults.cardColors(containerColor = Surface1),
        shape = RoundedCornerShape(14.dp),
        border = BorderStroke(0.5.dp, if (match.score > 0.7f) AccentDim else Surface3)
    ) {
        Column(Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                FileMimeIcon(match.mimeCategory)
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(
                        match.file.name,
                        color = TextPrimary,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.SemiBold,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    Text(
                        match.file.parentFile?.name ?: "Internal storage",
                        color = TextSecond,
                        fontSize = 11.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
                Spacer(Modifier.width(8.dp))
                ConfidenceBadge(pct = confidencePct)
            }

            Spacer(Modifier.height(8.dp))

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                MetaChip(Icons.Outlined.DateRange, dateStr)
                MetaChip(Icons.Outlined.Storage, match.file.length().toHumanSize())
                MetaChip(Icons.AutoMirrored.Outlined.Label, match.mimeCategory)
            }

            AnimatedVisibility(visible = expanded) {
                Column {
                    Spacer(Modifier.height(10.dp))
                    HorizontalDivider(color = DividerCol)
                    Spacer(Modifier.height(10.dp))
                    Text(
                        "Why matched: ${match.matchReason}",
                        color = Accent,
                        fontSize = 11.sp,
                        modifier = Modifier.padding(bottom = 6.dp)
                    )
                    if (match.extractedText.isNotBlank()) {
                        Text(
                            match.extractedText.take(300),
                            color = TextSecond,
                            fontSize = 12.sp,
                            lineHeight = 18.sp,
                            maxLines = 8,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    Spacer(Modifier.height(12.dp))
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        FilledTonalButton(
                            onClick = onLoadToChat,
                            modifier = Modifier.weight(1f),
                            shape = RoundedCornerShape(10.dp),
                            colors = ButtonDefaults.filledTonalButtonColors(
                                containerColor = AccentDim,
                                contentColor = Accent
                            )
                        ) {
                            Icon(Icons.AutoMirrored.Outlined.Chat, null, modifier = Modifier.size(16.dp))
                            Spacer(Modifier.width(6.dp))
                            Text("Load in Chat", fontSize = 13.sp)
                        }

                        OutlinedButton(
                            onClick = onOpen,
                            modifier = Modifier.weight(1f),
                            shape = RoundedCornerShape(10.dp),
                            border = BorderStroke(0.5.dp, Surface3)
                        ) {
                            Icon(Icons.AutoMirrored.Outlined.OpenInNew, null, modifier = Modifier.size(16.dp), tint = TextSecond)
                            Spacer(Modifier.width(6.dp))
                            Text("Open", fontSize = 13.sp, color = TextSecond)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun FileMimeIcon(category: String) {
    val (icon, tint) = when (category) {
        "image"        -> Pair(Icons.Outlined.Image,         Color(0xFF7EB8FF))
        "pdf"          -> Pair(Icons.Outlined.PictureAsPdf,  Color(0xFFFF7070))
        "document"     -> Pair(Icons.Outlined.Description,   Color(0xFF7ED4A0))
        "spreadsheet"  -> Pair(Icons.Outlined.TableChart,    Color(0xFF90EE90))
        "text"         -> Pair(Icons.AutoMirrored.Outlined.TextSnippet,   Color(0xFFFFD580))
        else           -> Pair(Icons.AutoMirrored.Outlined.InsertDriveFile, TextSecond)
    }
    Box(
        Modifier
            .size(40.dp)
            .clip(RoundedCornerShape(10.dp))
            .background(tint.copy(alpha = 0.12f)),
        contentAlignment = Alignment.Center
    ) {
        Icon(icon, null, tint = tint, modifier = Modifier.size(22.dp))
    }
}

@Composable
private fun ConfidenceBadge(pct: Int) {
    val bg = when {
        pct >= 80 -> Success.copy(alpha = 0.18f)
        pct >= 50 -> Accent.copy(alpha = 0.15f)
        else      -> Surface3
    }
    val textColor = when {
        pct >= 80 -> Success
        pct >= 50 -> Accent
        else      -> TextSecond
    }
    Box(
        Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(bg)
            .padding(horizontal = 8.dp, vertical = 3.dp)
    ) {
        Text("$pct%", color = textColor, fontSize = 12.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun MetaChip(icon: ImageVector, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(icon, null, tint = TextSecond, modifier = Modifier.size(13.dp))
        Spacer(Modifier.width(3.dp))
        Text(label, color = TextSecond, fontSize = 11.sp)
    }
}

@Composable
private fun NoResultsView(query: String, totalScanned: Int) {
    Column(
        Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(Icons.Outlined.SearchOff, null, tint = Surface3, modifier = Modifier.size(56.dp))
        Spacer(Modifier.height(16.dp))
        Text("No files found", color = TextPrimary, fontSize = 16.sp, fontWeight = FontWeight.Medium)
        Spacer(Modifier.height(8.dp))
        Text(
            "Searched $totalScanned files but nothing matched\n\"$query\"",
            color = TextSecond,
            fontSize = 13.sp,
            textAlign = androidx.compose.ui.text.style.TextAlign.Center,
            lineHeight = 20.sp
        )
        Spacer(Modifier.height(16.dp))
        Text(
            "Try different keywords or check storage permissions.",
            color = Color(0xFF444444),
            fontSize = 12.sp,
            textAlign = androidx.compose.ui.text.style.TextAlign.Center
        )
    }
}

@Composable
private fun ErrorView(message: String, onRetry: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(Icons.Outlined.ErrorOutline, null, tint = Color(0xFFFF6B6B), modifier = Modifier.size(48.dp))
        Spacer(Modifier.height(12.dp))
        Text("Search failed", color = TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.Medium)
        Spacer(Modifier.height(8.dp))
        Text(message, color = TextSecond, fontSize = 12.sp, textAlign = androidx.compose.ui.text.style.TextAlign.Center)
        Spacer(Modifier.height(16.dp))
        TextButton(onClick = onRetry) { Text("Reset", color = Accent) }
    }
}

@Composable
private fun SearchButton(enabled: Boolean, onClick: () -> Unit) {
    Box(
        Modifier
            .fillMaxWidth()
            .background(
                Brush.verticalGradient(listOf(Color.Transparent, Black))
            )
            .padding(16.dp)
    ) {
        Button(
            onClick = onClick,
            enabled = enabled,
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp),
            shape = RoundedCornerShape(14.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Accent,
                contentColor = Black,
                disabledContainerColor = Surface3,
                disabledContentColor = TextSecond
            )
        ) {
            Icon(Icons.Filled.Search, null, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(10.dp))
            Text(
                "Search My Device",
                fontWeight = FontWeight.Bold,
                fontSize = 15.sp
            )
        }
    }
}

private fun Long.toHumanSize(): String = when {
    this < 1_024         -> "$this B"
    this < 1_048_576     -> "${this / 1_024} KB"
    this < 1_073_741_824 -> "${this / 1_048_576} MB"
    else                 -> "${this / 1_073_741_824} GB"
}
