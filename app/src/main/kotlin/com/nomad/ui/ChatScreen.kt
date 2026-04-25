package com.nomad.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.net.Uri
import android.speech.RecognizerIntent
import android.util.Base64
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Article
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.nomad.data.*
import com.nomad.viewmodel.ChatUiState
import com.nomad.viewmodel.ChatViewModel
import com.nomad.viewmodel.PerformanceMode
import com.nomad.viewmodel.ResponseMode
import kotlinx.coroutines.launch
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    val state     by viewModel.chatState.collectAsState()
    val listState = rememberLazyListState()
    val context = LocalContext.current
    var inputText by remember { mutableStateOf("") }
    var showSettings by remember { mutableStateOf(false) }
    
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    val filePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? -> uri?.let { viewModel.attachFile(it) } }

    val voiceRecognizerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val spokenText = result.data
            ?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            ?.firstOrNull()
            ?.trim()
            .orEmpty()

        if (spokenText.isNotBlank()) {
            inputText = spokenText
            viewModel.sendMessage(spokenText)
            inputText = ""
        } else {
            viewModel.addSysMsg("Voice input did not return any text. Please try again.")
        }
    }

    val voicePermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            val voiceIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
                putExtra(RecognizerIntent.EXTRA_PROMPT, "Speak to Nomad")
            }

            if (voiceIntent.resolveActivity(context.packageManager) != null) {
                voiceRecognizerLauncher.launch(voiceIntent)
            } else {
                viewModel.addSysMsg("Voice input is not available on this device.")
            }
        } else {
            viewModel.addSysMsg("Microphone permission was denied, so voice input stays off.")
        }
    }

    LaunchedEffect(state.messages.size) {
        if (state.messages.isNotEmpty())
            listState.animateScrollToItem(state.messages.lastIndex)
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet(
                modifier = Modifier.width(300.dp)
            ) {
                ChatHistoryDrawer(
                    state = state,
                    onSelectChat = {
                        viewModel.loadChatSession(it)
                        scope.launch { drawerState.close() }
                    },
                    onDeleteChat = { viewModel.deleteChatSession(it) },
                    onNewChat = {
                        viewModel.clearChat()
                        scope.launch { drawerState.close() }
                    }
                )
            }
        }
    ) {
        Scaffold(
            topBar = { 
                ChatTopBar(
                    state = state, 
                    onMenu = { scope.launch { drawerState.open() } },
                    onSettings = { showSettings = true }
                ) 
            },
            containerColor = Color.Black,
            bottomBar = {
                Surface(
                    color = Color.Black,
                    tonalElevation = 0.dp,
                    shadowElevation = 0.dp,
                    modifier = Modifier.navigationBarsPadding()
                ) {
                    Column {
                        val pending = state.pendingAttachment
                        if (pending != null) {
                            AttachmentChip(
                                attachment = pending,
                                onRemove   = { viewModel.clearAttachment() }
                            )
                        }
                        if (state.isProcessingFile) {
                            LinearProgressIndicator(
                                modifier = Modifier.fillMaxWidth().height(2.dp),
                                color = Color.White,
                                trackColor = Color.Transparent
                            )
                        }
                        ChatInputBar(
                            text         = inputText,
                            onTextChange = { inputText = it },
                            isGenerating = state.isGenerating,
                            hasAttachment = state.pendingAttachment != null,
                            onSend       = {
                                viewModel.sendMessage(inputText)
                                inputText = ""
                            },
                            onStop       = { viewModel.stopGeneration() },
                            onAttach     = { filePicker.launch("*/*") },
                            onLocalSearch = {
                                if (inputText.isBlank()) {
                                    inputText = "find "
                                } else if (!inputText.lowercase().startsWith("find")) {
                                    inputText = "find $inputText"
                                }
                            },
                            onWebSearch = {
                                if (inputText.isBlank()) {
                                    inputText = "search "
                                } else if (!inputText.lowercase().startsWith("search")) {
                                    inputText = "search $inputText"
                                }
                            },
                            webSearchEnabled = state.webSearchEnabled,
                            onVoiceInput = {
                                if (ContextCompat.checkSelfPermission(
                                        context,
                                        Manifest.permission.RECORD_AUDIO
                                    ) == PackageManager.PERMISSION_GRANTED
                                ) {
                                    val voiceIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                                        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                                        putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
                                        putExtra(RecognizerIntent.EXTRA_PROMPT, "Speak to Nomad")
                                    }

                                    if (voiceIntent.resolveActivity(context.packageManager) != null) {
                                        voiceRecognizerLauncher.launch(voiceIntent)
                                    } else {
                                        viewModel.addSysMsg("Voice input is not available on this device.")
                                    }
                                } else {
                                    voicePermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                                }
                            }
                        )
                    }
                }
            }
        ) { padding ->
            Box(
                Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .background(Color.Black)
            ) {
                if (state.messages.isEmpty()) {
                    WelcomeContent(state)
                } else {
                    LazyColumn(
                        state = listState,
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(start = 12.dp, end = 12.dp, top = 12.dp, bottom = 20.dp)
                    ) {
                        items(state.messages, key = { it.id }) { msg ->
                            MessageBubble(
                                msg = msg,
                                state = state,
                                onEdit = { id, text -> viewModel.editMessage(id, text) }
                            )
                        }
                    }
                }

                if (state.isFirstLaunch) {
                    val modelsState by viewModel.modelsState.collectAsState()
                    FirstLaunchModelSelection(
                        models = modelsState.models,
                        downloadStates = modelsState.downloadStates,
                        onSelect = { viewModel.downloadModel(it) },
                        onCancel = { viewModel.cancelDownload(it.id) },
                        onDelete = { viewModel.deleteModel(it) },
                        onDismiss = { viewModel.setFirstLaunchSeen() },
                        totalRamGb = modelsState.totalDeviceRamGb,
                        physicalRamGb = modelsState.physicalRamGb,
                        availableStorageGb = modelsState.availableStorageGb,
                        recommendedModelId = modelsState.recommendedModelId,
                        isModelDownloaded = { viewModel.modelManager.getLocalFile(it).exists() }
                    )
                }
                
                if (state.loadingModel) LoadingOverlay(state.loadProgress, state.loadStatus)

                if (showSettings) {
                    val modelsState by viewModel.modelsState.collectAsState()
                    SettingsSheet(
                        state = state,
                        onDismiss = { showSettings = false },
                        onUpdateName = { viewModel.updateUserName(it) },
                        onUpdateSystemPrompt = { viewModel.updateSystemPrompt(it) },
                        onUpdateUseGpu = { viewModel.updateUseGpu(it) },
                        onUpdateSaveChat = { viewModel.updateSaveChat(it) },
                        onUpdateTemp = { viewModel.updateTemperature(it) },
                        onUpdateMaxTokens = { viewModel.updateMaxTokens(it) },
                        onUpdatePerfMode = { viewModel.updatePerformanceMode(it) },
                        onUpdateResponseMode = { viewModel.updateResponseMode(it) },
                        onUpdateLocalFileHelper = { viewModel.updateLocalFileHelper(it) },
                        onUpdateWebSearch = { viewModel.updateWebSearchEnabled(it) },
                        onDismissRationale = { viewModel.dismissPermissionRationale() },
                        onUpdateContextLength = { viewModel.updateContextLength(it) },
                        onUpdateLanguage = { viewModel.setLanguage(it) },
                        onSelectModel = { viewModel.loadModel(it) },
                        onDownloadMore = { 
                            viewModel.resetFirstLaunch()
                            showSettings = false
                        },
                        isModelDownloaded = { viewModel.modelManager.getLocalFile(it).exists() },
                        models = modelsState.models,
                        recommendedModelId = modelsState.recommendedModelId,
                        memoryBudgetGb = modelsState.totalDeviceRamGb,
                        physicalRamGb = modelsState.physicalRamGb
                    )
                }

                state.error?.let { err ->
                    Snackbar(
                        modifier = Modifier.align(Alignment.BottomCenter).padding(16.dp),
                        containerColor = Color(0xFF1A1A1A),
                        contentColor = Color.White,
                        shape = RoundedCornerShape(0.dp)
                    ) { Text(err) }
                }
            }
        }
    }
}

@Composable
private fun WelcomeContent(state: ChatUiState) {
    Column(
        Modifier.fillMaxSize().padding(horizontal = 24.dp, vertical = 32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        NomadIcon(modifier = Modifier.size(64.dp), color = Color.White)
        
        Spacer(Modifier.height(32.dp))

        Text(
            "Hello, ${state.userName}",
            style = MaterialTheme.typography.displaySmall.copy(fontWeight = FontWeight.Light),
            color = Color.White,
            textAlign = TextAlign.Center
        )
        
        Spacer(Modifier.height(12.dp))
        
        Text(
            "How can I help you today?",
            style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Light),
            color = Color.White.copy(alpha = 0.7f),
            textAlign = TextAlign.Center
        )

        Spacer(Modifier.height(48.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterHorizontally)
        ) {
            StatusPill(
                label = state.loadedModel?.name ?: "No model loaded",
                accent = Color.White
            )
            StatusPill(
                label = if (state.localFileHelperEnabled) "Local Insight ON" else "Local Insight OFF",
                accent = Color.White
            )
        }
    }
}

@Composable
private fun ChatHistoryDrawer(
    state: ChatUiState,
    onSelectChat: (ChatSession) -> Unit,
    onDeleteChat: (String) -> Unit,
    onNewChat: () -> Unit
) {
    Column(
        Modifier.fillMaxSize().background(Color.Black).padding(24.dp)
    ) {
        Row(
            Modifier.fillMaxWidth().padding(bottom = 32.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text("NOMAD", 
                style = MaterialTheme.typography.headlineMedium, 
                fontWeight = FontWeight.Black,
                letterSpacing = 2.sp,
                color = Color.White
            )
            IconButton(
                onClick = onNewChat,
                modifier = Modifier
                    .border(1.dp, Color.White)
                    .size(40.dp)
            ) {
                Icon(Icons.Default.Add, "New Chat", tint = Color.White)
            }
        }
        
        Text("CHRONOLOGY", 
            style = MaterialTheme.typography.labelLarge, 
            color = Color.DarkGray,
            letterSpacing = 1.sp,
            fontWeight = FontWeight.Bold
        )
        
        Spacer(Modifier.height(16.dp))
        
        LazyColumn(Modifier.weight(1f)) {
            items(state.chatHistory) { session ->
                val isSelected = state.currentSessionId == session.id
                Surface(
                    onClick = { onSelectChat(session) },
                    shape = RoundedCornerShape(0.dp),
                    color = if (isSelected) Color(0xFF151515) else Color.Transparent,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                        .border(
                            1.dp,
                            if (isSelected) Color.White else Color.Transparent
                        )
                ) {
                    Row(
                        Modifier.padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                    if (isSelected) {
                        Box(
                            Modifier.size(8.dp).clip(CircleShape).background(Color.White)
                        )
                    }
                    Spacer(Modifier.width(16.dp))
                        Text(
                            session.title.uppercase(),
                            modifier = Modifier.weight(1f),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                            color = if (isSelected) Color.White else Color.Gray,
                            letterSpacing = 0.5.sp
                        )
                        IconButton(
                            onClick = { onDeleteChat(session.id) },
                            modifier = Modifier.size(24.dp)
                        ) {
                            Icon(Icons.Default.Delete, null, tint = Color.DarkGray, modifier = Modifier.size(16.dp))
                        }
                    }
                }
            }
        }
        
        Spacer(Modifier.height(16.dp))
        HorizontalDivider(color = Color.DarkGray)
        Spacer(Modifier.height(16.dp))
        Text("OFFLINE LOCAL AI", style = MaterialTheme.typography.labelSmall, color = Color.Gray, fontWeight = FontWeight.Bold)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChatTopBar(
    state: ChatUiState, 
    onMenu: () -> Unit, 
    onSettings: () -> Unit
) {
    Surface(
        color = Color.Black,
        shadowElevation = 0.dp,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = onMenu) {
                Icon(Icons.Default.Menu, "Menu", tint = Color.White)
            }

            Spacer(Modifier.width(8.dp))

            Text(
                "Nomad",
                style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Normal),
                color = Color.White,
                modifier = Modifier.weight(1f)
            )

            if (state.tokensPerSec > 0f) {
                Text(
                    "${"%.1f".format(state.tokensPerSec)} t/s",
                    style = MaterialTheme.typography.labelMedium,
                    color = Color.White.copy(alpha = 0.6f),
                    modifier = Modifier.padding(end = 12.dp)
                )
            }

            IconButton(onClick = onSettings) {
                Icon(Icons.Default.Settings, "Settings", tint = Color.White)
            }
        }
    }
}

@Composable
private fun StatusPill(label: String, accent: Color) {
    Surface(
        color = Color.Black,
        shape = RoundedCornerShape(999.dp),
        border = BorderStroke(1.dp, Color.White.copy(alpha = 0.2f))
    ) {
        Text(
            text = label,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            style = MaterialTheme.typography.labelMedium,
            color = Color.White
        )
    }
}

@Composable
private fun ResponseModeStrip(
    selectedMode: ResponseMode,
    onSelect: (ResponseMode) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 14.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        ResponseMode.values().forEach { mode ->
            val isSelected = selectedMode == mode
            Surface(
                onClick = { onSelect(mode) },
                modifier = Modifier.weight(1f),
                color = if (isSelected) Color(0xFF24A27B) else Color(0xFF11151B),
                shape = RoundedCornerShape(18.dp),
                border = BorderStroke(1.dp, if (isSelected) Color(0xFF3FD3A1) else Color(0xFF242C38))
            ) {
                Column(
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = mode.name,
                        style = MaterialTheme.typography.labelMedium,
                        color = Color.White,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = when (mode) {
                            ResponseMode.FAST -> "Quick"
                            ResponseMode.BALANCED -> "Normal"
                            ResponseMode.THINKING -> "Deep"
                        },
                        style = MaterialTheme.typography.labelSmall,
                        color = Color(0xFFCFD7E1)
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SettingsSheet(
    state: ChatUiState,
    onDismiss: () -> Unit,
    onUpdateName: (String) -> Unit,
    onUpdateSystemPrompt: (String) -> Unit,
    onUpdateUseGpu: (Boolean) -> Unit,
    onUpdateSaveChat: (Boolean) -> Unit,
    onUpdateTemp: (Float) -> Unit,
    onUpdateMaxTokens: (Int) -> Unit,
    onUpdatePerfMode: (PerformanceMode) -> Unit,
    onUpdateResponseMode: (ResponseMode) -> Unit,
    onUpdateLocalFileHelper: (Boolean) -> Unit,
    onUpdateWebSearch: (Boolean) -> Unit,
    onDismissRationale: () -> Unit,
    onUpdateContextLength: (Int) -> Unit,
    onUpdateLanguage: (String) -> Unit,
    onSelectModel: (ModelInfo) -> Unit,
    onDownloadMore: () -> Unit,
    isModelDownloaded: (ModelInfo) -> Boolean,
    models: List<ModelInfo>,
    recommendedModelId: String?,
    memoryBudgetGb: Double,
    physicalRamGb: Double
) {
    var name by remember { mutableStateOf(state.userName) }
    var systemPrompt by remember { mutableStateOf(state.systemPrompt) }
    var showLangMenu by remember { mutableStateOf(false) }
    var showModelMenu by remember { mutableStateOf(false) }
    var showResponseMenu by remember { mutableStateOf(false) }

    val languages = listOf(
        "English", "Hindi", "Marathi", "Gujarati", "Tamil", "Bengali",
        "Telugu", "Kannada", "Malayalam", "Punjabi", "Odia", "Urdu",
        "Assamese", "Sanskrit", "Konkani", "Sindhi", "Dogri", "Maithili", "Santhali",
        "Kashmiri", "Nepali", "Manipuri", "Bodo"
    )

    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = Color.Black,
        contentColor = Color.White,
        dragHandle = null
    ) {
        Column(
            Modifier
                .fillMaxSize()
                .padding(24.dp)
                .statusBarsPadding()
                .navigationBarsPadding()
                .verticalScroll(rememberScrollState())
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("SETTINGS", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Black)
                Spacer(Modifier.weight(1f))
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Default.Close, null, tint = Color.White)
                }
            }
            
            Spacer(Modifier.height(32.dp))

            // --- RESPONSE MODE ---
            Text("Response Style", style = MaterialTheme.typography.labelLarge, color = Color.White)
            Spacer(Modifier.height(12.dp))
            Box {
                OutlinedButton(
                    onClick = { showResponseMenu = true },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    border = BorderStroke(1.dp, Color.White.copy(alpha = 0.3f)),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.White)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(state.responseMode.name, fontWeight = FontWeight.Bold)
                        Spacer(Modifier.weight(1f))
                        Icon(Icons.Default.ArrowDropDown, null)
                    }
                }
                DropdownMenu(
                    expanded = showResponseMenu,
                    onDismissRequest = { showResponseMenu = false },
                    modifier = Modifier.background(Color.Black).border(1.dp, Color.White.copy(alpha = 0.2f)).fillMaxWidth(0.85f)
                ) {
                    ResponseMode.values().forEach { mode ->
                        DropdownMenuItem(
                            text = { 
                                Column {
                                    Text(mode.name, color = Color.White, fontWeight = FontWeight.Bold)
                                    Text(
                                        when(mode) {
                                            ResponseMode.FAST -> "Shortest and fastest replies."
                                            ResponseMode.BALANCED -> "Balanced speed and detail."
                                            ResponseMode.THINKING -> "Deep thinking for complex tasks."
                                        },
                                        style = MaterialTheme.typography.labelSmall,
                                        color = Color.White.copy(alpha = 0.6f)
                                    )
                                }
                            },
                            onClick = {
                                onUpdateResponseMode(mode)
                                showResponseMenu = false
                            }
                        )
                    }
                }
            }
            Spacer(Modifier.height(24.dp))

            // --- MODEL SELECTION ---
            Text("Active Model", style = MaterialTheme.typography.labelLarge, color = Color.White)
            Spacer(Modifier.height(12.dp))
            recommendedModelId?.let { bestId ->
                models.firstOrNull { it.id == bestId }?.let { bestModel ->
                    Surface(
                        color = Color.Black,
                        shape = RoundedCornerShape(14.dp),
                        border = BorderStroke(1.dp, Color.White),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(Modifier.padding(14.dp)) {
                            Text("BEST MATCH FOR THIS DEVICE", color = Color.White, fontWeight = FontWeight.Black, fontSize = 10.sp)
                            Spacer(Modifier.height(4.dp))
                            Text(
                                "${bestModel.name} (${bestModel.paramCount})",
                                color = Color.White,
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }
                    Spacer(Modifier.height(12.dp))
                }
            }
            Box {
                OutlinedButton(
                    onClick = { showModelMenu = true },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    border = BorderStroke(1.dp, Color(0xFF333333)),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.White)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.ModelTraining, null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(12.dp))
                        Text(state.loadedModel?.name?.uppercase() ?: "SELECT MODEL", fontWeight = FontWeight.Bold)
                        Spacer(Modifier.weight(1f))
                        Icon(Icons.Default.ArrowDropDown, null)
                    }
                }
                DropdownMenu(
                    expanded = showModelMenu,
                    onDismissRequest = { showModelMenu = false },
                    modifier = Modifier.background(Color(0xFF1A1A1A)).fillMaxWidth(0.8f)
                ) {
                    val downloadedModels = models.filter { isModelDownloaded(it) }
                    
                    if (downloadedModels.isEmpty()) {
                        DropdownMenuItem(
                            text = { Text("No models downloaded", color = Color.Gray) },
                            onClick = { }
                        )
                    }

                    downloadedModels.forEach { model ->
                        DropdownMenuItem(
                            text = { 
                                Column {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Text(model.name, color = Color.White)
                                        if (model.id == recommendedModelId) {
                                            Spacer(Modifier.width(8.dp))
                                            Surface(
                                                color = Color(0xFF173525),
                                                shape = RoundedCornerShape(999.dp)
                                            ) {
                                                Text(
                                                    "BEST",
                                                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                                                    style = MaterialTheme.typography.labelSmall,
                                                    color = Color(0xFF86E0B5),
                                                    fontWeight = FontWeight.Bold
                                                )
                                            }
                                        }
                                    }
                                    Text("${model.paramCount} • ${model.filename}", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                                }
                            },
                            onClick = {
                                onSelectModel(model)
                                showModelMenu = false
                            }
                        )
                    }
                    
                    HorizontalDivider(color = Color.DarkGray)
                    
                    DropdownMenuItem(
                        text = { 
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Default.CloudDownload, null, tint = Color(0xFF9B72CB), modifier = Modifier.size(16.dp))
                                Spacer(Modifier.width(8.dp))
                                Text("DOWNLOAD MORE MODELS", color = Color(0xFF9B72CB), fontWeight = FontWeight.Bold)
                            }
                        },
                        onClick = {
                            showModelMenu = false
                            onDownloadMore()
                        }
                    )
                }
            }
            Spacer(Modifier.height(32.dp))

            // --- PERFORMANCE SETTINGS ---
            Text("Performance Optimization", style = MaterialTheme.typography.labelLarge, color = Color.White)
            Spacer(Modifier.height(12.dp))

            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PerformanceMode.values().forEach { mode ->
                    val isSelected = state.performanceMode == mode
                    Surface(
                        onClick = { onUpdatePerfMode(mode) },
                        modifier = Modifier.weight(1f),
                        color = if (isSelected) Color.White else Color.Black,
                        shape = RoundedCornerShape(8.dp),
                        border = BorderStroke(1.dp, Color.White.copy(alpha = 0.3f))
                    ) {
                        Column(Modifier.padding(8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(mode.name, 
                                style = MaterialTheme.typography.labelSmall, 
                                color = if (isSelected) Color.Black else Color.White,
                                fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }
            Text(
                text = when(state.performanceMode) {
                    PerformanceMode.HIGH -> "Max performance. Uses more battery."
                    PerformanceMode.BALANCED -> "Optimal balance for daily use."
                    PerformanceMode.POWER_SAVER -> "Slows down to keep device cool."
                },
                style = MaterialTheme.typography.labelSmall,
                color = Color.White.copy(alpha = 0.4f),
                modifier = Modifier.padding(top = 8.dp, start = 4.dp)
            )

            Spacer(Modifier.height(32.dp))

            // --- USER SETTINGS ---
            Text("User Profile", style = MaterialTheme.typography.labelLarge, color = Color.White)
            Spacer(Modifier.height(12.dp))
            
            Text("Display Name", style = MaterialTheme.typography.labelMedium, color = Color.White.copy(alpha = 0.5f))
            Spacer(Modifier.height(8.dp))
            OutlinedTextField(
                value = name,
                onValueChange = { 
                    name = it
                    onUpdateName(it)
                },
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Color.White,
                    unfocusedBorderColor = Color.White.copy(alpha = 0.2f),
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White
                )
            )

            Spacer(Modifier.height(24.dp))

            // --- AI MODEL SETTINGS ---
            Text("AI Model Settings", style = MaterialTheme.typography.labelLarge, color = Color.White)
            Spacer(Modifier.height(12.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("GPU Acceleration", style = MaterialTheme.typography.bodyLarge)
                    Text("Use RAM/GPU for faster inference", style = MaterialTheme.typography.labelSmall, color = Color.White.copy(alpha = 0.5f))
                }
                Switch(
                    checked = state.useGpu,
                    onCheckedChange = onUpdateUseGpu,
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = Color.Black,
                        checkedTrackColor = Color.White,
                        uncheckedThumbColor = Color.White.copy(alpha = 0.5f),
                        uncheckedTrackColor = Color.Black
                    )
                )
            }

            Spacer(Modifier.height(16.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Save Chat History", style = MaterialTheme.typography.bodyLarge)
                    Text("Auto-save conversations to local storage", style = MaterialTheme.typography.labelSmall, color = Color.White.copy(alpha = 0.5f))
                }
                Switch(
                    checked = state.saveChat,
                    onCheckedChange = onUpdateSaveChat,
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = Color.Black,
                        checkedTrackColor = Color.White,
                        uncheckedThumbColor = Color.White.copy(alpha = 0.5f),
                        uncheckedTrackColor = Color.Black
                    )
                )
            }

            Spacer(Modifier.height(16.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Local Insight Helper", style = MaterialTheme.typography.bodyLarge)
                    Text("Allow AI to read your local files.", style = MaterialTheme.typography.labelSmall, color = Color.White.copy(alpha = 0.5f))
                }
                Switch(
                    checked = state.localFileHelperEnabled,
                    onCheckedChange = onUpdateLocalFileHelper,
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = Color.Black,
                        checkedTrackColor = Color.White,
                        uncheckedThumbColor = Color.White.copy(alpha = 0.5f),
                        uncheckedTrackColor = Color.Black
                    )
                )
            }

            Spacer(Modifier.height(16.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Web Search", style = MaterialTheme.typography.bodyLarge)
                    Text("Allow AI to search the web for real-time info.", style = MaterialTheme.typography.labelSmall, color = Color.White.copy(alpha = 0.5f))
                }
                Switch(
                    checked = state.webSearchEnabled,
                    onCheckedChange = onUpdateWebSearch,
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = Color.Black,
                        checkedTrackColor = Color.White,
                        uncheckedThumbColor = Color.White.copy(alpha = 0.5f),
                        uncheckedTrackColor = Color.Black
                    )
                )
            }

            Spacer(Modifier.height(24.dp))

            // --- PERMISSION RATIONALE DIALOG ---
            if (state.showPermissionRationale) {
                AlertDialog(
                    onDismissRequest = onDismissRationale,
                    title = { Text("Local File Access", color = Color.White) },
                    text = { Text("Nomad needs file access only when you use local file search or analyze files on this device.", color = Color.White.copy(alpha = 0.6f)) },
                    confirmButton = {
                        TextButton(onClick = { 
                            onUpdateLocalFileHelper(true) 
                            onDismissRationale()
                        }) {
                            Text("GRANT", color = Color.White)
                        }
                    },
                    dismissButton = {
                        TextButton(onClick = onDismissRationale) {
                            Text("CANCEL", color = Color.White.copy(alpha = 0.4f))
                        }
                    },
                    containerColor = Color.Black,
                    modifier = Modifier.border(1.dp, Color.White.copy(alpha = 0.2f), RoundedCornerShape(28.dp))
                )
            }

            Spacer(Modifier.height(24.dp))

            // --- REGIONAL SETTINGS ---
            Text("Regional Settings", style = MaterialTheme.typography.labelLarge, color = Color.White)
            Spacer(Modifier.height(12.dp))

            Column {
                Text("AI Language Preference", style = MaterialTheme.typography.labelMedium, color = Color.White.copy(alpha = 0.5f))
                Spacer(Modifier.height(8.dp))
                Box {
                    OutlinedButton(
                        onClick = { showLangMenu = true },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp),
                        border = BorderStroke(1.dp, Color.White.copy(alpha = 0.2f)),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.White)
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Language, null, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(12.dp))
                            Text(state.preferredLanguage.uppercase(), fontWeight = FontWeight.Bold)
                            Spacer(Modifier.weight(1f))
                            Icon(Icons.Default.ArrowDropDown, null)
                        }
                    }
                    DropdownMenu(
                        expanded = showLangMenu,
                        onDismissRequest = { showLangMenu = false },
                        modifier = Modifier.background(Color.Black).border(1.dp, Color.White.copy(alpha = 0.2f)).fillMaxWidth(0.85f)
                    ) {
                        languages.forEach { lang ->
                            DropdownMenuItem(
                                text = { Text(lang, color = Color.White) },
                                onClick = {
                                    onUpdateLanguage(lang)
                                    showLangMenu = false
                                }
                            )
                        }
                    }
                }
                Text(
                    "Note: AI will respond in ${state.preferredLanguage}.",
                    style = MaterialTheme.typography.labelSmall,
                    color = Color.White.copy(alpha = 0.4f),
                    modifier = Modifier.padding(top = 4.dp, start = 4.dp)
                )
            }

            Spacer(Modifier.height(24.dp))

            Text("Creativity: ${"%.1f".format(state.temperature)}", style = MaterialTheme.typography.bodyMedium)
            Slider(
                value = state.temperature,
                onValueChange = onUpdateTemp,
                valueRange = 0.1f..1.5f,
                steps = 14,
                colors = SliderDefaults.colors(thumbColor = Color.White, activeTrackColor = Color.White, inactiveTrackColor = Color.White.copy(alpha = 0.2f))
            )

            Spacer(Modifier.height(16.dp))

            Text("Max Response Length: ${state.maxTokens}", style = MaterialTheme.typography.bodyMedium)
            Slider(
                value = state.maxTokens.toFloat(),
                onValueChange = { onUpdateMaxTokens(it.toInt()) },
                valueRange = 64f..8192f,
                steps = 63,
                colors = SliderDefaults.colors(thumbColor = Color.White, activeTrackColor = Color.White, inactiveTrackColor = Color.White.copy(alpha = 0.2f))
            )

            Spacer(Modifier.height(16.dp))

            Text("Context Length: ${state.contextLength}", style = MaterialTheme.typography.bodyMedium)
            Slider(
                value = state.contextLength.toFloat(),
                onValueChange = { onUpdateContextLength(it.toInt()) },
                valueRange = 512f..8192f,
                steps = 15,
                colors = SliderDefaults.colors(thumbColor = Color.White, activeTrackColor = Color.White, inactiveTrackColor = Color.White.copy(alpha = 0.2f))
            )

            Spacer(Modifier.height(24.dp))

            Text("System Instructions", style = MaterialTheme.typography.labelMedium, color = Color.White.copy(alpha = 0.5f))
            Spacer(Modifier.height(8.dp))
            OutlinedTextField(
                value = systemPrompt,
                onValueChange = { 
                    systemPrompt = it
                    onUpdateSystemPrompt(it)
                },
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Color.White,
                    unfocusedBorderColor = Color.White.copy(alpha = 0.2f),
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White
                )
            )

            Spacer(Modifier.height(32.dp))
        }
    }
}

@Composable
private fun AttachmentChip(attachment: Attachment, onRemove: () -> Unit) {
    Surface(
        modifier = Modifier.padding(8.dp),
        color = Color(0xFF171B22),
        shape = RoundedCornerShape(18.dp),
        border = BorderStroke(1.dp, Color(0xFF2A313C))
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                when(attachment.type) {
                    AttachmentType.IMAGE -> Icons.Default.Image
                    AttachmentType.PDF   -> Icons.Default.Description
                    AttachmentType.TEXT  -> Icons.AutoMirrored.Filled.Article
                    else -> Icons.Default.AttachFile
                },
                null, tint = Color.White, modifier = Modifier.size(20.dp)
            )
            Spacer(Modifier.width(8.dp))
            Column(Modifier.weight(1f)) {
                Text(attachment.fileName,
                    style = MaterialTheme.typography.bodySmall, 
                    color = Color(0xFFF5F7FA),
                    maxLines = 1)
                Text(FileProcessor.formatSize(attachment.fileSizeBytes),
                    style = MaterialTheme.typography.labelSmall,
                    color = Color(0xFF8F9BA8))
            }
            IconButton(onClick = onRemove) {
                Icon(Icons.Default.Cancel, "Remove", tint = Color(0xFFB6C0CC))
            }
        }
    }
}

@Composable
private fun MessageBubble(
    msg: ChatMessage,
    state: ChatUiState,
    onEdit: (Long, String) -> Unit
) {
    if (msg.role == Role.SYSTEM) {
        Box(
            Modifier.fillMaxWidth().padding(horizontal = 32.dp, vertical = 8.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(msg.content.uppercase(),
                style     = MaterialTheme.typography.labelSmall,
                modifier  = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                textAlign = TextAlign.Center,
                color     = Color.White.copy(alpha = 0.5f),
                letterSpacing = 1.sp)
        }
        return
    }

    val isUser = msg.role == Role.USER
    val clipboard = LocalClipboardManager.current
    var isEditing by remember { mutableStateOf(false) }
    var editValue by remember { mutableStateOf(msg.content) }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 12.dp),
        horizontalAlignment = if (isUser) Alignment.End else Alignment.Start
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(bottom = 4.dp)
        ) {
            if (!isUser) {
                NomadIcon(modifier = Modifier.size(20.dp), color = Color.White)
                Spacer(Modifier.width(8.dp))
            }
            Text(
                text = if (isUser) "You" else "Nomad",
                style = MaterialTheme.typography.labelMedium,
                color = Color.White.copy(alpha = 0.7f),
                fontWeight = FontWeight.Bold
            )
        }

        Surface(
            shape = RoundedCornerShape(16.dp),
            color = if (isUser) Color.White.copy(alpha = 0.1f) else Color.Transparent,
            modifier = Modifier.widthIn(max = 340.dp)
        ) {
            Column(Modifier.padding(horizontal = if (isUser) 16.dp else 0.dp, vertical = if (isUser) 12.dp else 0.dp)) {
                if (!isUser && msg.searchResults != null) {
                    var expanded by remember { mutableStateOf(false) }
                    Column(
                        Modifier
                            .fillMaxWidth()
                            .padding(bottom = 12.dp)
                            .clickable { expanded = !expanded }
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                Icons.Default.Language,
                                null,
                                tint = Color.White.copy(alpha = 0.5f),
                                modifier = Modifier.size(16.dp)
                            )
                            Spacer(Modifier.width(8.dp))
                            Text(
                                "WEB SEARCH RESULTS",
                                style = MaterialTheme.typography.labelSmall,
                                color = Color.White.copy(alpha = 0.5f),
                                letterSpacing = 1.sp
                            )
                        }
                        if (expanded) {
                            Text(
                                text = msg.searchResults,
                                color = Color.White.copy(alpha = 0.4f),
                                style = MaterialTheme.typography.bodySmall,
                                modifier = Modifier.padding(top = 8.dp)
                                    .background(Color.White.copy(alpha = 0.05f), RoundedCornerShape(8.dp))
                                    .padding(8.dp)
                            )
                        }
                    }
                }

                if (!isUser && msg.thought != null) {
                    var expanded by remember { mutableStateOf(false) }
                    Column(
                        Modifier
                            .fillMaxWidth()
                            .padding(bottom = 12.dp)
                            .clickable { expanded = !expanded }
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                Icons.Default.Psychology,
                                null,
                                tint = Color.White.copy(alpha = 0.5f),
                                modifier = Modifier.size(16.dp)
                            )
                            Spacer(Modifier.width(8.dp))
                            Text(
                                "THOUGHT PROCESS",
                                style = MaterialTheme.typography.labelSmall,
                                color = Color.White.copy(alpha = 0.5f),
                                letterSpacing = 1.sp
                            )
                        }
                        if (expanded) {
                            Text(
                                text = msg.thought,
                                color = Color.White.copy(alpha = 0.4f),
                                style = MaterialTheme.typography.bodySmall,
                                modifier = Modifier.padding(top = 8.dp)
                            )
                        }
                    }
                }

                    if (msg.isStreaming && msg.content == "...") {
                        Box(modifier = Modifier.padding(vertical = 12.dp)) {
                            Column {
                                if (state.isSearchingWeb && msg.role == Role.ASSISTANT) {
                                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(bottom = 8.dp)) {
                                        Icon(Icons.Default.Language, null, tint = Color.White.copy(alpha = 0.5f), modifier = Modifier.size(16.dp))
                                        Spacer(Modifier.width(8.dp))
                                        Text("Searching web...", style = MaterialTheme.typography.labelSmall, color = Color.White.copy(alpha = 0.5f))
                                    }
                                }
                                BouncingDots()
                            }
                        }
                    } else if (!isUser && msg.content.contains("```")) {
                    // Optimized splitting for Gemini-like code block rendering
                    val blocks = msg.content.split("```")
                    blocks.forEachIndexed { index, part ->
                        if (index % 2 == 1) {
                            // This is inside a code block
                            val lines = part.split("\n")
                            val lang = if (lines.firstOrNull()?.contains(Regex("^[a-zA-Z]+$")) == true) lines.first() else ""
                            val code = if (lang.isNotEmpty()) part.substringAfter("\n") else part
                            CodeBlock(code.trim(), lang)
                        } else {
                            // This is normal text
                            if (part.isNotBlank()) {
                                Text(
                                    text = part.trim(),
                                    color = Color.White,
                                    style = MaterialTheme.typography.bodyLarge.copy(lineHeight = 26.sp),
                                    modifier = Modifier.padding(vertical = 4.dp)
                                )
                            }
                        }
                    }
                } else {
                    Text(
                        text = msg.content,
                        color = Color.White,
                        style = MaterialTheme.typography.bodyLarge.copy(lineHeight = 26.sp)
                    )
                }

                if (msg.isStreaming) {
                    // Only show cursor if we've actually started getting content
                    if (msg.content != "...") {
                        TerminalCursor()
                    }
                }
            }
        }

        Row(
            modifier = Modifier.padding(top = 4.dp),
            horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
        ) {
            IconButton(onClick = { clipboard.setText(AnnotatedString(msg.content)) }, modifier = Modifier.size(32.dp)) {
                Icon(Icons.Default.ContentCopy, null, tint = Color.White.copy(alpha = 0.3f), modifier = Modifier.size(16.dp))
            }
            if (isUser) {
                IconButton(onClick = { isEditing = true }, modifier = Modifier.size(32.dp)) {
                    Icon(Icons.Default.Edit, null, tint = Color.White.copy(alpha = 0.3f), modifier = Modifier.size(16.dp))
                }
            }
        }
    }
}

@Composable
private fun CodeBlock(code: String, language: String = "") {
    val clipboardManager = LocalClipboardManager.current
    var copied by remember { mutableStateOf(false) }

    LaunchedEffect(copied) {
        if (copied) {
            kotlinx.coroutines.delay(2000)
            copied = false
        }
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 12.dp)
            .background(Color.White.copy(alpha = 0.05f), RoundedCornerShape(12.dp))
            .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(12.dp))
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                language.uppercase().ifEmpty { "CODE" },
                style = MaterialTheme.typography.labelMedium,
                color = Color.White.copy(alpha = 0.5f),
                fontWeight = FontWeight.Bold
            )
            Row(
                Modifier.clickable {
                    clipboardManager.setText(AnnotatedString(code.trim()))
                    copied = true
                },
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    if (copied) Icons.Default.Check else Icons.Default.ContentCopy,
                    null,
                    tint = if (copied) Color.Green else Color.White.copy(alpha = 0.5f),
                    modifier = Modifier.size(14.dp)
                )
                Spacer(Modifier.width(6.dp))
                Text(
                    if (copied) "Copied" else "Copy",
                    style = MaterialTheme.typography.labelSmall,
                    color = if (copied) Color.Green else Color.White.copy(alpha = 0.5f),
                    fontWeight = FontWeight.Bold
                )
            }
        }
        
        Box(
            modifier = Modifier
                .horizontalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 12.dp)
        ) {
            Text(
                text = code.trim(),
                color = Color.White,
                style = MaterialTheme.typography.bodyMedium.copy(
                    fontFamily = FontFamily.Monospace,
                    lineHeight = 22.sp
                )
            )
        }
    }
}

@Composable
private fun ChatInputBar(
    text: String,
    onTextChange: (String) -> Unit,
    isGenerating: Boolean,
    hasAttachment: Boolean,
    onSend: () -> Unit,
    onStop: () -> Unit,
    onAttach: () -> Unit,
    onLocalSearch: () -> Unit,
    onWebSearch: () -> Unit,
    onVoiceInput: () -> Unit,
    webSearchEnabled: Boolean = false
) {
    var showMenu by remember { mutableStateOf(false) }

    Surface(
        color = Color.Black,
        tonalElevation = 0.dp
    ) {
        Row(
            modifier = Modifier
                .padding(horizontal = 16.dp, vertical = 16.dp)
                .fillMaxWidth()
                .background(Color.White.copy(alpha = 0.05f), RoundedCornerShape(28.dp))
                .border(1.dp, Color.White.copy(alpha = 0.1f), RoundedCornerShape(28.dp))
                .padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box {
                IconButton(onClick = { showMenu = true }) {
                    Icon(Icons.Default.Add, null, tint = Color.White)
                }
                DropdownMenu(
                    expanded = showMenu,
                    onDismissRequest = { showMenu = false },
                    modifier = Modifier.background(Color.Black).border(1.dp, Color.White.copy(alpha = 0.1f))
                ) {
                    DropdownMenuItem(
                        text = { Text("Upload File", color = Color.White) },
                        leadingIcon = { Icon(Icons.Default.AttachFile, null, tint = Color.White) },
                        onClick = {
                            showMenu = false
                            onAttach()
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Find Local", color = Color.White) },
                        leadingIcon = { Icon(Icons.Default.Search, null, tint = Color.White) },
                        onClick = {
                            showMenu = false
                            onLocalSearch()
                        }
                    )
                    if (webSearchEnabled) {
                        DropdownMenuItem(
                            text = { Text("Web Search", color = Color.White) },
                            leadingIcon = { Icon(Icons.Default.Language, null, tint = Color.White) },
                            onClick = {
                                showMenu = false
                                onWebSearch()
                            }
                        )
                    }
                }
            }

            BasicTextField(
                value = text,
                onValueChange = onTextChange,
                enabled = true,
                modifier = Modifier
                    .weight(1f)
                    .padding(horizontal = 8.dp, vertical = 12.dp),
                textStyle = LocalTextStyle.current.copy(
                    color = Color.White,
                    fontSize = 16.sp
                ),
                cursorBrush = SolidColor(Color.White),
                decorationBox = { innerTextField ->
                    Box(contentAlignment = Alignment.CenterStart) {
                        if (text.isEmpty()) {
                            Text(
                                "Ask Nomad anything...",
                                color = Color.White.copy(alpha = 0.4f),
                                fontSize = 16.sp
                            )
                        }
                        innerTextField()
                    }
                }
            )

            if (isGenerating) {
                // Stop button: visible while model is busy. Clicking this stops generation.
                IconButton(
                    onClick = onStop,
                    modifier = Modifier
                        .size(40.dp)
                        .background(Color.White, CircleShape)
                ) {
                    Icon(
                        Icons.Default.Stop,
                        null,
                        tint = Color.Black,
                        modifier = Modifier.size(20.dp)
                    )
                }
            } else if (text.isNotBlank() || hasAttachment) {
                IconButton(
                    onClick = onSend,
                    modifier = Modifier
                        .size(40.dp)
                        .background(Color.White, CircleShape)
                ) {
                    Icon(
                        Icons.Default.ArrowUpward,
                        null,
                        tint = Color.Black,
                        modifier = Modifier.size(20.dp)
                    )
                }
            } else {
                IconButton(onClick = onVoiceInput) {
                    Icon(Icons.Default.Mic, null, tint = Color.White)
                }
            }
        }
    }
}

@Composable
private fun FirstLaunchModelSelection(
    models: List<ModelInfo>,
    downloadStates: Map<String, DownloadState>,
    onSelect: (ModelInfo) -> Unit,
    onCancel: (ModelInfo) -> Unit,
    onDelete: (ModelInfo) -> Unit,
    onDismiss: () -> Unit,
    totalRamGb: Double,
    physicalRamGb: Double,
    availableStorageGb: Double,
    recommendedModelId: String?,
    isModelDownloaded: (ModelInfo) -> Boolean
) {
    Box(
        Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.95f))
            .padding(24.dp)
            .clickable(enabled = false) {}, // Block clicks to background
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            NomadIcon(Modifier.size(48.dp))
            Spacer(Modifier.height(16.dp))
            Text(
                "NOMAD SETUP",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Black,
                color = Color.White,
                letterSpacing = 2.sp
            )
            Text(
                "Pick the best local model for this device",
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray,
                textAlign = TextAlign.Center
            )
            
            Spacer(Modifier.height(32.dp))

            Surface(
                modifier = Modifier.fillMaxWidth(),
                color = Color(0xFF11151B),
                shape = RoundedCornerShape(18.dp),
                border = BorderStroke(1.dp, Color(0xFF242C38))
            ) {
                Column(Modifier.padding(16.dp)) {
                    Text("Device budget", color = Color.White, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.height(6.dp))
                    Text(
                        "Effective memory budget ${"%.1f".format(totalRamGb)} GB • physical RAM ${"%.1f".format(physicalRamGb)} GB • free storage ${"%.1f".format(availableStorageGb)} GB",
                        color = Color(0xFFB6C0CC),
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }

            Spacer(Modifier.height(16.dp))

            models.forEach { model ->
                val isSafe = model.ramRequirementGb <= totalRamGb * 0.9
                val isDownloaded = isModelDownloaded(model)
                val downloadState = downloadStates[model.id]

                Surface(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp),
                    color = Color(0xFF0A0A0A),
                    shape = RoundedCornerShape(0.dp),
                    border = BorderStroke(1.dp, if (isSafe) Color(0xFF222222) else Color(0xFF333333))
                ) {
                    Column(Modifier.padding(16.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Text(model.name.uppercase(), fontWeight = FontWeight.Black, color = if (isSafe) Color.White else Color.Gray)
                                    if (model.id == recommendedModelId) {
                                        Spacer(Modifier.width(8.dp))
                                        Surface(
                                            color = Color(0xFF173525),
                                            shape = RoundedCornerShape(999.dp)
                                        ) {
                                            Text(
                                                "BEST",
                                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                                                style = MaterialTheme.typography.labelSmall,
                                                color = Color(0xFF86E0B5),
                                                fontWeight = FontWeight.Bold
                                            )
                                        }
                                    }
                                }
                                Text(
                                    "${model.paramCount} • ${model.ramRequirementGb} GB RAM target • ${model.sizeLabel}",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color.Gray
                                )
                            }
                            
                            if (isDownloaded) {
                                Text("READY", color = Color.White, style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold)
                                Spacer(Modifier.width(8.dp))
                                IconButton(onClick = { onDelete(model) }, modifier = Modifier.size(24.dp)) {
                                    Icon(Icons.Default.Delete, null, tint = Color.DarkGray, modifier = Modifier.size(16.dp))
                                }
                            } else if (downloadState is DownloadState.Progress) {
                                Text("${downloadState.percent}%", color = Color.White, style = MaterialTheme.typography.labelSmall)
                                Spacer(Modifier.width(8.dp))
                                IconButton(onClick = { onCancel(model) }, modifier = Modifier.size(24.dp)) {
                                    Icon(Icons.Default.Close, null, tint = Color.White, modifier = Modifier.size(16.dp))
                                }
                            } else {
                                IconButton(onClick = { onSelect(model) }) {
                                    Icon(Icons.Default.Download, null, tint = if (isSafe) Color.White else Color.Gray)
                                }
                            }
                        }
                        
                        if (downloadState is DownloadState.Progress) {
                            Spacer(Modifier.height(8.dp))
                            LinearProgressIndicator(
                                progress = { downloadState.percent / 100f },
                                modifier = Modifier.fillMaxWidth().height(2.dp),
                                color = Color.White,
                                trackColor = Color(0xFF111111)
                            )
                        }

                        if (!isSafe && !isDownloaded) {
                            Text("THIS MODEL MAY FEEL HEAVY ON THIS DEVICE", style = MaterialTheme.typography.labelSmall,
                                color = Color.DarkGray, modifier = Modifier.padding(top = 4.dp))
                        }
                    }
                }
            }
            
            Spacer(Modifier.height(32.dp))
            
            Button(
                onClick = onDismiss,
                shape = RoundedCornerShape(0.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color.White, contentColor = Color.Black),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("ENTER NOMAD", fontWeight = FontWeight.Black)
            }
        }
    }
}

@Composable
fun ScanlineEffect() {
    // Component removed to clean up UI as per user request
}

@Composable
fun TerminalCursor() {
    val infiniteTransition = rememberInfiniteTransition(label = "cursor")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(500, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "alpha"
    )
    Box(
        Modifier
            .padding(start = 2.dp, top = 2.dp)
            .size(width = 8.dp, height = 18.dp)
            .alpha(alpha)
            .background(Color.White.copy(alpha = 0.7f)) // Changed to White to match theme
    )
}
@Composable
fun NomadIcon(modifier: Modifier = Modifier, color: Color = Color.White) {
    Box(modifier = modifier, contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val w = size.width
            val h = size.height

            // Star shape
            val starPath = androidx.compose.ui.graphics.Path().apply {
                moveTo(w * 0.5f, h * 0.208f)
                cubicTo(w * 0.5f, h * 0.458f, w * 0.5f, h * 0.458f, w * 0.75f, h * 0.458f)
                cubicTo(w * 0.5f, h * 0.458f, w * 0.5f, h * 0.458f, w * 0.5f, h * 0.708f)
                cubicTo(w * 0.5f, h * 0.458f, w * 0.5f, h * 0.458f, w * 0.25f, h * 0.458f)
                cubicTo(w * 0.5f, h * 0.458f, w * 0.5f, h * 0.458f, w * 0.5f, h * 0.208f)
                close()
            }
            drawPath(starPath, color)

            // Rotated ellipse
            rotate(-20f, pivot = androidx.compose.ui.geometry.Offset(w * 0.5f, h * 0.458f)) {
                val ellipsePath = androidx.compose.ui.graphics.Path().apply {
                    addOval(androidx.compose.ui.geometry.Rect(w * 0.084f, h * 0.292f, w * 0.916f, h * 0.624f))
                }
                drawPath(ellipsePath, color, style = Stroke(width = 2.dp.toPx()))
            }
        }
    }
}

@Composable
fun BouncingDots() {
    val infiniteTransition = rememberInfiniteTransition()
    val dotCount = 3
    val animations = (0 until dotCount).map { index ->
        infiniteTransition.animateFloat(
            initialValue = 0f,
            targetValue = 1f,
            animationSpec = infiniteRepeatable(
                animation = tween(600, easing = LinearEasing),
                repeatMode = RepeatMode.Reverse,
                initialStartOffset = StartOffset(index * 200)
            )
        )
    }

    Row(
        modifier = Modifier.padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        animations.forEach { anim ->
            Box(
                Modifier
                    .size(8.dp)
                    .alpha(0.3f + 0.7f * anim.value)
                    .offset(y = (-4).dp * anim.value)
                    .background(Color.White, CircleShape)
            )
        }
    }
}
@Composable
private fun LoadingOverlay(progress: Float, status: String) {
    Box(
        Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.8f)),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            CircularProgressIndicator(
                progress = { progress },
                color = Color.White,
                strokeWidth = 2.dp,
                modifier = Modifier.size(48.dp)
            )
            Spacer(Modifier.height(16.dp))
            Text(
                status,
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White,
                fontWeight = FontWeight.Bold
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "Preparing model... ${(progress * 100).toInt()}%",
                style = MaterialTheme.typography.labelSmall,
                color = Color.White,
                letterSpacing = 1.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}
