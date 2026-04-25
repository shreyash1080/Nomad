package com.eigen.ui

import android.graphics.BitmapFactory
import android.net.Uri
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.eigen.data.*
import com.eigen.viewmodel.ChatUiState
import com.eigen.viewmodel.ChatViewModel
import com.eigen.viewmodel.PerformanceMode
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    val state     by viewModel.chatState.collectAsState()
    val listState = rememberLazyListState()
    var inputText by remember { mutableStateOf("") }
    var showSettings by remember { mutableStateOf(false) }
    
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    val filePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? -> uri?.let { viewModel.attachFile(it) } }

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
                            onVoiceInput = { /* TODO */ }
                        )
                    }
                }
            }
        ) { padding ->
            Box(Modifier.fillMaxSize().padding(padding).background(Color.Black)) {
                if (state.messages.isEmpty()) {
                    WelcomeContent(state)
                } else {
                    LazyColumn(
                        state = listState,
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(bottom = 16.dp)
                    ) {
                        items(state.messages) { msg ->
                            MessageBubble(
                                msg = msg,
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
                        isModelDownloaded = { viewModel.modelManager.getLocalFile(it).exists() }
                    )
                }
                
                if (state.loadingModel) LoadingOverlay(state.loadProgress)

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
                        onUpdateLocalFileHelper = { viewModel.updateLocalFileHelper(it) },
                        onDismissRationale = { viewModel.dismissPermissionRationale() },
                        onUpdateContextLength = { viewModel.updateContextLength(it) },
                        onUpdateLanguage = { viewModel.setLanguage(it) },
                        onSelectModel = { viewModel.loadModel(it) },
                        onDownloadMore = { 
                            viewModel.resetFirstLaunch()
                            showSettings = false
                        },
                        isModelDownloaded = { viewModel.modelManager.getLocalFile(it).exists() },
                        models = modelsState.models
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
        Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.weight(1f))
        
        EigenIcon(modifier = Modifier.size(80.dp))
        
        Spacer(Modifier.height(32.dp))
        
        Text("EIGEN", 
            style = MaterialTheme.typography.displaySmall.copy(
                fontWeight = FontWeight.Black,
                letterSpacing = 8.sp
            ),
            color = Color.White
        )
        
        Text("NEURAL NEWTON : PRIVATE INFERENCE", 
            style = MaterialTheme.typography.labelMedium,
            color = Color.Gray,
            letterSpacing = 3.sp
        )

        Spacer(Modifier.height(64.dp))
        
        if (state.loadedModel == null) {
            Surface(
                color = Color.Transparent,
                border = BorderStroke(1.dp, Color.DarkGray),
                modifier = Modifier.padding(8.dp)
            ) {
                Text(
                    "SYSTEM READY • CONFIGURE MODEL",
                    color = Color.Gray,
                    modifier = Modifier.padding(horizontal = 24.dp, vertical = 12.dp),
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.sp
                )
            }
        } else {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    "CORE ACTIVE",
                    style = MaterialTheme.typography.labelSmall,
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 2.sp
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    state.loadedModel.name.uppercase(),
                    color = Color.Gray,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Light,
                    letterSpacing = 1.sp
                )
            }
        }

        Spacer(Modifier.height(32.dp))
        Text(
            "OFFLINE • ENCRYPTED • SECURE",
            style = MaterialTheme.typography.labelSmall,
            color = Color(0xFF222222),
            letterSpacing = 2.sp
        )

        Spacer(Modifier.weight(1.5f))
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
            Text("EIGEN", 
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
        Text("OFFLINE NEURAL ENGINE", style = MaterialTheme.typography.labelSmall, color = Color.Gray, fontWeight = FontWeight.Bold)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChatTopBar(
    state: ChatUiState, 
    onMenu: () -> Unit, 
    onSettings: () -> Unit
) {
    CenterAlignedTopAppBar(
        navigationIcon = {
            IconButton(onClick = onMenu) {
                Icon(Icons.Default.Menu, "Menu", tint = Color.White)
            }
        },
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                EigenIcon(Modifier.size(20.dp))
                Spacer(Modifier.width(12.dp))
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("EIGEN", 
                        style = MaterialTheme.typography.titleLarge.copy(
                            fontWeight = FontWeight.Black,
                            letterSpacing = 1.sp
                        )
                    )
                    if (state.loadedModel != null) {
                        Surface(
                            color = Color(0xFF1A1A1A),
                            shape = RoundedCornerShape(4.dp),
                            modifier = Modifier.padding(top = 2.dp)
                        ) {
                            Text(
                                state.loadedModel.name.uppercase(),
                                style = MaterialTheme.typography.labelSmall,
                                color = Color.White,
                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
                Spacer(Modifier.width(32.dp)) // Offset to maintain visual center
            }
        },
        actions = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (state.tokensPerSec > 0f) {
                    Column(horizontalAlignment = Alignment.End, modifier = Modifier.padding(end = 8.dp)) {
                        Text("${"%.1f".format(state.tokensPerSec)} t/s", style = MaterialTheme.typography.labelSmall, color = Color(0xFF00FF00), fontWeight = FontWeight.Bold)
                        Text("${state.firstTokenLatencyMs}ms", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                    }
                }
                IconButton(onClick = onSettings) {
                    Icon(Icons.Default.Settings, "Settings", tint = Color.White)
                }
            }
        },
        colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
            containerColor = Color.Black,
            titleContentColor = Color.White
        )
    )
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
    onUpdateLocalFileHelper: (Boolean) -> Unit,
    onDismissRationale: () -> Unit,
    onUpdateContextLength: (Int) -> Unit,
    onUpdateLanguage: (String) -> Unit,
    onSelectModel: (ModelInfo) -> Unit,
    onDownloadMore: () -> Unit,
    isModelDownloaded: (ModelInfo) -> Boolean,
    models: List<ModelInfo>
) {
    var name by remember { mutableStateOf(state.userName) }
    var systemPrompt by remember { mutableStateOf(state.systemPrompt) }
    var showLangMenu by remember { mutableStateOf(false) }
    var showModelMenu by remember { mutableStateOf(false) }

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
                Text("CONFIGURATION", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Black)
                Spacer(Modifier.weight(1f))
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Default.Close, null, tint = Color.White)
                }
            }
            
            Spacer(Modifier.height(32.dp))

            // --- MODEL SELECTION ---
            Text("Active Model", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.height(12.dp))
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
                                    Text(model.name, color = Color.White)
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
            Text("Performance Optimization", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.height(12.dp))

            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PerformanceMode.values().forEach { mode ->
                    val isSelected = state.performanceMode == mode
                    Surface(
                        onClick = { onUpdatePerfMode(mode) },
                        modifier = Modifier.weight(1f),
                        color = if (isSelected) Color.White else Color(0xFF111111),
                        shape = RoundedCornerShape(8.dp),
                        border = BorderStroke(1.dp, if (isSelected) Color.White else Color(0xFF222222))
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
                    PerformanceMode.HIGH -> "Max performance. Uses more battery and may generate heat."
                    PerformanceMode.BALANCED -> "Optimal balance between speed and battery life."
                    PerformanceMode.POWER_SAVER -> "Slows down generation to keep device cool and save power."
                },
                style = MaterialTheme.typography.labelSmall,
                color = Color.DarkGray,
                modifier = Modifier.padding(top = 8.dp, start = 4.dp)
            )

            Spacer(Modifier.height(24.dp))

            // --- USER SETTINGS ---
            Text("User Profile", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.height(12.dp))
            
            Text("Display Name", style = MaterialTheme.typography.labelMedium, color = Color.Gray)
            Spacer(Modifier.height(8.dp))
            OutlinedTextField(
                value = name,
                onValueChange = { 
                    name = it
                    onUpdateName(it)
                },
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(0.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Color.White,
                    unfocusedBorderColor = Color(0xFF222222),
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White
                )
            )

            Spacer(Modifier.height(24.dp))

            // --- AI MODEL SETTINGS ---
            Text("AI Model Settings", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.height(12.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("GPU Acceleration", style = MaterialTheme.typography.bodyLarge)
                    Text("Use RAM/GPU for faster inference", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                }
                Switch(
                    checked = state.useGpu,
                    onCheckedChange = onUpdateUseGpu,
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = Color.Black,
                        checkedTrackColor = Color.White,
                        uncheckedThumbColor = Color.Gray,
                        uncheckedTrackColor = Color(0xFF111111)
                    )
                )
            }

            Spacer(Modifier.height(16.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Save Chat History", style = MaterialTheme.typography.bodyLarge)
                    Text("Auto-save conversations to local storage", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                }
                Switch(
                    checked = state.saveChat,
                    onCheckedChange = onUpdateSaveChat,
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = Color.Black,
                        checkedTrackColor = Color.White,
                        uncheckedThumbColor = Color.Gray,
                        uncheckedTrackColor = Color(0xFF111111)
                    )
                )
            }

            Spacer(Modifier.height(16.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Local File Helper", style = MaterialTheme.typography.bodyLarge)
                    Text("Allow AI to read your local files (images, docs) to give better answers.", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                }
                Switch(
                    checked = state.localFileHelperEnabled,
                    onCheckedChange = onUpdateLocalFileHelper,
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = Color.Black,
                        checkedTrackColor = Color.White,
                        uncheckedThumbColor = Color.Gray,
                        uncheckedTrackColor = Color(0xFF111111)
                    )
                )
            }

            Spacer(Modifier.height(24.dp))

            // --- PERMISSION RATIONALE DIALOG ---
            if (state.showPermissionRationale) {
                AlertDialog(
                    onDismissRequest = onDismissRationale,
                    title = { Text("Storage Permission", color = Color.White) },
                    text = { Text("Eigen needs access to your files to analyze them. This is only used locally on your device.", color = Color.Gray) },
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
                            Text("CANCEL", color = Color.Gray)
                        }
                    },
                    containerColor = Color(0xFF1A1A1A)
                )
            }

            Spacer(Modifier.height(24.dp))

            // --- REGIONAL SETTINGS ---
            Text("Regional Settings", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.height(12.dp))

            Column {
                Text("AI Language Preference", style = MaterialTheme.typography.labelMedium, color = Color.Gray)
                Spacer(Modifier.height(8.dp))
                Box {
                    OutlinedButton(
                        onClick = { showLangMenu = true },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp),
                        border = BorderStroke(1.dp, Color(0xFF333333)),
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
                        modifier = Modifier.background(Color(0xFF1A1A1A)).fillMaxWidth(0.8f)
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
                    "Note: Switching to ${state.preferredLanguage} will force the AI to respond in that language.",
                    style = MaterialTheme.typography.labelSmall,
                    color = Color.DarkGray,
                    modifier = Modifier.padding(top = 4.dp, start = 4.dp)
                )
            }

            Spacer(Modifier.height(24.dp))

            Text("Creativity (Temperature): ${"%.1f".format(state.temperature)}", style = MaterialTheme.typography.bodyMedium)
            Slider(
                value = state.temperature,
                onValueChange = onUpdateTemp,
                valueRange = 0.1f..1.5f,
                steps = 14,
                colors = SliderDefaults.colors(thumbColor = Color.White, activeTrackColor = Color.White)
            )

            Spacer(Modifier.height(16.dp))

            Text("Max Response Length: ${state.maxTokens} tokens", style = MaterialTheme.typography.bodyMedium)
            Text("Controls how long the AI's answers can be.", style = MaterialTheme.typography.labelSmall, color = Color.DarkGray)
            Slider(
                value = state.maxTokens.toFloat(),
                onValueChange = { onUpdateMaxTokens(it.toInt()) },
                valueRange = 64f..2048f,
                steps = 31,
                colors = SliderDefaults.colors(thumbColor = Color.White, activeTrackColor = Color.White)
            )

            Spacer(Modifier.height(16.dp))

            Text("Context Length: ${state.contextLength} tokens", style = MaterialTheme.typography.bodyMedium)
            Text("How much the AI remembers in one conversation.", style = MaterialTheme.typography.labelSmall, color = Color.DarkGray)
            Slider(
                value = state.contextLength.toFloat(),
                onValueChange = { onUpdateContextLength(it.toInt()) },
                valueRange = 512f..8192f,
                steps = 15,
                colors = SliderDefaults.colors(thumbColor = Color.White, activeTrackColor = Color.White)
            )

            Spacer(Modifier.height(24.dp))

            Text("System Instructions", style = MaterialTheme.typography.labelMedium, color = Color.Gray)
            Text("Overrides how the AI behaves.", style = MaterialTheme.typography.labelSmall, color = Color.DarkGray)
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
                    focusedBorderColor = Color(0xFF4285F4),
                    unfocusedBorderColor = Color(0xFF333333),
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
        color = Color(0xFF1A1A1A),
        shape = RoundedCornerShape(8.dp),
        border = BorderStroke(1.dp, Color(0xFF333333))
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
                    color = Color.White,
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
private fun MessageBubble(
    msg: ChatMessage,
    onEdit: (Long, String) -> Unit
) {
    if (msg.role == Role.SYSTEM) {
        Box(
            Modifier.fillMaxWidth().padding(horizontal = 32.dp, vertical = 8.dp),
            contentAlignment = Alignment.Center
        ) {
            Surface(
                color = Color(0xFF111111),
                shape = RoundedCornerShape(4.dp),
                border = BorderStroke(1.dp, Color(0xFF222222))
            ) {
                Text(msg.content.uppercase(),
                    style     = MaterialTheme.typography.labelSmall,
                    modifier  = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                    textAlign = TextAlign.Center,
                    color     = Color.Gray,
                    letterSpacing = 1.sp)
            }
        }
        return
    }

    val isUser = msg.role == Role.USER
    val clipboard = LocalClipboardManager.current
    var isEditing by remember { mutableStateOf(false) }
    var editValue by remember { mutableStateOf(msg.content) }

    Row(
        modifier            = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        if (!isUser) {
            Box(
                Modifier
                    .size(32.dp)
                    .padding(top = 4.dp),
                contentAlignment = Alignment.Center
            ) {
                EigenIcon(modifier = Modifier.size(28.dp))
            }
            Spacer(Modifier.width(12.dp))
        }

        Column(
            modifier = Modifier.weight(1f, fill = false),
            horizontalAlignment = if (isUser) Alignment.End else Alignment.Start
        ) {
            msg.attachment?.let { att ->
                if (att.type == AttachmentType.IMAGE && att.thumbnailBase64 != null) {
                    val bytes = Base64.decode(att.thumbnailBase64, Base64.DEFAULT)
                    val bmp   = try { BitmapFactory.decodeByteArray(bytes, 0, bytes.size) } catch(_: Exception) { null }
                    if (bmp != null) {
                        Image(
                            bitmap            = bmp.asImageBitmap(),
                            contentDescription = "Attached image",
                            modifier          = Modifier
                                .widthIn(max = 260.dp)
                                .border(1.dp, Color(0xFF222222), RoundedCornerShape(4.dp))
                                .padding(2.dp)
                                .clip(RoundedCornerShape(2.dp))
                                .padding(bottom = 12.dp),
                            contentScale      = ContentScale.FillWidth
                        )
                    }
                }
            }

            Surface(
                shape = if (isUser) RoundedCornerShape(16.dp) else RoundedCornerShape(0.dp),
                color = if (isUser && isEditing) Color(0xFF1A1A1A) else Color.Transparent,
                border = null,
                modifier = Modifier.widthIn(max = 320.dp)
            ) {
                Column(Modifier.padding(horizontal = 16.dp, vertical = 12.dp)) {
                    if (!isUser && msg.thought != null) {
                        var expanded by remember { mutableStateOf(false) }
                        Column(
                            Modifier
                                .fillMaxWidth()
                                .padding(bottom = 12.dp)
                                .background(Color(0xFF0F0F0F), RoundedCornerShape(4.dp))
                                .border(1.dp, Color(0xFF1A1A1A), RoundedCornerShape(4.dp))
                                .clickable { expanded = !expanded }
                                .padding(12.dp)
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    Icons.Default.Psychology, 
                                    null, 
                                    tint = Color.White, 
                                    modifier = Modifier.size(16.dp)
                                )
                                Spacer(Modifier.width(8.dp))
                                Text(
                                    "THOUGHT PROCESS", 
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color.White,
                                    fontWeight = FontWeight.Bold,
                                    modifier = Modifier.weight(1f)
                                )
                                Icon(
                                    if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                                    null,
                                    tint = Color.DarkGray,
                                    modifier = Modifier.size(16.dp)
                                )
                            }
                            if (expanded) {
                                Spacer(Modifier.height(8.dp))
                                Text(
                                    text = msg.thought,
                                    color = Color.Gray,
                                    style = MaterialTheme.typography.bodySmall.copy(
                                        lineHeight = 18.sp,
                                        fontStyle = androidx.compose.ui.text.font.FontStyle.Italic
                                    )
                                )
                            }
                        }
                    }

                    if (msg.isStreaming && msg.content == "...") {
                        BouncingDots()
                    } else if (!isUser && msg.content.contains("```")) {
                        val parts = msg.content.split("```")
                        parts.forEachIndexed { index, part ->
                            if (index % 2 == 0) {
                                if (part.isNotBlank()) {
                                    Text(
                                        text = part.trim(),
                                        color = Color(0xFFDDDDDD),
                                        style = MaterialTheme.typography.bodyLarge.copy(lineHeight = 26.sp)
                                    )
                                }
                            } else {
                                CodeBlock(part)
                            }
                        }
                    } else {
                        Text(
                            text       = msg.content,
                            color      = if (isUser) Color.White else Color(0xFFDDDDDD),
                            style      = MaterialTheme.typography.bodyLarge.copy(
                                lineHeight = 26.sp,
                                letterSpacing = 0.1.sp
                            ),
                            fontFamily = if (msg.content.contains("```")) FontFamily.Monospace
                            else FontFamily.SansSerif
                        )
                    }

                    if (msg.isStreaming) {
                        TerminalCursor()
                    }

                    if (isEditing) {
                        OutlinedTextField(
                            value = editValue,
                            onValueChange = { editValue = it },
                            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedTextColor = Color.White,
                                unfocusedTextColor = Color.White
                            )
                        )
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                            TextButton(onClick = { isEditing = false }) { Text("Cancel", color = Color.Gray) }
                            TextButton(onClick = { 
                                onEdit(msg.id, editValue)
                                isEditing = false 
                            }) { Text("SAVE", color = Color.White, fontWeight = FontWeight.Bold) }
                        }
                    } else {
                        // --- Message Actions ---
                        Row(
                            Modifier
                                .fillMaxWidth()
                                .padding(top = 8.dp),
                            horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            IconButton(onClick = { clipboard.setText(AnnotatedString(msg.content)) }, modifier = Modifier.size(24.dp)) {
                                Icon(Icons.Default.ContentCopy, "Copy", tint = Color.DarkGray, modifier = Modifier.size(14.dp))
                            }
                            if (isUser) {
                                Spacer(Modifier.width(8.dp))
                                IconButton(onClick = { isEditing = true }, modifier = Modifier.size(24.dp)) {
                                    Icon(Icons.Default.Edit, "Edit", tint = Color.DarkGray, modifier = Modifier.size(14.dp))
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if (isUser) {
            Spacer(Modifier.width(16.dp))
            Box(
                Modifier.size(24.dp).padding(top = 4.dp).background(Color(0xFF333333), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.Person, null, modifier = Modifier.size(14.dp), tint = Color.LightGray)
            }
        }
    }
}

@Composable
private fun CodeBlock(code: String) {
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
            .padding(vertical = 8.dp)
            .background(Color(0xFF0D0D0D), RoundedCornerShape(8.dp))
            .border(1.dp, Color(0xFF1A1A1A), RoundedCornerShape(8.dp))
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xFF1A1A1A))
                .padding(horizontal = 12.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                "TERMINAL", 
                style = MaterialTheme.typography.labelSmall, 
                color = Color.Gray,
                fontWeight = FontWeight.Bold
            )
            Text(
                if (copied) "COPIED" else "COPY",
                modifier = Modifier.clickable {
                    clipboardManager.setText(AnnotatedString(code.trim()))
                    copied = true
                },
                style = MaterialTheme.typography.labelSmall,
                color = Color.White,
                fontWeight = FontWeight.Bold
            )
        }
        
        Box(
            modifier = Modifier
                .horizontalScroll(rememberScrollState())
                .padding(16.dp)
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
    onVoiceInput: () -> Unit
) {
    Surface(
        color = Color.Black,
        tonalElevation = 0.dp
    ) {
        Row(
            modifier = Modifier
                .padding(horizontal = 16.dp, vertical = 12.dp)
                .fillMaxWidth()
                .background(Color(0xFF111111), RoundedCornerShape(28.dp))
                .padding(horizontal = 4.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(
                onClick = onVoiceInput,
                modifier = Modifier.size(48.dp)
            ) {
                Icon(
                    Icons.Default.Mic,
                    "Voice Input",
                    tint = Color.Gray
                )
            }

            IconButton(
                onClick = onAttach,
                modifier = Modifier.size(48.dp)
            ) {
                Icon(
                    Icons.Default.Add,
                    "Attach",
                    tint = if (hasAttachment) Color.White else Color.Gray
                )
            }

            BasicTextField(
                value = text,
                onValueChange = onTextChange,
                modifier = Modifier
                    .weight(1f)
                    .padding(horizontal = 4.dp),
                textStyle = LocalTextStyle.current.copy(color = Color.White, fontSize = 16.sp),
                cursorBrush = SolidColor(Color.White),
                decorationBox = { innerTextField ->
                    Box(contentAlignment = Alignment.CenterStart) {
                        if (text.isEmpty()) {
                            Text("Ask Eigen...", color = Color.DarkGray, fontSize = 16.sp)
                        }
                        innerTextField()
                    }
                }
            )

            if (isGenerating) {
                IconButton(
                    onClick = onStop,
                    modifier = Modifier
                        .size(40.dp)
                        .background(Color.White, CircleShape)
                ) {
                    Icon(Icons.Default.Stop, "Stop", tint = Color.Black, modifier = Modifier.size(20.dp))
                }
            } else {
                IconButton(
                    onClick = onSend,
                    enabled = text.isNotBlank() || hasAttachment,
                    modifier = Modifier
                        .size(40.dp)
                        .background(
                            if (text.isNotBlank() || hasAttachment) Color.White else Color.Transparent,
                            CircleShape
                        )
                ) {
                    Icon(
                        Icons.Default.ArrowUpward,
                        "Send",
                        tint = if (text.isNotBlank() || hasAttachment) Color.Black else Color.DarkGray,
                        modifier = Modifier.size(20.dp)
                    )
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
            EigenIcon(Modifier.size(48.dp))
            Spacer(Modifier.height(16.dp))
            Text(
                "NEURAL HUB",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Black,
                color = Color.White,
                letterSpacing = 2.sp
            )
            Text(
                "Configure your local inference engine",
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray,
                textAlign = TextAlign.Center
            )
            
            Spacer(Modifier.height(32.dp))

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
                                Text(model.name.uppercase(), fontWeight = FontWeight.Black, color = if (isSafe) Color.White else Color.Gray)
                                Text("${model.paramCount} • ${model.ramRequirementGb}GB RAM REQ", 
                                    style = MaterialTheme.typography.labelSmall, color = Color.Gray)
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
                            Text("DEVICE MAY REQUIRE MORE RAM", style = MaterialTheme.typography.labelSmall, 
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
                Text("ENTER EIGEN", fontWeight = FontWeight.Black)
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
            .padding(start = 2.dp)
            .size(width = 8.dp, height = 16.dp)
            .alpha(alpha)
            .background(Color.White)
    )
}
@Composable
fun EigenIcon(modifier: Modifier = Modifier, color: Color = Color.White) {
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
private fun LoadingOverlay(progress: Float) {
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
                "NEURAL ENGINE BOOTING... ${(progress * 100).toInt()}%", 
                style = MaterialTheme.typography.labelSmall,
                color = Color.White,
                letterSpacing = 1.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}
