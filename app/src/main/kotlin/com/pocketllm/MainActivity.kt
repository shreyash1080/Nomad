package com.pocketllm

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.SmartToy
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.pocketllm.ui.ChatScreen
import com.pocketllm.ui.ModelsScreen
import com.pocketllm.ui.theme.PocketLLMTheme
import com.pocketllm.viewmodel.ChatViewModel

class MainActivity : ComponentActivity() {

    private val viewModel: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            PocketLLMTheme {
                PocketLLMApp(viewModel)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PocketLLMApp(viewModel: ChatViewModel) {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route ?: "chat"

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    selected = currentRoute == "chat",
                    onClick  = { navController.navigate("chat") {
                        popUpTo("chat") { inclusive = true }
                    }},
                    icon     = { Icon(Icons.Default.Chat, "Chat") },
                    label    = { Text("Chat") }
                )
                NavigationBarItem(
                    selected = currentRoute == "models",
                    onClick  = { navController.navigate("models") {
                        popUpTo("chat")
                    }},
                    icon     = { Icon(Icons.Default.SmartToy, "Models") },
                    label    = { Text("Models") }
                )
            }
        }
    ) { innerPadding ->
        NavHost(
            navController    = navController,
            startDestination = "chat",
            modifier         = Modifier.padding(innerPadding)
        ) {
            composable("chat")   { ChatScreen(viewModel) }
            composable("models") { ModelsScreen(viewModel) }
        }
    }
}
