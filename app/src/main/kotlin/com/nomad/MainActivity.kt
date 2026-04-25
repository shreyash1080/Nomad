package com.nomad

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.*
import androidx.core.content.ContextCompat
import com.nomad.ui.ChatScreen
import com.nomad.ui.theme.NomadTheme
import com.nomad.viewmodel.ChatViewModel

class MainActivity : ComponentActivity() {

    private val viewModel: ChatViewModel by viewModels()

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val anyGranted = permissions.isEmpty() || permissions.any { it.value }
        viewModel.onPermissionResult(anyGranted)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Listen for permission requests from ViewModel
        lifecycle.addObserver(object : androidx.lifecycle.DefaultLifecycleObserver {
            override fun onResume(owner: androidx.lifecycle.LifecycleOwner) {
            }
        })

        setContent {
            NomadTheme {
                val state by viewModel.chatState.collectAsState()
                
                // When UI sets showPermissionRationale to false AND localFileHelperEnabled is still false (but was attempted)
                // it might mean we should trigger the system dialog.
                // However, the current SettingsSheet triggers viewModel.onUpdateLocalFileHelper(true) 
                // which should trigger the system dialog if we connect them.
                
                LaunchedEffect(state.showPermissionRationale) {
                    if (state.showPermissionRationale) {
                        checkPermissions()
                    }
                }

                NomadApp(viewModel)
            }
        }
    }

    private fun checkPermissions() {
        val permissions = mutableListOf<String>()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.READ_MEDIA_IMAGES)
            permissions.add(Manifest.permission.READ_MEDIA_VIDEO)
            permissions.add(Manifest.permission.READ_MEDIA_AUDIO)
        } else {
            permissions.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }

        val toRequest = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (toRequest.isNotEmpty()) {
            requestPermissionLauncher.launch(toRequest.toTypedArray())
        } else {
            viewModel.onPermissionResult(true)
        }
    }
}

@Composable
fun NomadApp(viewModel: ChatViewModel) {
    ChatScreen(viewModel)
}
