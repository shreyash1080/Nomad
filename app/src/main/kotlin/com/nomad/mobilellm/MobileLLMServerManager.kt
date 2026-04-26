package com.nomad.mobilellm

import android.content.Context
import android.util.Log
import com.nomad.utils.AssetHelper
import java.io.File
import java.util.concurrent.Executors

/**
 * MobileLLMServerManager
 * 
 * Manages the lifecycle of the Python MobileAirLLM server.
 * Extracted assets to internal storage and attempts to run the server.
 */
object MobileLLMServerManager {
    private const val TAG = "MobileLLMServer"
    private var serverProcess: Process? = null
    private val executor = Executors.newSingleThreadExecutor()

    /**
     * Starts the server by extracting assets and executing the python script.
     * Note: This assumes a python interpreter is available. 
     * In a production "no-external-app" scenario, you would integrate Chaquopy 
     * or bundle a python binary in assets/bin.
     */
    fun startServer(context: Context) {
        if (isServerRunning()) {
            Log.i(TAG, "Server is already running.")
            return
        }

        executor.execute {
            try {
                Log.i(TAG, "Extracting Python environment...")
                val envDir = AssetHelper.extractMobileLLM(context)
                val serverFile = File(envDir, "mobilellm/server.py")
                
                if (!serverFile.exists()) {
                    Log.e(TAG, "Server script not found at ${serverFile.absolutePath}")
                    return@execute
                }

                Log.i(TAG, "Starting server from ${serverFile.absolutePath}")
                
                // On non-rooted Android, we can't just run "python3". 
                // We must use the app's internal executable path or a bundled interpreter.
                // For this environment, we will check for common Python paths or fail gracefully with a better log.
                val pythonPath = listOf("/system/bin/python3", "/system/xbin/python3", "python3", "python").find { cmd ->
                    try { ProcessBuilder(cmd, "--version").start().waitFor() == 0 } catch (e: Exception) { false }
                } ?: "python3"

                val pb = ProcessBuilder(pythonPath, "-u", "mobilellm/server.py")
                pb.directory(envDir)
                pb.redirectErrorStream(true)
                
                // Ensure the python path includes our extracted directory
                pb.environment()["PYTHONPATH"] = envDir.absolutePath
                pb.environment()["PYTHONUNBUFFERED"] = "1"
                
                serverProcess = pb.start()
                
                serverProcess?.inputStream?.bufferedReader()?.use { reader ->
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        Log.d(TAG, "[Python Output] $line")
                    }
                }
                
                val exitCode = serverProcess?.waitFor()
                Log.i(TAG, "Server process exited with code $exitCode")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start server", e)
            } finally {
                serverProcess = null
            }
        }
    }

    fun stopServer() {
        Log.i(TAG, "Stopping server...")
        serverProcess?.destroy()
        serverProcess = null
    }

    fun isServerRunning(): Boolean {
        val process = serverProcess
        if (process == null) return false
        return try {
            process.exitValue()
            false
        } catch (e: IllegalThreadStateException) {
            true
        }
    }
}
