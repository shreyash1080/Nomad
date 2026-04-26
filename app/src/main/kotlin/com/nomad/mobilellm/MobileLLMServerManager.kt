package com.nomad.mobilellm

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * MobileLLMServerManager
 *
 * Manages the Python MobileAirLLM server process lifecycle.
 *
 * Fixes vs original:
 *  - Searches correct Termux Python paths (was only checking /system/bin which
 *    never has Python on Android without root).
 *  - Uses AtomicBoolean for thread-safe running state.
 *  - Sends SIGTERM for clean shutdown instead of destroy() (SIGKILL).
 *  - Captures server stdout/stderr in a dedicated log thread so the process
 *    doesn't deadlock when the output pipe fills up.
 *  - isServerRunning() no longer blocks on a network call.
 */
object MobileLLMServerManager {
    private const val TAG = "MobileLLMServer"
    private const val SERVER_PORT = 8765

    @Volatile
    private var serverProcess: Process? = null
    private val _isRunning = AtomicBoolean(false)
    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "MobileLLMServer").apply { isDaemon = true }
    }

    // ── Python binary candidates — Termux first, then system fallbacks ────────
    private val PYTHON_CANDIDATES = listOf(
        // Termux standard location (most common)
        "/data/data/com.termux/files/usr/bin/python3",
        "/data/data/com.termux/files/usr/bin/python",
        // Termux with different package names
        "/data/user/0/com.termux/files/usr/bin/python3",
        // UserLAnd / proot-distro paths
        "/data/data/tech.ula/files/usr/bin/python3",
        // System (rooted devices / some custom ROMs)
        "/system/bin/python3",
        "/system/xbin/python3",
        // PATH fallback
        "python3",
        "python",
    )

    /**
     * Attempts to start the MobileAirLLM server.
     * Call this from a coroutine (it posts work to a background executor).
     * The actual HTTP readiness must be checked via [isServerResponding].
     */
    fun startServer(context: Context) {
        if (_isRunning.get()) {
            Log.i(TAG, "Server already running, skipping start.")
            return
        }

        executor.execute {
            try {
                val python = findPythonBinary()
                if (python == null) {
                    Log.e(TAG, """
                        ┌─────────────────────────────────────────────────────┐
                        │ Python not found on this device.                    │
                        │ Install Termux from F-Droid and run:                │
                        │   pkg install python && pip install mobilellm       │
                        └─────────────────────────────────────────────────────┘
                    """.trimIndent())
                    return@execute
                }

                Log.i(TAG, "Using Python: $python")

                // Find the mobilellm package (installed via pip in Termux)
                val envDir = findMobileLLMRoot(python)
                if (envDir == null) {
                    Log.e(TAG, "mobilellm package not found. Run: pip install mobilellm")
                    return@execute
                }

                Log.i(TAG, "Starting server from $envDir")

                val pb = ProcessBuilder(
                    python, "-m", "mobilellm.server",
                    "--host", "127.0.0.1",
                    "--port", SERVER_PORT.toString()
                )
                pb.directory(File(envDir))
                pb.redirectErrorStream(true)

                // Set PYTHONPATH and Termux lib paths
                pb.environment().apply {
                    put("PYTHONUNBUFFERED", "1")
                    put("PYTHONPATH", envDir)
                    // Termux lib path so PyTorch .so files load
                    val termuxLib = "/data/data/com.termux/files/usr/lib"
                    val existing = get("LD_LIBRARY_PATH") ?: ""
                    put("LD_LIBRARY_PATH", if (existing.isNotEmpty()) "$termuxLib:$existing" else termuxLib)
                }

                val proc = pb.start()
                serverProcess = proc
                _isRunning.set(true)

                // Drain stdout/stderr in separate thread to prevent pipe deadlock
                Thread({
                    try {
                        proc.inputStream.bufferedReader().use { reader ->
                            reader.lineSequence().forEach { line ->
                                Log.d(TAG, "[py] $line")
                            }
                        }
                    } catch (_: Exception) {}
                }, "MobileLLMLog").apply { isDaemon = true; start() }

                val exitCode = proc.waitFor()
                Log.i(TAG, "Server process exited: $exitCode")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start server", e)
            } finally {
                _isRunning.set(false)
                serverProcess = null
            }
        }
    }

    /**
     * Graceful shutdown: sends SIGTERM first, then SIGKILL after 3s.
     */
    fun stopServer() {
        Log.i(TAG, "Stopping MobileAirLLM server…")
        val proc = serverProcess ?: return
        try {
            // SIGTERM — lets Python flush and exit cleanly
            Runtime.getRuntime().exec("kill -TERM ${pid(proc)}")
            Thread.sleep(3000)
        } catch (_: Exception) {}
        if (_isRunning.get()) {
            proc.destroy()
        }
        serverProcess = null
        _isRunning.set(false)
    }

    /** Quick process-level check (no network call). */
    fun isProcessRunning(): Boolean {
        val proc = serverProcess ?: return false
        return try {
            proc.exitValue()
            false  // process has exited
        } catch (_: IllegalThreadStateException) {
            true   // still running
        }
    }

    /** Full HTTP liveness check — use this to confirm server is accepting requests. */
    suspend fun isServerResponding(): Boolean = withContext(Dispatchers.IO) {
        try {
            val url = java.net.URL("http://127.0.0.1:$SERVER_PORT/health")
            val conn = url.openConnection() as java.net.HttpURLConnection
            conn.connectTimeout = 1500
            conn.readTimeout = 1500
            val code = conn.responseCode
            conn.disconnect()
            code == 200
        } catch (_: Exception) {
            false
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    private fun findPythonBinary(): String? {
        for (candidate in PYTHON_CANDIDATES) {
            try {
                val result = ProcessBuilder(candidate, "--version")
                    .redirectErrorStream(true)
                    .start()
                    .waitFor()
                if (result == 0) {
                    Log.d(TAG, "Found Python: $candidate")
                    return candidate
                }
            } catch (_: Exception) {}
        }
        return null
    }

    /**
     * Finds the directory that contains the mobilellm package by running:
     *   python -c "import mobilellm; print(mobilellm.__file__)"
     */
    private fun findMobileLLMRoot(python: String): String? {
        return try {
            val proc = ProcessBuilder(
                python, "-c",
                "import mobilellm, os; print(os.path.dirname(os.path.dirname(mobilellm.__file__)))"
            ).redirectErrorStream(true).start()
            val output = proc.inputStream.bufferedReader().readText().trim()
            proc.waitFor()
            if (output.isNotEmpty() && File(output).isDirectory) output else null
        } catch (e: Exception) {
            Log.e(TAG, "Could not locate mobilellm package: $e")
            null
        }
    }

    private fun pid(proc: Process): Int {
        return try {
            val f = proc.javaClass.getDeclaredField("pid")
            f.isAccessible = true
            f.getInt(proc)
        } catch (_: Exception) {
            -1
        }
    }
}