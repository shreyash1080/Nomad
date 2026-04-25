package com.nomad.data

import android.app.ActivityManager
import android.content.Context
import android.os.Environment
import android.os.StatFs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.TimeUnit

sealed class DownloadState {
    object Idle        : DownloadState()
    object Connecting  : DownloadState()
    data class Progress(val percent: Int, val bytesDownloaded: Long, val totalBytes: Long) : DownloadState()
    object Success : DownloadState()
    data class Error(val message: String) : DownloadState()
}

class ModelManager(private val context: Context) {

    private val http = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.SECONDS)    // no timeout for large files
        .followRedirects(true)
        .build()

    /** Directory where model files are stored - using media dir for persistence across reinstalls */
    val modelsDir: File get() {
        val mediaDirs = context.getExternalMediaDirs()
        val baseDir = if (mediaDirs.isNotEmpty()) {
            File(mediaDirs[0], "models")
        } else {
            File(context.filesDir, "models")
        }
        baseDir.mkdirs()
        return baseDir
    }

    fun getAvailableModels(): List<ModelInfo> = ModelCatalog.models

    data class MemoryStats(
        val physicalRamGb: Double,
        val thresholdGb: Double,
        val lowMemory: Boolean,
        val memoryClassMb: Int,
        val largeMemoryClassMb: Int,
        val availableStorageGb: Double,
        val virtualHeadroomGb: Double,
        val effectiveRamBudgetGb: Double
    )

    fun getMemoryStats(): MemoryStats {
        val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        actManager.getMemoryInfo(memInfo)
        val statFs = StatFs(Environment.getDataDirectory().absolutePath)
        val availableStorageGb = statFs.availableBytes.toDouble() / (1024 * 1024 * 1024)
        val virtualHeadroomGb = minOf(
            availableStorageGb * 0.18,
            memInfo.totalMem.toDouble() / (1024 * 1024 * 1024) * 0.5,
            4.0
        )
        val physicalRamGb = memInfo.totalMem.toDouble() / (1024 * 1024 * 1024)
        
        return MemoryStats(
            physicalRamGb = physicalRamGb,
            thresholdGb = memInfo.threshold.toDouble() / (1024 * 1024 * 1024),
            lowMemory = memInfo.lowMemory,
            memoryClassMb = actManager.memoryClass,
            largeMemoryClassMb = actManager.largeMemoryClass,
            availableStorageGb = availableStorageGb,
            virtualHeadroomGb = virtualHeadroomGb,
            effectiveRamBudgetGb = physicalRamGb + virtualHeadroomGb
        )
    }

    fun getTotalRAM(): Double = getMemoryStats().effectiveRamBudgetGb

    fun recommendBestModel(models: List<ModelInfo> = getAvailableModels()): ModelInfo {
        val stats = getMemoryStats()
        // More aggressive budget: Use 90% of physical RAM as a safe baseline, 
        // but allow up to 110% of physical RAM if the model is marked as recommended.
        val physicalBudget = stats.physicalRamGb
        
        return models
            .filter { model ->
                val requirement = model.ramRequirementGb
                // If it's a 3B+ model and we have at least 6GB RAM, it's usually fine
                if (model.paramCountValue >= 3.0 && physicalBudget >= 6.0) true
                // Otherwise, stay within a slightly more generous budget
                else requirement <= physicalBudget * 1.2
            }
            .maxByOrNull { it.paramCountValue }
            ?: models.minBy { it.ramRequirementGb }
    }

    /** Returns the File for a model if it is already downloaded */
    fun getLocalFile(model: ModelInfo): File =
        File(modelsDir, model.filename)

    fun isDownloaded(model: ModelInfo): Boolean =
        getLocalFile(model).exists()

    /** Lists all downloaded model files */
    fun listDownloaded(): List<File> =
        modelsDir.listFiles()?.filter { it.extension == "gguf" } ?: emptyList()

    /** Deletes a downloaded model file */
    fun delete(model: ModelInfo): Boolean =
        getLocalFile(model).delete()

    /**
     * Downloads [model] and emits [DownloadState] updates.
     * Resumes partial downloads automatically.
     */
    fun downloadModel(model: ModelInfo): Flow<DownloadState> = flow {
        emit(DownloadState.Connecting)

        val dest     = getLocalFile(model)
        val tempFile = File(modelsDir, "${model.filename}.part")

        val resumeFrom = if (tempFile.exists()) tempFile.length() else 0L

        try {
            val reqBuilder = Request.Builder().url(model.downloadUrl)
            if (resumeFrom > 0) {
                reqBuilder.header("Range", "bytes=$resumeFrom-")
            }

            val response = http.newCall(reqBuilder.build()).execute()
            if (!response.isSuccessful) {
                emit(DownloadState.Error("HTTP ${response.code}: ${response.message}"))
                return@flow
            }

            val body        = response.body ?: run {
                emit(DownloadState.Error("Empty response body"))
                return@flow
            }
            val contentLen  = body.contentLength()
            val totalBytes  = if (contentLen < 0) model.fileSizeBytes else resumeFrom + contentLen

            body.byteStream().use { input ->
                java.io.FileOutputStream(tempFile, resumeFrom > 0).use { output ->
                    val buf       = ByteArray(65_536)
                    var downloaded = resumeFrom
                    var lastEmitTime = 0L

                    while (true) {
                        val n = try { input.read(buf) } catch (e: Exception) { -1 }
                        if (n <= 0) break
                        output.write(buf, 0, n)
                        downloaded += n

                        val currentTime = System.currentTimeMillis()
                        if (currentTime - lastEmitTime > 500) { // Limit updates to every 500ms
                            val pct = if (totalBytes > 0)
                                ((downloaded * 100) / totalBytes).toInt()
                            else 0
                            
                            lastEmitTime = currentTime
                            emit(DownloadState.Progress(pct, downloaded, totalBytes))
                        }
                    }
                    output.flush()
                }
            }

            if (tempFile.length() >= totalBytes && totalBytes > 0) {
                tempFile.renameTo(dest)
                emit(DownloadState.Progress(100, totalBytes, totalBytes))
                emit(DownloadState.Success)
            } else {
                emit(DownloadState.Error("Download interrupted: file incomplete"))
            }

        } catch (e: Exception) {
            emit(DownloadState.Error(e.message ?: "Unknown error"))
        }
    }.flowOn(Dispatchers.IO)
}
