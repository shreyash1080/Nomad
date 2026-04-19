package com.pocketllm.data

import android.content.Context
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
    data class Done(val file: File)  : DownloadState()
    data class Error(val message: String) : DownloadState()
}

class ModelManager(private val context: Context) {

    private val http = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.SECONDS)    // no timeout for large files
        .followRedirects(true)
        .build()

    /** Directory where model files are stored */
    val modelsDir: File get() {
        val dir = File(context.filesDir, "models")
        dir.mkdirs()
        return dir
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
    fun download(model: ModelInfo): Flow<DownloadState> = flow {
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
                emit(DownloadState.Done(dest))
            } else {
                emit(DownloadState.Error("Download interrupted: file incomplete"))
            }

        } catch (e: Exception) {
            emit(DownloadState.Error(e.message ?: "Unknown error"))
        }
    }.flowOn(Dispatchers.IO)
}
