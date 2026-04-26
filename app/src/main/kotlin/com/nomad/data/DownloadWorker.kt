package com.nomad.data

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.work.*
import kotlinx.coroutines.flow.collect

class DownloadWorker(
    private val context: Context,
    workerParams: WorkerParameters
) : CoroutineWorker(context, workerParams) {

    private val notificationManager =
        context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    override suspend fun doWork(): Result {
        val modelId = inputData.getString("model_id") ?: return Result.failure()
        val model = ModelCatalog.models.find { it.id == modelId } ?: return Result.failure()

        // Create notification channel
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                "downloads",
                "Model Downloads",
                NotificationManager.IMPORTANCE_LOW
            )
            notificationManager.createNotificationChannel(channel)
        }

        setForeground(createForegroundInfo(model.name, 0))

        val modelManager = ModelManager(context)
        var result = Result.success()

        try {
            modelManager.downloadModel(model).collect { state ->
                when (state) {
                    is DownloadState.Progress -> {
                        setProgress(workDataOf("progress" to state.percent))
                        if (hasNotificationPermission()) {
                            notificationManager.notify(
                                modelId.hashCode(),
                                createNotification(model.name, state.percent)
                            )
                        }
                    }
                    is DownloadState.Success -> {
                        if (hasNotificationPermission()) {
                            notificationManager.notify(
                                modelId.hashCode(),
                                NotificationCompat.Builder(context, "downloads")
                                    .setSmallIcon(android.R.drawable.stat_sys_download_done)
                                    .setContentTitle("Download Complete")
                                    .setContentText(model.name)
                                    .setPriority(NotificationCompat.PRIORITY_LOW)
                                    .build()
                            )
                        }
                    }
                    is DownloadState.Error -> {
                        Log.e("DownloadWorker", "Download error: ${state.message}")
                        result = Result.failure(workDataOf("error" to state.message))
                    }
                    else -> {}
                }
            }
        } catch (e: Exception) {
            Log.e("DownloadWorker", "Worker exception", e)
            result = Result.failure(workDataOf("error" to (e.message ?: "Unknown error")))
        }

        return result
    }

    private fun hasNotificationPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ActivityCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED
        } else {
            true
        }
    }

    private fun createForegroundInfo(modelName: String, progress: Int): ForegroundInfo {
        val notification = createNotification(modelName, progress)
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ForegroundInfo(
                modelName.hashCode(),
                notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            )
        } else {
            ForegroundInfo(modelName.hashCode(), notification)
        }
    }

    private fun createNotification(modelName: String, progress: Int) =
        NotificationCompat.Builder(context, "downloads")
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("Downloading $modelName")
            .setContentText("$progress%")
            .setProgress(100, progress, false)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
}
