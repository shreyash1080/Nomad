package com.nomad.data

import android.content.Context
import androidx.work.*
import com.nomad.engine.FileSearchEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.TimeUnit

class FileIndexWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val db = FileIndexDatabase.getInstance(applicationContext)
        val engine = FileSearchEngine(applicationContext)
        
        try {
            val allFiles = engine.scanAllFiles()
            val total = allFiles.size
            var scannedCount = 0
            
            val existingPaths = db.dao().getAllPaths().toSet()
            val currentPaths = allFiles.map { it.absolutePath }.toSet()
            
            // Delete removed files
            existingPaths.filter { it !in currentPaths }.forEach { path ->
                db.dao().delete(path)
            }
            
            // Process each file
            allFiles.forEach { file ->
                val lastModified = file.lastModified()
                
                // Check if we need to index/re-index
                // We'll need to check the DB for the current file record
                // To optimize, maybe a query that checks if it exists and matches lastModified
                
                // For now, simpler: extract and upsert
                // A better optimization: val existing = db.dao().getByPath(file.absolutePath)
                // if (existing != null && existing.lastModifiedMs == lastModified) skip
                
                val text = engine.extractText(file)
                val keywords = tokenizeForIndex(file.name + " " + text)
                
                val entity = FileIndexEntity(
                    absolutePath = file.absolutePath,
                    fileName = file.name,
                    extension = file.extension.lowercase(),
                    mimeCategory = engine.mimeCategory(file),
                    sizeBytes = file.length(),
                    lastModifiedMs = lastModified,
                    lastIndexedAtMs = System.currentTimeMillis(),
                    extractedText = text.take(800),
                    keywords = keywords
                )
                
                db.dao().upsertWithFts(entity)
                
                scannedCount++
                setProgress(workDataOf("progress" to scannedCount, "total" to total))
            }
            
            Result.success()
        } catch (e: Exception) {
            Result.failure(workDataOf("error" to (e.message ?: "Unknown error")))
        }
    }

    private fun tokenizeForIndex(text: String): String {
        return text.lowercase()
            .replace(Regex("[^a-z0-9\\s]"), " ")
            .split(Regex("\\s+"))
            .filter { it.length >= 3 }
            .distinct()
            .joinToString(" ")
    }
}

fun scheduleIndexing(context: Context) {
    val request = OneTimeWorkRequestBuilder<FileIndexWorker>()
        .setConstraints(
            Constraints.Builder()
                .setRequiresBatteryNotLow(false)
                .build()
        )
        .build()
    WorkManager.getInstance(context).enqueueUniqueWork(
        "file_index", ExistingWorkPolicy.KEEP, request
    )
}

fun schedulePeriodicReIndex(context: Context) {
    val request = PeriodicWorkRequestBuilder<FileIndexWorker>(6, TimeUnit.HOURS).build()
    WorkManager.getInstance(context).enqueueUniquePeriodicWork(
        "file_index_periodic", ExistingPeriodicWorkPolicy.KEEP, request
    )
}
