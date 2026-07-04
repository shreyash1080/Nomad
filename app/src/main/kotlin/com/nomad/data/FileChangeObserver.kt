package com.nomad.data

import android.content.Context
import android.os.FileObserver
import android.util.Log
import com.nomad.engine.FileSearchEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.io.File

class FileChangeObserver(
    private val context: Context,
    private val path: String
) : FileObserver(path, CREATE or DELETE or MODIFY or MOVED_FROM or MOVED_TO) {

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val db = FileIndexDatabase.getInstance(context)
    private val engine = FileSearchEngine(context)

    override fun onEvent(event: Int, path: String?) {
        if (path == null) return
        val fullPath = "${this.path}/$path"
        val file = File(fullPath)

        when (event) {
            CREATE, MODIFY, MOVED_TO -> {
                if (file.isFile && file.length() > 0) {
                    scope.launch {
                        val text = engine.extractText(file)
                        val entity = FileIndexEntity(
                            absolutePath = file.absolutePath,
                            fileName = file.name,
                            extension = file.extension.lowercase(),
                            mimeCategory = engine.mimeCategory(file),
                            sizeBytes = file.length(),
                            lastModifiedMs = file.lastModified(),
                            lastIndexedAtMs = System.currentTimeMillis(),
                            extractedText = text.take(800),
                            keywords = engine.tokenizeForIndex(file.name + " " + text)
                        )
                        db.dao().upsert(entity)
                    }
                }
            }
            DELETE, MOVED_FROM -> {
                scope.launch {
                    db.dao().delete(fullPath)
                }
            }
        }
    }
}
