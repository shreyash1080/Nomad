package com.nomad.data

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

data class ChatSession(
    val id: String = UUID.randomUUID().toString(),
    val title: String,
    val timestamp: Long = System.currentTimeMillis(),
    val messages: List<ChatMessage>
)

class ChatHistoryManager(context: Context) {
    private val gson = Gson()
    private val historyDir = File(context.filesDir, "chat_history").apply { if (!exists()) mkdirs() }

    suspend fun saveChat(session: ChatSession) = withContext(Dispatchers.IO) {
        val file = File(historyDir, "${session.id}.json")
        file.writeText(gson.toJson(session))
    }

    suspend fun getAllChats(): List<ChatSession> = withContext(Dispatchers.IO) {
        historyDir.listFiles()
            ?.filter { it.extension == "json" }
            ?.mapNotNull { 
                try {
                    gson.fromJson(it.readText(), ChatSession::class.java)
                } catch (e: Exception) { null }
            }
            ?.sortedByDescending { it.timestamp }
            ?: emptyList()
    }

    suspend fun deleteChat(id: String) = withContext(Dispatchers.IO) {
        File(historyDir, "$id.json").delete()
    }
}
