package com.nomad.engine

import android.content.Context
import android.graphics.Bitmap
import android.graphics.pdf.PdfRenderer
import android.os.Environment
import android.os.ParcelFileDescriptor
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.zip.ZipFile
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

data class FileMatch(
    val file: File,
    val mimeCategory: String,
    val extractedText: String,
    val score: Float,
    val matchReason: String
)

class FileSearchEngine(private val context: Context) {

    companion object {
        private const val TAG = "FileSearchEngine"
        private const val LLM_RERANK_LIMIT = 25

        private val EXTENSION_MAP = mapOf(
            "jpg" to "image", "jpeg" to "image", "png" to "image",
            "gif" to "image", "bmp" to "image", "webp" to "image",
            "heic" to "image", "heif" to "image",
            "pdf" to "pdf",
            "doc" to "document", "docx" to "document", "odt" to "document",
            "xls" to "spreadsheet", "xlsx" to "spreadsheet", "ods" to "spreadsheet", "csv" to "spreadsheet",
            "ppt" to "presentation", "pptx" to "presentation",
            "txt" to "text", "md" to "text", "rtf" to "text", "json" to "text",
            "xml" to "text", "html" to "text", "htm" to "text"
        )

        private val SCAN_ROOTS = listOf(
            Environment.getExternalStorageDirectory(),
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS),
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM)
        )
        
        private val MONTHS = mapOf(
            "01" to "january jan", "02" to "february feb", "03" to "march mar",
            "04" to "april apr", "05" to "may", "06" to "june jun",
            "07" to "july jul", "08" to "august aug", "09" to "september sep",
            "10" to "october oct", "11" to "november nov", "12" to "december dec"
        )
    }

    suspend fun scanAllFiles(): List<File> = withContext(Dispatchers.IO) {
        val results = mutableListOf<File>()
        val visited = mutableSetOf<String>()

        fun walk(dir: File) {
            if (!dir.exists() || !dir.isDirectory) return
            val canonical = runCatching { dir.canonicalPath }.getOrNull() ?: return
            if (!visited.add(canonical)) return
            dir.listFiles()?.forEach { f ->
                if (f.isDirectory) walk(f)
                else if (f.isFile && f.length() > 0 && f.extension.lowercase() in EXTENSION_MAP) results.add(f)
            }
        }

        SCAN_ROOTS.forEach { walk(it) }
        context.filesDir?.let { walk(it) }
        context.getExternalFilesDir(null)?.let { walk(it) }
        results
    }

    suspend fun extractText(file: File): String = withContext(Dispatchers.IO) {
        val ext = file.extension.lowercase()
        runCatching {
            when (EXTENSION_MAP[ext]) {
                "image" -> extractImageText(file)
                "pdf" -> extractPdfText(file)
                "document" -> if (ext == "docx") extractDocxText(file) else file.readQuick()
                "spreadsheet" -> if (ext == "xlsx") extractXlsxText(file) else if (ext == "csv") file.readQuick() else ""
                "text" -> file.readQuick()
                else -> ""
            }
        }.getOrElse { "" }
    }

    private suspend fun extractImageText(file: File): String = suspendCoroutine { cont ->
        try {
            val image = InputImage.fromFilePath(context, android.net.Uri.fromFile(file))
            val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            recognizer.process(image)
                .addOnSuccessListener { result -> cont.resume(result.text.take(1000)) }
                .addOnFailureListener { cont.resume("") }
        } catch (e: Exception) { cont.resume("") }
    }

    private suspend fun extractPdfText(file: File): String = withContext(Dispatchers.IO) {
        val sb = StringBuilder()
        runCatching {
            val pfd = ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY)
            PdfRenderer(pfd).use { renderer ->
                val pageCount = minOf(renderer.pageCount, 3)
                for (i in 0 until pageCount) {
                    renderer.openPage(i).use { page ->
                        val bmp = Bitmap.createBitmap(page.width, page.height, Bitmap.Config.ARGB_8888)
                        page.render(bmp, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                        val ocrText = suspendCoroutine<String> { cont ->
                            TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
                                .process(InputImage.fromBitmap(bmp, 0))
                                .addOnSuccessListener { r -> cont.resume(r.text) }
                                .addOnFailureListener { cont.resume("") }
                        }
                        sb.append(ocrText).append(" ")
                        bmp.recycle()
                    }
                }
            }
        }
        sb.toString().take(1000)
    }

    private fun extractDocxText(file: File): String = runCatching {
        ZipFile(file).use { zip ->
            val entry = zip.getEntry("word/document.xml") ?: return ""
            zip.getInputStream(entry).bufferedReader().readText()
                .replace(Regex("<[^>]+>"), " ").replace(Regex("\\s+"), " ").trim().take(1000)
        }
    }.getOrElse { "" }

    private fun extractXlsxText(file: File): String = runCatching {
        ZipFile(file).use { zip ->
            val entry = zip.getEntry("xl/sharedStrings.xml") ?: return ""
            zip.getInputStream(entry).bufferedReader().readText()
                .replace(Regex("<[^>]+>"), " ").replace(Regex("\\s+"), " ").trim().take(1000)
        }
    }.getOrElse { "" }

    private fun File.readQuick(): String = inputStream().bufferedReader().use { r ->
        val buf = CharArray(1000)
        val read = r.read(buf)
        if (read > 0) String(buf, 0, read) else ""
    }

    fun tokenizeForIndex(text: String): String {
        val raw = text.lowercase()
        val tokens = mutableSetOf<String>()
        
        // 1. Basic splitting
        raw.split(Regex("[^a-z0-9]")).filter { it.length >= 2 }.forEach { tokens.add(it) }
        
        // 2. Date decomposition (e.g., 20260415 -> 2026, 04, 15, april)
        Regex("(\\d{4})(\\d{2})(\\d{2})").findAll(raw).forEach { m ->
            tokens.add(m.groupValues[1])
            val month = m.groupValues[2]
            tokens.add(month)
            tokens.add(m.groupValues[3])
            MONTHS[month]?.let { it.split(" ").forEach { t -> tokens.add(t) } }
        }
        
        // 3. Month detection in text
        MONTHS.forEach { (num, names) ->
            if (raw.contains(num)) names.split(" ").forEach { tokens.add(it) }
            names.split(" ").forEach { name -> if (raw.contains(name)) tokens.add(num) }
        }

        return tokens.joinToString(" ")
    }

    fun scoreFile(file: File, extractedText: String, description: String): Pair<Float, String> {
        val descLower = description.lowercase()
        val queryTokens = descLower.split(Regex("[^a-z0-9]")).filter { it.length >= 2 && it !in STOP_WORDS }
        val nameLower = file.name.lowercase()
        val contentLower = extractedText.lowercase()
        
        var score = 0f
        val reasons = mutableListOf<String>()

        // Type match
        val isPhotoQuery = descLower.contains("phot") || descLower.contains("img") || descLower.contains("pic")
        if (isPhotoQuery && EXTENSION_MAP[file.extension.lowercase()] == "image") {
            score += 0.3f
            reasons.add("image match")
        }

        // Token match with prefix support
        queryTokens.forEach { qt ->
            if (nameLower.contains(qt)) {
                score += 0.4f / queryTokens.size
                reasons.add("name: $qt")
            } else if (contentLower.contains(qt)) {
                score += 0.2f / queryTokens.size
                reasons.add("content: $qt")
            }
        }

        // Date match (e.g., searching "2026" or "apr")
        if (descLower.contains("20") && nameLower.contains("20")) { score += 0.1f; reasons.add("year match") }

        return score.coerceIn(0f, 1f) to reasons.distinct().joinToString(", ")
    }

    suspend fun search(
        description: String,
        db: com.nomad.data.FileIndexDatabase?,
        llmInference: LlamaInference?,
        onProgress: (String) -> Unit
    ): List<FileMatch> = withContext(Dispatchers.Default) {
        if (db == null) return@withContext emptyList()
        onProgress("Searching index...")

        val tokens = description.lowercase().split(Regex("[^a-z0-9]")).filter { it.length >= 2 && it !in STOP_WORDS }
        if (tokens.isEmpty()) return@withContext emptyList()

        // Multi-stage retrieval
        val ftsQuery = tokens.joinToString(" OR ") { "$it*" }
        val candidates = db.dao().searchFts(ftsQuery)

        onProgress("Ranking ${candidates.size} items...")
        candidates.map { entity ->
            val file = File(entity.absolutePath)
            val (score, reason) = scoreFile(file, entity.extractedText, description)
            FileMatch(file, entity.mimeCategory, entity.extractedText, score, reason)
        }
        .filter { it.score > 0.1f }
        .sortedByDescending { it.score }
        .take(LLM_RERANK_LIMIT)
    }

    private val STOP_WORDS = setOf("find", "show", "get", "me", "the", "and", "for", "with", "from")
    
    fun mimeCategory(file: File): String = EXTENSION_MAP[file.extension.lowercase()] ?: "other"
}
