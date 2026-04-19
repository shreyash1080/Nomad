package com.pocketllm.data

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.InputStream

/** What kind of file was attached */
enum class AttachmentType { IMAGE, PDF, TEXT, EXCEL, WORD, UNKNOWN }

data class Attachment(
    val uri: Uri,
    val fileName: String,
    val type: AttachmentType,
    /** Extracted text for text/pdf/excel/word; null for images */
    val extractedText: String? = null,
    /** Base64 thumbnail for display */
    val thumbnailBase64: String? = null,
    val fileSizeBytes: Long = 0L
)

object FileProcessor {

    suspend fun process(context: Context, uri: Uri): Attachment = withContext(Dispatchers.IO) {
        val fileName = getFileName(context, uri) ?: "file"
        val size     = getFileSize(context, uri)
        val ext      = fileName.substringAfterLast('.', "").lowercase()
        val type     = when (ext) {
            "jpg", "jpeg", "png", "webp", "gif", "bmp", "heic" -> AttachmentType.IMAGE
            "pdf"                                               -> AttachmentType.PDF
            "txt", "md", "csv", "json", "xml", "html", "kt",
            "py", "js", "ts", "java", "cpp", "c", "h"         -> AttachmentType.TEXT
            "xls", "xlsx", "ods"                               -> AttachmentType.EXCEL
            "doc", "docx", "odt"                               -> AttachmentType.WORD
            else                                                -> AttachmentType.UNKNOWN
        }

        when (type) {
            AttachmentType.IMAGE -> processImage(context, uri, fileName, size)
            AttachmentType.PDF   -> processPdf(context, uri, fileName, size)
            AttachmentType.TEXT  -> processText(context, uri, fileName, size, type)
            AttachmentType.EXCEL -> processExcel(context, uri, fileName, size)
            AttachmentType.WORD  -> processWord(context, uri, fileName, size)
            else                 -> processText(context, uri, fileName, size, type)
        }
    }

    // ── IMAGE ─────────────────────────────────────────────────────────────────
    private suspend fun processImage(context: Context, uri: Uri, name: String, size: Long): Attachment = withContext(Dispatchers.IO) {
        try {
            val options = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            context.contentResolver.openInputStream(uri)?.use { 
                BitmapFactory.decodeStream(it, null, options) 
            }

            // Downsample if image is huge (keep it under ~2048px for thumb/desc)
            val maxDim = 1200
            var inSampleSize = 1
            if (options.outHeight > maxDim || options.outWidth > maxDim) {
                val halfHeight = options.outHeight / 2
                val halfWidth = options.outWidth / 2
                while (halfHeight / inSampleSize >= maxDim && halfWidth / inSampleSize >= maxDim) {
                    inSampleSize *= 2
                }
            }

            val decodeOptions = BitmapFactory.Options().apply {
                this.inSampleSize = inSampleSize
            }
            val bitmap = context.contentResolver.openInputStream(uri)?.use {
                BitmapFactory.decodeStream(it, null, decodeOptions)
            } ?: return@withContext unknown(uri, name, size)

            // Make a small thumbnail for the chat bubble
            val thumb = makeThumbnail(bitmap, 150)
            val thumbB64 = bitmapToBase64(thumb)

            // Extract text from image using OCR
            val extractedText = extractTextFromImage(context, uri)
            val desc = if (extractedText.isNotBlank()) {
                "Text extracted from image '$name':\n$extractedText"
            } else {
                "User attached an image: '$name' (${options.outWidth}×${options.outHeight} px). No text detected."
            }
            
            Attachment(uri, name, AttachmentType.IMAGE, desc, thumbB64, size)
        } catch (e: Exception) {
            Attachment(uri, name, AttachmentType.IMAGE, "Error processing image: ${e.message}", null, size)
        }
    }

    private suspend fun extractTextFromImage(context: Context, uri: Uri): String = withContext(Dispatchers.IO) {
        try {
            val image = com.google.mlkit.vision.common.InputImage.fromFilePath(context, uri)
            val recognizer = com.google.mlkit.vision.text.TextRecognition.getClient(com.google.mlkit.vision.text.latin.TextRecognizerOptions.DEFAULT_OPTIONS)
            val result = com.google.android.gms.tasks.Tasks.await(recognizer.process(image))
            result.text
        } catch (e: Exception) {
            ""
        }
    }

    // ── PDF ───────────────────────────────────────────────────────────────────
    private suspend fun processPdf(context: Context, uri: Uri, name: String, size: Long): Attachment = withContext(Dispatchers.IO) {
        try {
            // Read first 2MB max to avoid OOM
            val limit = 2 * 1024 * 1024
            val buffer = ByteArray(minOf(size.toInt().coerceAtLeast(1), limit))
            context.contentResolver.openInputStream(uri)?.use { input ->
                var totalRead = 0
                while (totalRead < buffer.size) {
                    val read = input.read(buffer, totalRead, buffer.size - totalRead)
                    if (read == -1) break
                    totalRead += read
                }
            }
            val text = extractPdfText(buffer)
            Attachment(uri, name, AttachmentType.PDF, text, null, size)
        } catch (e: Exception) {
            Attachment(uri, name, AttachmentType.PDF,
                "PDF file: '$name' (${formatSize(size)}). Could not extract text: ${e.message}", null, size)
        }
    }

    // ── PLAIN TEXT / CODE / CSV / JSON ────────────────────────────────────────
    private suspend fun processText(context: Context, uri: Uri, name: String, size: Long, type: AttachmentType): Attachment = withContext(Dispatchers.IO) {
        try {
            val text = context.contentResolver.openInputStream(uri)?.use { input ->
                input.bufferedReader().use { reader ->
                    // Read first 10k chars max
                    val buffer = CharArray(10000)
                    val read = reader.read(buffer)
                    if (read > 0) String(buffer, 0, read) else ""
                }
            } ?: ""
            
            val trimmed = if (text.length >= 10000) text + "\n[... truncated]" else text
            Attachment(uri, name, type, "File '$name':\n$trimmed", null, size)
        } catch (e: Exception) {
            Attachment(uri, name, type, "Could not read file: ${e.message}", null, size)
        }
    }

    // ── EXCEL (basic — reads as CSV-like text) ────────────────────────────────
    private suspend fun processExcel(context: Context, uri: Uri, name: String, size: Long): Attachment = withContext(Dispatchers.IO) {
        try {
            val limit = 1 * 1024 * 1024 // 1MB
            val buffer = ByteArray(minOf(size.toInt().coerceAtLeast(1), limit))
            context.contentResolver.openInputStream(uri)?.use { input ->
                input.read(buffer)
            }
            val text = extractStringsFromBinary(buffer)
            Attachment(uri, name, AttachmentType.EXCEL,
                "Excel file '$name' (${formatSize(size)}). Extracted content:\n$text", null, size)
        } catch (e: Exception) {
            Attachment(uri, name, AttachmentType.EXCEL,
                "Excel file: '$name' (${formatSize(size)}). Content could not be read.", null, size)
        }
    }

    // ── WORD DOC ──────────────────────────────────────────────────────────────
    private suspend fun processWord(context: Context, uri: Uri, name: String, size: Long): Attachment = withContext(Dispatchers.IO) {
        try {
            val limit = 1 * 1024 * 1024 // 1MB
            val buffer = ByteArray(minOf(size.toInt().coerceAtLeast(1), limit))
            context.contentResolver.openInputStream(uri)?.use { input ->
                input.read(buffer)
            }
            val text = extractStringsFromBinary(buffer)
            Attachment(uri, name, AttachmentType.WORD,
                "Word document '$name' (${formatSize(size)}). Content:\n$text", null, size)
        } catch (e: Exception) {
            Attachment(uri, name, AttachmentType.WORD,
                "Word document: '$name'. Could not extract text.", null, size)
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /** Very basic PDF text extractor — finds BT...ET blocks and (text) strings */
    private fun extractPdfText(bytes: ByteArray): String {
        val raw    = String(bytes, Charsets.ISO_8859_1)
        val result = StringBuilder()

        // Extract text between BT and ET markers (Basic PDF text extraction)
        var i = 0
        while (i < raw.length) {
            val btIdx = raw.indexOf("BT", i)
            if (btIdx < 0) break
            val etIdx = raw.indexOf("ET", btIdx)
            if (etIdx < 0) break
            val block = raw.substring(btIdx, etIdx)
            // Find (text) strings in this block
            var j = 0
            while (j < block.length) {
                val start = block.indexOf('(', j)
                if (start < 0) break
                val end = findCloseParen(block, start)
                if (end < 0) break
                val piece = block.substring(start + 1, end)
                    .replace("\\n", "\n")
                    .replace("\\r", "")
                    .replace("\\(", "(")
                    .replace("\\)", ")")
                if (piece.isNotBlank() && piece.all { it.code in 32..126 || it == '\n' })
                    result.append(piece).append(" ")
                j = end + 1
            }
            i = etIdx + 2
        }

        val text = result.toString().trim()
        return if (text.isBlank()) "PDF file — text could not be extracted (possibly scanned/image-based)."
        else text.take(6000)
    }

    private fun findCloseParen(s: String, open: Int): Int {
        var i = open + 1
        while (i < s.length) {
            when {
                s[i] == '\\' -> i += 2
                s[i] == ')'  -> return i
                else         -> i++
            }
        }
        return -1
    }

    /** Extract readable ASCII strings from binary (works for docx/xlsx XML inside ZIP) */
    private fun extractStringsFromBinary(bytes: ByteArray): String {
        // docx/xlsx are ZIP files — scan for XML text content
        val sb  = StringBuilder()
        val raw = String(bytes, Charsets.ISO_8859_1)

        // Find XML text content between > and <
        var i = 0
        while (i < raw.length && sb.length < 5000) {
            val gt = raw.indexOf('>', i)
            if (gt < 0) break
            val lt = raw.indexOf('<', gt)
            if (lt < 0) break
            val piece = raw.substring(gt + 1, lt).trim()
            if (piece.length > 1 && piece.all { it.code in 32..126 })
                sb.append(piece).append(" ")
            i = lt + 1
        }
        return sb.toString().trim().take(5000).ifBlank { "Could not extract readable text." }
    }

    private fun makeThumbnail(bmp: Bitmap, size: Int): Bitmap {
        val scale  = size.toFloat() / maxOf(bmp.width, bmp.height)
        val w      = (bmp.width  * scale).toInt().coerceAtLeast(1)
        val h      = (bmp.height * scale).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(bmp, w, h, true)
    }

    private fun bitmapToBase64(bmp: Bitmap): String {
        val out = ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.JPEG, 70, out)
        return Base64.encodeToString(out.toByteArray(), Base64.DEFAULT)
    }

    private fun getFileName(context: Context, uri: Uri): String? {
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (idx >= 0) return cursor.getString(idx)
            }
        }
        return uri.lastPathSegment
    }

    private fun getFileSize(context: Context, uri: Uri): Long {
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val idx = cursor.getColumnIndex(OpenableColumns.SIZE)
                if (idx >= 0) return cursor.getLong(idx)
            }
        }
        return 0L
    }

    private fun unknown(uri: Uri, name: String, size: Long) =
        Attachment(uri, name, AttachmentType.UNKNOWN, "Attached file: $name", null, size)

    fun formatSize(bytes: Long) = when {
        bytes < 1024        -> "$bytes B"
        bytes < 1_048_576   -> "${"%.1f".format(bytes / 1024.0)} KB"
        else                -> "${"%.1f".format(bytes / 1_048_576.0)} MB"
    }
}