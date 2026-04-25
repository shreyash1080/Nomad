package com.nomad.data

import android.content.Context
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log

/**
 * FIXED: Added `path` field so file results can show the actual location.
 */
data class FileMetadata(
    val name: String,
    val path: String,       // Human-readable location, e.g. "DCIM/Camera/" or "/sdcard/Downloads/resume.pdf"
    val size: Long,
    val dateAdded: Long,
    val mimeType: String?
)

enum class FileSearchCategory { ALL, IMAGE, DOCUMENT, AUDIO, VIDEO }

data class FileSearchIntent(
    val searchTerm: String,
    val category: FileSearchCategory,
    val recentOnly: Boolean
)

object LocalFileHelper {
    private const val TAG = "Nomad_FileHelper"

    /**
     * FIXED: Removed "search" from finder verbs.
     * "search" is now exclusively handled by WebSearchHelper.
     * File search only triggers on "find", "show", "list", "locate", "where is",
     * "recent", "latest", "newest" — these are clearly file-oriented verbs.
     */
    fun parseIntent(rawQuery: String): FileSearchIntent? {
        val query = rawQuery.lowercase().trim()
        if (query.isBlank()) return null

        val recentOnly = listOf("recent", "latest", "newest", "last").any(query::contains)

        // IMPORTANT: "search" is intentionally NOT in this list — it belongs to web search.
        val hasFinderVerb = listOf(
            "find", "show me", "show my", "list", "look for",
            "locate", "where is", "where are", "recent", "latest", "newest"
        ).any { query.contains(it) }

        if (!hasFinderVerb) return null

        val category = when {
            listOf("image", "photo", "picture", "screenshot", "selfie").any(query::contains) -> FileSearchCategory.IMAGE
            listOf("pdf", "doc", "word", "text", "excel", "ppt", "spreadsheet", "document").any(query::contains) -> FileSearchCategory.DOCUMENT
            listOf("audio", "song", "music", "mp3").any(query::contains) -> FileSearchCategory.AUDIO
            listOf("video", "movie", "mp4", "clip").any(query::contains) -> FileSearchCategory.VIDEO
            else -> FileSearchCategory.ALL
        }

        // Strip only navigation/article words, keep file-relevant terms (extensions, names, etc.)
        val searchTerm = query
            .replace(
                Regex("\\b(find|show me|show my|list|look for|locate|where is|where are|me|my|the|a|an)\\b"),
                " "
            )
            .replace(Regex("\\s+"), " ")
            .trim()

        return FileSearchIntent(searchTerm, category, recentOnly)
    }

    // ── Recent Files ────────────────────────────────────────────────────────

    fun getRecentFiles(
        context: Context,
        category: FileSearchCategory = FileSearchCategory.ALL,
        limit: Int = 10
    ): List<FileMetadata> {
        val buckets = bucketsFor(category)
        return buckets
            .flatMap { fetchRecent(context, it, limit) }
            .sortedByDescending { it.dateAdded }
            .take(limit)
    }

    private fun fetchRecent(context: Context, target: QueryTarget, limit: Int): List<FileMetadata> {
        val results = mutableListOf<FileMetadata>()
        val projection = buildProjection()
        val sortOrder = "${MediaStore.MediaColumns.DATE_ADDED} DESC"

        try {
            context.contentResolver.query(
                target.uri(), projection,
                target.mimeSelection,
                target.mimeArgs.ifEmpty { null }?.toTypedArray(),
                sortOrder
            )?.use { cursor ->
                val nameIdx = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.DISPLAY_NAME)
                val sizeIdx = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.SIZE)
                val dateIdx = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.DATE_ADDED)
                val mimeIdx = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.MIME_TYPE)
                val pathIdx = getPathColumnIndex(cursor)

                while (cursor.moveToNext() && results.size < limit) {
                    results += FileMetadata(
                        name = cursor.getString(nameIdx) ?: "Unnamed",
                        path = resolveDisplayPath(cursor, pathIdx, cursor.getString(nameIdx) ?: ""),
                        size = cursor.getLong(sizeIdx),
                        dateAdded = cursor.getLong(dateIdx),
                        mimeType = cursor.getString(mimeIdx)
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Recent query error for ${target.label}: ${e.message}")
        }
        return results
    }

    // ── Search Files ────────────────────────────────────────────────────────

    fun searchFiles(
        context: Context,
        query: String,
        category: FileSearchCategory = FileSearchCategory.ALL,
        limit: Int = 10
    ): List<FileMetadata> {
        return loadMatches(context, query.trim(), category, limit)
    }

    private fun loadMatches(
        context: Context,
        query: String,
        category: FileSearchCategory,
        limit: Int
    ): List<FileMetadata> {
        val fetchLimit = limit * 6
        return bucketsFor(category)
            .flatMap { queryTarget(context, it, query, fetchLimit) }
            .distinctBy { "${it.name.lowercase()}|${it.path}" }
            .sortedByDescending { score(it, query) }
            .take(limit)
    }

    private fun queryTarget(
        context: Context,
        target: QueryTarget,
        query: String,
        limit: Int
    ): List<FileMetadata> {
        val results = mutableListOf<FileMetadata>()
        val projection = buildProjection()

        val selectionParts = mutableListOf<String>()
        val selectionArgs = mutableListOf<String>()

        val tokens = tokenize(query)

        // Exact filename match takes priority
        if (query.isNotBlank()) {
            selectionParts += "${MediaStore.MediaColumns.DISPLAY_NAME} LIKE ?"
            selectionArgs += "%$query%"
        }

        // Token OR search as a secondary condition
        if (tokens.isNotEmpty()) {
            val tokenConditions = tokens.map { "${MediaStore.MediaColumns.DISPLAY_NAME} LIKE ?" }
            selectionParts += "(${tokenConditions.joinToString(" OR ")})"
            tokens.forEach { selectionArgs += "%$it%" }
        }

        // Append any MIME type restriction from the target
        target.mimeSelection?.let { selectionParts += it }
        target.mimeArgs.forEach { selectionArgs += it }

        // Combine with OR (match any condition) so results are broader
        val selection = if (selectionParts.isEmpty()) null
        else selectionParts.joinToString(" OR ")

        val sortOrder = "${MediaStore.MediaColumns.DATE_ADDED} DESC"

        try {
            context.contentResolver.query(
                target.uri(), projection,
                selection,
                selectionArgs.toTypedArray(),
                sortOrder
            )?.use { cursor ->
                val nameIdx = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.DISPLAY_NAME)
                val sizeIdx = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.SIZE)
                val dateIdx = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.DATE_ADDED)
                val mimeIdx = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.MIME_TYPE)
                val pathIdx = getPathColumnIndex(cursor)

                while (cursor.moveToNext() && results.size < limit) {
                    results += FileMetadata(
                        name = cursor.getString(nameIdx) ?: "Unnamed file",
                        path = resolveDisplayPath(cursor, pathIdx, cursor.getString(nameIdx) ?: ""),
                        size = cursor.getLong(sizeIdx),
                        dateAdded = cursor.getLong(dateIdx),
                        mimeType = cursor.getString(mimeIdx)
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Query error for ${target.label}: ${e.message}")
        }
        return results
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private fun buildProjection(): Array<String> {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            arrayOf(
                MediaStore.MediaColumns.DISPLAY_NAME,
                MediaStore.MediaColumns.RELATIVE_PATH,   // e.g. "DCIM/Camera/"
                MediaStore.MediaColumns.SIZE,
                MediaStore.MediaColumns.DATE_ADDED,
                MediaStore.MediaColumns.MIME_TYPE
            )
        } else {
            arrayOf(
                MediaStore.MediaColumns.DISPLAY_NAME,
                MediaStore.MediaColumns.DATA,            // full path on older Android
                MediaStore.MediaColumns.SIZE,
                MediaStore.MediaColumns.DATE_ADDED,
                MediaStore.MediaColumns.MIME_TYPE
            )
        }
    }

    private fun getPathColumnIndex(cursor: android.database.Cursor): Int {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            cursor.getColumnIndex(MediaStore.MediaColumns.RELATIVE_PATH)
        } else {
            cursor.getColumnIndex(MediaStore.MediaColumns.DATA)
        }
    }

    /**
     * Returns a human-readable path string for display:
     * - Android 10+: "DCIM/Camera/" (RELATIVE_PATH)
     * - Android <10: "/storage/emulated/0/Downloads/resume.pdf" (DATA)
     */
    private fun resolveDisplayPath(
        cursor: android.database.Cursor,
        pathIdx: Int,
        name: String
    ): String {
        if (pathIdx < 0) return name
        val raw = cursor.getString(pathIdx)?.trim() ?: return name
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // RELATIVE_PATH ends with "/", e.g. "DCIM/Camera/"
            if (raw.isNotBlank()) "$raw$name" else name
        } else {
            // DATA is the full absolute path
            raw.ifBlank { name }
        }
    }

    private fun bucketsFor(category: FileSearchCategory): List<QueryTarget> = when (category) {
        FileSearchCategory.IMAGE    -> listOf(QueryTarget.Images)
        FileSearchCategory.DOCUMENT -> listOf(QueryTarget.Downloads)
        FileSearchCategory.AUDIO    -> listOf(QueryTarget.Audio)
        FileSearchCategory.VIDEO    -> listOf(QueryTarget.Video)
        FileSearchCategory.ALL      -> listOf(
            QueryTarget.Images,
            QueryTarget.Downloads,
            QueryTarget.Video,
            QueryTarget.Audio
        )
    }

    private fun score(file: FileMetadata, query: String): Int {
        val name = file.name.lowercase()
        val q = query.lowercase()
        var s = 0
        if (name == q) s += 100
        if (name.startsWith(q)) s += 75
        if (name.contains(q)) s += 50
        tokenize(q).forEach { if (name.contains(it)) s += 10 }
        return s
    }

    private fun tokenize(text: String): List<String> =
        text.split(Regex("[\\s_\\-\\.]+")).filter { it.length >= 2 }

    // ── Query Targets ────────────────────────────────────────────────────────

    private sealed class QueryTarget(
        val label: String,
        val mimeSelection: String? = null,
        val mimeArgs: List<String> = emptyList()
    ) {
        abstract fun uri(): Uri

        data object Images : QueryTarget("images") {
            override fun uri(): Uri =
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                    MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL)
                else MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        }

        data object Video : QueryTarget("videos") {
            override fun uri(): Uri =
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                    MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL)
                else MediaStore.Video.Media.EXTERNAL_CONTENT_URI
        }

        data object Audio : QueryTarget("audio") {
            override fun uri(): Uri =
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                    MediaStore.Audio.Media.getContentUri(MediaStore.VOLUME_EXTERNAL)
                else MediaStore.Audio.Media.EXTERNAL_CONTENT_URI
        }

        data object Downloads : QueryTarget(
            label = "downloads",
            mimeSelection = "${MediaStore.MediaColumns.MIME_TYPE} LIKE ? OR ${MediaStore.MediaColumns.MIME_TYPE} LIKE ?",
            mimeArgs = listOf("application/%", "text/%")
        ) {
            override fun uri(): Uri =
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                    MediaStore.Downloads.getContentUri(MediaStore.VOLUME_EXTERNAL)
                else MediaStore.Files.getContentUri("external")
        }
    }
}