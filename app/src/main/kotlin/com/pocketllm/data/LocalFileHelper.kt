package com.eigen.data

import android.content.Context
import android.provider.MediaStore
import android.util.Log

data class FileMetadata(
    val name: String,
    val size: Long,
    val dateAdded: Long,
    val mimeType: String?
)

object LocalFileHelper {
    private const val TAG = "Eigen_FileHelper"

    fun searchFiles(context: Context, query: String): List<FileMetadata> {
        val results = mutableListOf<FileMetadata>()
        val projection = arrayOf(
            MediaStore.Files.FileColumns.DISPLAY_NAME,
            MediaStore.Files.FileColumns.SIZE,
            MediaStore.Files.FileColumns.DATE_ADDED,
            MediaStore.Files.FileColumns.MIME_TYPE
        )

        val uri = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
            MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL)
        } else {
            MediaStore.Files.getContentUri("external")
        }
        val selection = "${MediaStore.Files.FileColumns.DISPLAY_NAME} LIKE ?"
        val selectionArgs = arrayOf("%$query%")
        val sortOrder = "${MediaStore.Files.FileColumns.DATE_ADDED} DESC"

        try {
            context.contentResolver.query(
                uri,
                projection,
                selection,
                selectionArgs,
                sortOrder
            )?.use { cursor ->
                val nameIdx = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DISPLAY_NAME)
                val sizeIdx = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.SIZE)
                val dateIdx = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DATE_ADDED)
                val mimeIdx = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.MIME_TYPE)

                while (cursor.moveToNext() && results.size < 10) {
                    results.add(
                        FileMetadata(
                            name = cursor.getString(nameIdx),
                            size = cursor.getLong(sizeIdx),
                            dateAdded = cursor.getLong(dateIdx),
                            mimeType = cursor.getString(mimeIdx)
                        )
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error searching files: ${e.message}")
        }

        return results
    }

    fun getRecentFiles(context: Context, limit: Int = 5): List<FileMetadata> {
        val results = mutableListOf<FileMetadata>()
        val projection = arrayOf(
            MediaStore.Files.FileColumns.DISPLAY_NAME,
            MediaStore.Files.FileColumns.SIZE,
            MediaStore.Files.FileColumns.DATE_ADDED,
            MediaStore.Files.FileColumns.MIME_TYPE
        )

        val uri = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
            MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL)
        } else {
            MediaStore.Files.getContentUri("external")
        }
        val sortOrder = "${MediaStore.Files.FileColumns.DATE_ADDED} DESC"

        try {
            context.contentResolver.query(
                uri,
                projection,
                null,
                null,
                sortOrder
            )?.use { cursor ->
                val nameIdx = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DISPLAY_NAME)
                val sizeIdx = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.SIZE)
                val dateIdx = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DATE_ADDED)
                val mimeIdx = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.MIME_TYPE)

                while (cursor.moveToNext() && results.size < limit) {
                    results.add(
                        FileMetadata(
                            name = cursor.getString(nameIdx),
                            size = cursor.getLong(sizeIdx),
                            dateAdded = cursor.getLong(dateIdx),
                            mimeType = cursor.getString(mimeIdx)
                        )
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting recent files: ${e.message}")
        }

        return results
    }
}
