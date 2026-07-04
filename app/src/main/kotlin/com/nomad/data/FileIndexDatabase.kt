package com.nomad.data

import android.content.Context
import androidx.room.*

@Entity(tableName = "file_index")
data class FileIndexEntity(
    @PrimaryKey val absolutePath: String,
    val fileName: String,
    val extension: String,
    val mimeCategory: String,       // "image" | "pdf" | "document" | "spreadsheet" | "text" | "other"
    val sizeBytes: Long,
    val lastModifiedMs: Long,
    val lastIndexedAtMs: Long,
    val extractedText: String,      // first 800 chars of content
    val keywords: String            // space-separated tokenized keywords
)

@Fts4
@Entity(tableName = "file_index_fts")
data class FileIndexFtsEntity(
    val absolutePath: String,
    val fileName: String,
    val extractedText: String,
    val keywords: String
)

@Dao
interface FileIndexDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(entity: FileIndexEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertFts(entity: FileIndexFtsEntity)

    @Query("DELETE FROM file_index WHERE absolutePath = :path")
    suspend fun delete(path: String)

    @Query("DELETE FROM file_index_fts WHERE absolutePath = :path")
    suspend fun deleteFts(path: String)

    @Transaction
    suspend fun upsertWithFts(entity: FileIndexEntity) {
        upsert(entity)
        upsertFts(FileIndexFtsEntity(
            absolutePath = entity.absolutePath,
            fileName = entity.fileName,
            extractedText = entity.extractedText,
            keywords = entity.keywords
        ))
    }

    @Query("""
        SELECT fi.* FROM file_index fi
        JOIN file_index_fts fts ON fi.absolutePath = fts.absolutePath
        WHERE file_index_fts MATCH :query
        LIMIT 100
    """)
    suspend fun searchFts(query: String): List<FileIndexEntity>

    @Query("SELECT absolutePath FROM file_index")
    suspend fun getAllPaths(): List<String>
    
    @Query("SELECT COUNT(*) FROM file_index")
    suspend fun count(): Int
}

@Database(entities = [FileIndexEntity::class, FileIndexFtsEntity::class], version = 2, exportSchema = false)
abstract class FileIndexDatabase : RoomDatabase() {
    abstract fun dao(): FileIndexDao
    companion object {
        @Volatile private var INSTANCE: FileIndexDatabase? = null
        fun getInstance(context: Context): FileIndexDatabase =
            INSTANCE ?: synchronized(this) {
                Room.databaseBuilder(context.applicationContext, FileIndexDatabase::class.java, "file_index.db")
                    .fallbackToDestructiveMigration()
                    .build().also { INSTANCE = it }
            }
    }
}
