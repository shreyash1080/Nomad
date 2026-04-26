package com.nomad.utils

import android.content.Context
import java.io.File
import java.io.FileOutputStream

object AssetHelper {
    /**
     * Recursively copies assets from a folder to a target directory on disk.
     */
    fun copyAssets(context: Context, assetPath: String, targetDir: File) {
        val assets = context.assets.list(assetPath) ?: return
        if (assets.isEmpty()) {
            // It's a file
            copyFile(context, assetPath, File(targetDir, File(assetPath).name))
        } else {
            // It's a directory
            val newTargetDir = File(targetDir, assetPath.substringAfterLast("/"))
            if (!newTargetDir.exists()) newTargetDir.mkdirs()
            for (asset in assets) {
                copyAssets(context, "$assetPath/$asset", targetDir)
            }
        }
    }

    private fun copyFile(context: Context, assetPath: String, targetFile: File) {
        try {
            context.assets.open(assetPath).use { input ->
                FileOutputStream(targetFile).use { output ->
                    input.copyTo(output)
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    
    /**
     * Specific helper to extract the MobileAirLLM environment
     */
    fun extractMobileLLM(context: Context): File {
        val targetDir = File(context.filesDir, "mobilellm_env")
        if (!targetDir.exists()) targetDir.mkdirs()
        
        // Simplified extraction for the flat structure we created in assets
        copyFolder(context, "mobilellm", targetDir)
        return targetDir
    }

    private fun copyFolder(context: Context, assetPath: String, targetDir: File) {
        val assets = context.assets.list(assetPath) ?: return
        if (assets.isEmpty()) return

        for (asset in assets) {
            val fullAssetPath = "$assetPath/$asset"
            val subAssets = context.assets.list(fullAssetPath)
            
            if (subAssets.isNullOrEmpty()) {
                // File
                val destFile = File(targetDir, asset)
                copyFile(context, fullAssetPath, destFile)
            } else {
                // Directory
                val subDir = File(targetDir, asset)
                subDir.mkdirs()
                copyFolder(context, fullAssetPath, subDir)
            }
        }
    }
}
