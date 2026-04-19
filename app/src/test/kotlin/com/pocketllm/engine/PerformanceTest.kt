package com.pocketllm.engine

import org.junit.Test
import kotlin.test.assertTrue

class PerformanceTest {

    @Test
    fun testPromptBuildingPerformance() {
        val system = "You are a helpful assistant."
        val history = listOf(
            "Hello" to "Hi there!",
            "How are you?" to "I am a local AI running on your device, performing optimally.",
            "What can you do?" to "I can help you with coding, writing, and answering questions quickly."
        )
        val user = "Tell me a joke."

        val start = System.currentTimeMillis()
        val prompt = LlamaEngine.buildChatPrompt(system, history, user)
        val end = System.currentTimeMillis()

        println("Built prompt in ${end - start}ms")
        
        // Ensure prompt doesn't grow too large for fast initial processing
        assertTrue(prompt.length < 2000, "Prompt is too large for fast response")
    }

    @Test
    fun testModelMemoryParsing() {
        val lightModel = com.pocketllm.data.ModelCatalog.models.find { it.id == "tinyllama-q4" }
        val heavyModel = com.pocketllm.data.ModelCatalog.models.find { it.id == "llama3.1-8b-q4" }

        assertTrue(lightModel!!.fileSizeBytes < 1_000_000_000, "TinyLlama size calculation error")
        assertTrue(heavyModel!!.fileSizeBytes > 4_000_000_000, "Llama 8B size calculation error")
    }
}
