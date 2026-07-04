package com.nomad.engine

/**
 * Minimal contract that FileSearchEngine needs to do LLM re-ranking.
 */
interface LlamaInference {
    /**
     * Run inference on [prompt] and call [onToken] for every generated token.
     * The call is expected to be a *suspend* function so it can be cancelled.
     */
    suspend fun runInference(prompt: String, onToken: (String) -> Unit)
}

/**
 * Wrap an existing inference lambda into the [LlamaInference] interface.
 */
class LlamaInferenceAdapter(
    private val inferFn: suspend (prompt: String, onToken: (String) -> Unit) -> Unit
) : LlamaInference {
    override suspend fun runInference(prompt: String, onToken: (String) -> Unit) =
        inferFn(prompt, onToken)
}
