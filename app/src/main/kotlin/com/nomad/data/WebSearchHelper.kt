package com.nomad.data

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.net.URLEncoder
import java.util.concurrent.TimeUnit

object WebSearchHelper {
    private const val TAG = "Nomad_WebSearch"

    private val client = OkHttpClient.Builder()
        .connectTimeout(12, TimeUnit.SECONDS)
        .readTimeout(18, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    /**
     * FIXED: Removed "search" overlap with file search.
     * Only explicit web-search intent words trigger this.
     * "find" is reserved exclusively for local file search.
     */
    fun isWebSearchIntent(rawQuery: String): Boolean {
        val query = rawQuery.lowercase().trim()

        // Explicit web search prefix triggers
        val explicitPrefixes = listOf(
            "search ", "google ", "web search ", "look up ", "lookup "
        )
        if (explicitPrefixes.any { query.startsWith(it) }) return true

        // Informational query triggers (question patterns)
        val questionTriggers = listOf(
            "who is ", "who was ", "who are ",
            "what is ", "what are ", "what was ",
            "when did ", "when was ", "when is ",
            "where is ", "where was ", "where are ",
            "how much is", "how many ", "how does ",
            "current price of", "price of ",
            "weather in ", "weather at ", "weather for ",
            "latest news", "news about ", "news on ",
            "stock price", "exchange rate",
            "definition of", "define ",
            "capital of ", "population of "
        )
        return questionTriggers.any { query.startsWith(it) || query.contains(it) }
    }

    fun cleanQuery(rawQuery: String): String {
        return rawQuery
            .replace(
                Regex(
                    "^(search|google|web search|web|look up|lookup)\\s+",
                    RegexOption.IGNORE_CASE
                ), ""
            )
            .trim()
    }

    /**
     * FIXED: Multi-level search strategy:
     * 1. DuckDuckGo Instant Answer JSON API (fast, structured)
     * 2. DuckDuckGo HTML scraping (broader coverage)
     * 3. Graceful failure message
     */
    suspend fun performSearch(query: String): String = withContext(Dispatchers.IO) {
        Log.d(TAG, "Searching for: $query")
        try {
            // Level 1: Instant Answer API (great for facts, definitions, calculations, market data)
            val instantResult = tryInstantAnswerApi(query)

            // Level 2: HTML scraping (covers general web results, top 5)
            val htmlResult = tryHtmlScrape(query)

            if (instantResult.isNullOrBlank() && htmlResult.isNullOrBlank()) {
                return@withContext "No clear results found for \"$query\". The query may be too specific or require real-time data."
            }

            val sb = StringBuilder()

            // Always include instant answer if available (contains specific data like Wikipedia, stock prices, etc.)
            if (!instantResult.isNullOrBlank()) {
                sb.append("### Instant Answer / Knowledge Card\n$instantResult\n\n")
            }

            // Include up to 5 web results for broader context
            if (!htmlResult.isNullOrBlank()) {
                sb.append("### Top Web Search Results\n$htmlResult")
            }

            sb.toString().trim()
        } catch (e: Exception) {
            Log.e(TAG, "Search error", e)
            "Web search failed: ${e.localizedMessage ?: "Network error. Check your internet connection."}"
        }
    }

    // ── Level 1: DuckDuckGo Instant Answer API ─────────────────────────────

    private fun tryInstantAnswerApi(query: String): String? {
        return try {
            val encodedQuery = URLEncoder.encode(query, "UTF-8")
            val url =
                "https://api.duckduckgo.com/?q=$encodedQuery&format=json&no_html=1&skip_disambig=1&no_redirect=1"
            val request = Request.Builder()
                .url(url)
                .header("User-Agent", "NomadAI/1.0 (Android local assistant)")
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return null
                val body = response.body?.string() ?: return null
                val json = runCatching { JSONObject(body) }.getOrNull() ?: return null

                // Priority 1 — Direct instant answer (calculations, unit conversions, etc.)
                val answer = json.optString("Answer").trim()
                if (answer.isNotBlank()) return "Answer: $answer"

                // Priority 2 — Wikipedia-style abstract
                val abstractText = json.optString("AbstractText").trim()
                val abstractSource = json.optString("AbstractSource").trim()
                if (abstractText.isNotBlank()) {
                    return if (abstractSource.isNotBlank()) {
                        "[$abstractSource] $abstractText"
                    } else abstractText
                }

                // Priority 3 — Dictionary definition
                val definition = json.optString("Definition").trim()
                val defSource = json.optString("DefinitionSource").trim()
                if (definition.isNotBlank()) {
                    return if (defSource.isNotBlank()) "[$defSource] $definition" else definition
                }

                // Priority 4 — Infobox structured data (people, places, companies)
                val infoboxResult = parseInfobox(json)
                if (!infoboxResult.isNullOrBlank()) return infoboxResult

                // Priority 5 — Related topic summaries (at least 3 meaningful ones)
                val relatedResult = parseRelatedTopics(json)
                if (!relatedResult.isNullOrBlank()) return relatedResult

                null
            }
        } catch (e: Exception) {
            Log.w(TAG, "Instant answer API failed: ${e.message}")
            null
        }
    }

    private fun parseInfobox(json: JSONObject): String? {
        val infobox = json.optJSONObject("Infobox") ?: return null
        val content = infobox.optJSONArray("content") ?: return null
        if (content.length() == 0) return null

        val sb = StringBuilder()
        // Also grab the entity name from the main JSON
        val entity = json.optString("Heading").trim()
        if (entity.isNotBlank()) sb.append("$entity\n")

        for (i in 0 until minOf(content.length(), 6)) {
            val entry = content.optJSONObject(i) ?: continue
            val label = entry.optString("label").trim()
            val value = entry.optString("value").trim()
            if (label.isNotBlank() && value.isNotBlank()) {
                sb.append("$label: $value\n")
            }
        }
        return if (sb.length > entity.length + 1) sb.toString().trim() else null
    }

    private fun parseRelatedTopics(json: JSONObject): String? {
        val related = json.optJSONArray("RelatedTopics") ?: return null
        if (related.length() == 0) return null

        val sb = StringBuilder()
        var count = 0
        for (i in 0 until related.length()) {
            val obj = related.optJSONObject(i) ?: continue
            val text = obj.optString("Text").trim()
            if (text.isNotBlank() && text.length > 15) {
                sb.append("• $text\n")
                count++
            }
            if (count >= 4) break
        }
        return if (count >= 2) sb.toString().trim() else null
    }

    // ── Level 2: DuckDuckGo HTML scraping ──────────────────────────────────

    private fun tryHtmlScrape(query: String): String? {
        return try {
            val encodedQuery = URLEncoder.encode(query, "UTF-8")
            val url = "https://html.duckduckgo.com/html/?q=$encodedQuery&kl=us-en"
            val request = Request.Builder()
                .url(url)
                .header(
                    "User-Agent",
                    "Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36 " +
                            "(KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36"
                )
                .header("Accept", "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8")
                .header("Accept-Language", "en-US,en;q=0.9")
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.w(TAG, "HTML scrape HTTP ${response.code}")
                    return null
                }
                val body = response.body?.string() ?: return null
                parseHtmlResults(body, query)
            }
        } catch (e: Exception) {
            Log.w(TAG, "HTML scrape failed: ${e.message}")
            null
        }
    }

    private fun parseHtmlResults(html: String, originalQuery: String): String? {
        val titles = mutableListOf<String>()
        val snippets = mutableListOf<String>()

        // Extract result titles
        val titleRegex = Regex(
            """class=["']result__a["'][^>]*>\s*(.+?)\s*</a>""",
            setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE)
        )
        titleRegex.findAll(html).take(5).forEach { match ->
            val title = match.groupValues[1].stripHtml().htmlDecode().trim()
            if (title.isNotBlank()) titles.add(title.take(80))
        }

        // Extract result snippets
        val snippetRegex = Regex(
            """class=["']result__snippet["'][^>]*>(.+?)</a>""",
            setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE)
        )
        snippetRegex.findAll(html).take(5).forEach { match ->
            val snippet = match.groupValues[1].stripHtml().htmlDecode().trim()
            if (snippet.isNotBlank() && snippet.length > 20) {
                // Shorten significantly to speed up inference (max 150 chars)
                val shortSnippet = if (snippet.length > 150) snippet.take(147) + "..." else snippet
                snippets.add(shortSnippet)
            }
        }

        // Fallback for snippets if the first regex misses
        if (snippets.isEmpty()) {
            val altSnippetRegex = Regex(
                """class=["']result-snippet["'][^>]*>(.+?)</(?:td|div|span|a)>""",
                setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.IGNORE_CASE)
            )
            altSnippetRegex.findAll(html).take(5).forEach { match ->
                val snippet = match.groupValues[1].stripHtml().htmlDecode().trim()
                if (snippet.isNotBlank() && snippet.length > 20) {
                    val shortSnippet = if (snippet.length > 150) snippet.take(147) + "..." else snippet
                    snippets.add(shortSnippet)
                }
            }
        }

        if (snippets.isEmpty() && titles.isEmpty()) {
            Log.w(TAG, "HTML parse found no results for: $originalQuery")
            return null
        }

        val sb = StringBuilder()
        // Ultra-compact format to save tokens and speed up generation
        val resultCount = minOf(titles.size, snippets.size, 5)
        for (i in 0 until resultCount) {
            sb.append("[${i + 1}] ${titles[i]}: ${snippets[i]}\n")
        }

        return sb.toString().trim().ifBlank { null }
    }

    // ── HTML utilities ──────────────────────────────────────────────────────

    private fun String.stripHtml(): String =
        this.replace(Regex("<[^>]*>"), " ").replace(Regex("\\s{2,}"), " ")

    private fun String.htmlDecode(): String = this
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
        .replace("&#39;", "'")
        .replace("&mdash;", "—")
        .replace("&ndash;", "–")
}