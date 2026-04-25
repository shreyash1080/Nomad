//  llama_wrapper.cpp — Production Ready + Maximum Performance
//  Refactored for stability and Gemma 2 support

#include <jni.h>
#include <android/log.h>
#include <unistd.h>
#include <sched.h>
#include <string>
#include <vector>
#include <atomic>
#include <cstring>
#include <cstdio>
#include <ctime>

#include "llama.h"
#include "ggml.h"

#define LOG_TAG "Nomad_Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ── Global state ──────────────────────────────────────────────────────────────
static llama_model*      g_model           = nullptr;
static llama_context*    g_ctx             = nullptr;
static std::atomic<bool> g_stop{false};
static int               g_n_ctx           = 0;
static int               g_n_threads       = 4;
static int               g_n_threads_batch = 4;
static std::string       g_utf8_cache;

// ── KV Prefix cache ──────────────────────────────────────────────────────────
static int                      g_prefix_n_tokens = 0;
static std::vector<llama_token> g_prefix_tokens;

// ── P-core CPU affinity ───────────────────────────────────────────────────────
static cpu_set_t g_p_core_mask;
static bool      g_mask_valid = false;

// ── Helpers ───────────────────────────────────────────────────────────────────
static std::string j2s(JNIEnv* env, jstring s) {
    if (!s) return "";
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string r(c);
    env->ReleaseStringUTFChars(s, c);
    return r;
}

static bool is_valid_utf8(const std::string& text) {
    auto* bytes = reinterpret_cast<const unsigned char*>(text.c_str());
    int num = 0;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }

    return true;
}

static std::vector<llama_token> tokenize_str(const std::string& text, bool add_special) {
    if (!g_model) return {};
    int n = -llama_tokenize(g_model, text.c_str(), (int32_t)text.size(),
                            nullptr, 0, add_special, true);
    if (n <= 0) return {};
    std::vector<llama_token> toks(n);
    int rc = llama_tokenize(g_model, text.c_str(), (int32_t)text.size(),
                            toks.data(), n, add_special, true);
    if (rc < 0) return {};
    toks.resize(rc);
    return toks;
}

static int build_perf_core_mask(cpu_set_t* mask) {
    CPU_ZERO(mask);
    int total = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (total <= 0) total = 4;
    std::vector<unsigned long> freqs(total, 0UL);
    unsigned long max_freq = 0;
    char path[128];
    for (int i = 0; i < total; i++) {
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        char line[32];
        if (fgets(line, sizeof(line), f)) {
            freqs[i] = strtoul(line, nullptr, 10);
            if (freqs[i] > max_freq) max_freq = freqs[i];
        }
        fclose(f);
    }
    int count = 0;
    if (max_freq == 0) {
        for (int i = total / 2; i < total; i++) { CPU_SET(i, mask); count++; }
    } else {
        for (int i = 0; i < total; i++) {
            if (freqs[i] >= max_freq * 8 / 10) { CPU_SET(i, mask); count++; }
        }
    }
    return count;
}

static void pin_to_p_cores() {
    if (g_mask_valid) sched_setaffinity(0, sizeof(cpu_set_t), &g_p_core_mask);
}

// ── Stop sequences ────────────────────────────────────────────────────────────
static const std::vector<std::string> STOP_SEQS = {
    "<|im_end|>",        // ChatML
    "<|eot_id|>",        // Llama 3
    "<end_of_turn>",     // Gemma 2
    "<start_of_turn>",   // Gemma 2 hallu guard
    "<|end|>",           // Phi-3
    "</s>",              // Mistral
    "[/INST]",           // Llama 2
    "User:",             // General hallu guard
    "\nUser:",
    "\nHuman:",
};

static std::string strip_stops(std::string text) {
    for (const auto& s : STOP_SEQS) {
        size_t pos = text.find(s);
        if (pos != std::string::npos) text = text.substr(0, pos);
    }
    size_t a = text.find_first_not_of(" \t\n\r");
    size_t b = text.find_last_not_of(" \t\n\r");
    return (a == std::string::npos) ? "" : text.substr(a, b - a + 1);
}

// ─────────────────────────────────────────────────────────────────────────────
extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_nomad_engine_LlamaEngine_loadModel(
        JNIEnv* env, jclass,
        jstring modelPath, jint nCtx, jint nThreads, jboolean useGpu)
{
    if (g_ctx)   { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_free_model(g_model); g_model = nullptr; }
    g_prefix_n_tokens = 0;
    g_prefix_tokens.clear();

    llama_backend_init();

    int p_count = build_perf_core_mask(&g_p_core_mask);
    g_mask_valid = (p_count > 0);
    pin_to_p_cores();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = useGpu ? 99 : 0;
    mp.use_mmap     = true;

    std::string path = j2s(env, modelPath);
    g_model = llama_load_model_from_file(path.c_str(), mp);
    if (!g_model) return JNI_FALSE;

    g_n_threads = (nThreads > 0) ? nThreads : p_count;
    g_n_threads = std::min(std::max(g_n_threads, 2), 8);
    g_n_threads_batch = g_n_threads;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t)nCtx;
    cp.n_threads       = (int32_t)g_n_threads;
    cp.n_threads_batch = (int32_t)g_n_threads_batch;
    cp.flash_attn      = true;
    cp.offload_kqv     = useGpu;

    g_ctx = llama_new_context_with_model(g_model, cp);
    if (!g_ctx) { llama_free_model(g_model); g_model = nullptr; return JNI_FALSE; }
    g_n_ctx = nCtx;

    // Warm-up
    llama_token bos = llama_token_bos(g_model);
    if (bos != LLAMA_TOKEN_NULL) {
        llama_batch wb = llama_batch_init(1, 0, 1);
        wb.token[0]    = bos;
        wb.pos[0]      = 0;
        wb.n_seq_id[0] = 1;
        wb.seq_id[0][0]= 0;
        wb.logits[0]   = 1;
        wb.n_tokens    = 1;
        llama_decode(g_ctx, wb);
        llama_batch_free(wb);
        llama_kv_cache_clear(g_ctx);
    }

    LOGI("Model loaded: %s, threads=%d", path.c_str(), g_n_threads);
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_nomad_engine_LlamaEngine_cacheSystemPrompt(
        JNIEnv* env, jclass, jstring jPrefix)
{
    if (!g_model || !g_ctx) return JNI_FALSE;
    pin_to_p_cores();

    std::string prefix = j2s(env, jPrefix);
    g_prefix_tokens = tokenize_str(prefix, true);
    if (g_prefix_tokens.empty()) return JNI_FALSE;

    llama_kv_cache_clear(g_ctx);

    // Use a robust batch setup
    int n_toks = (int)g_prefix_tokens.size();
    llama_batch batch = llama_batch_init(n_toks, 0, 1);
    batch.n_tokens = n_toks;
    for (int i = 0; i < n_toks; i++) {
        batch.token[i]     = g_prefix_tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (int8_t)(i == n_toks - 1); // CRITICAL: Enable logits for the last token
    }

    int rc = llama_decode(g_ctx, batch);
    llama_batch_free(batch);

    if (rc != 0) {
        LOGE("cacheSystemPrompt failed: %d", rc);
        g_prefix_tokens.clear();
        g_prefix_n_tokens = 0;
        return JNI_FALSE;
    }

    g_prefix_n_tokens = n_toks;
    LOGI("Cached %d prefix tokens", g_prefix_n_tokens);
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_nomad_engine_LlamaEngine_setThreads(
        JNIEnv*, jclass, jint nThreads, jint nThreadsBatch)
{
    if (!g_ctx) return;

    g_n_threads = std::min(std::max((int)nThreads, 1), 8);
    g_n_threads_batch = std::min(std::max((int)(nThreadsBatch > 0 ? nThreadsBatch : nThreads), 1), 8);
    llama_set_n_threads(g_ctx, g_n_threads, g_n_threads_batch);
    LOGI("Threads updated: gen=%d batch=%d", g_n_threads, g_n_threads_batch);
}

JNIEXPORT jstring JNICALL
Java_com_nomad_engine_LlamaEngine_generate(
    JNIEnv* env, jclass, jstring jPrompt, jint maxTokens,
    jfloat temperature, jfloat topP, jobject callback)
{
    if (!g_model || !g_ctx) return env->NewStringUTF("[ERROR: Model not loaded]");
    pin_to_p_cores();
    g_stop.store(false);
    g_utf8_cache.clear();

    std::string prompt = j2s(env, jPrompt);
    std::vector<llama_token> all_tokens = tokenize_str(prompt, true);
    if (all_tokens.empty()) return env->NewStringUTF("[ERROR: Tokenization failed]");

    // Truncate
    int max_p = g_n_ctx - maxTokens - 16;
    if (max_p <= 0) return env->NewStringUTF("[ERROR: maxTokens exceeds context window]");
    if ((int)all_tokens.size() > max_p) {
        size_t to_erase = all_tokens.size() - (size_t)max_p;
        all_tokens.erase(all_tokens.begin(), all_tokens.begin() + (long)to_erase);
    }

    // KV Cache Reuse
    int decode_start = 0;
    if (g_prefix_n_tokens > 0 && (int)all_tokens.size() >= g_prefix_n_tokens) {
        bool match = true;
        for (int i = 0; i < g_prefix_n_tokens; i++) {
            if (all_tokens[i] != g_prefix_tokens[i]) { match = false; break; }
        }
        if (match) {
            // Use -1 to clear all sequences at this position range
            llama_kv_cache_seq_rm(g_ctx, -1, g_prefix_n_tokens, -1);
            decode_start = g_prefix_n_tokens;
            LOGI("Prefix matched: skipping %d tokens", g_prefix_n_tokens);
        } else {
            llama_kv_cache_clear(g_ctx);
            decode_start = 0;
        }
    } else {
        llama_kv_cache_clear(g_ctx);
        decode_start = 0;
    }

    // Decode Prompt Suffix
    if (decode_start < (int)all_tokens.size()) {
        int n_toks = (int)all_tokens.size() - decode_start;
        llama_batch batch = llama_batch_init(n_toks, 0, 1);
        batch.n_tokens = n_toks;
        for (int i = 0; i < n_toks; i++) {
            batch.token[i]     = all_tokens[decode_start + i];
            batch.pos[i]       = decode_start + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (int8_t)(i == n_toks - 1);
        }
        int rc = llama_decode(g_ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) return env->NewStringUTF("[ERROR: Decode failed]");
    }

    // Sampler
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler* chain = llama_sampler_chain_init(sparams);
    if (temperature < 0.01f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(topP, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_dist((uint32_t)time(nullptr)));
    }

    jclass cls = env->GetObjectClass(callback);
    jmethodID onToken = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)Z");
    if (!onToken) {
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
        llama_sampler_free(chain);
        return env->NewStringUTF("[ERROR: Token callback unavailable]");
    }

    std::string result;
    std::string safe_result;
    char buf[256];
    int n_past = (int)all_tokens.size();
    bool stopped = false;

    // Pre-allocate single token batch for loop performance
    llama_batch nb = llama_batch_init(1, 0, 1);
    nb.n_tokens = 1;
    nb.n_seq_id[0] = 1;
    nb.seq_id[0][0] = 0;
    nb.logits[0] = true;

    for (int i = 0; i < maxTokens && !g_stop.load() && !stopped; i++) {
        llama_token tok = llama_sampler_sample(chain, g_ctx, -1);
        llama_sampler_accept(chain, tok);

        if (llama_token_is_eog(g_model, tok)) break;

        int n = llama_token_to_piece(g_model, tok, buf, sizeof(buf)-1, 0, true);
        if (n <= 0) break;
        buf[n] = '\0';
        result += buf;

        // Stop sequence check
        for (const auto& s : STOP_SEQS) {
            if (result.size() >= s.size() && result.compare(result.size() - s.size(), s.size(), s) == 0) {
                result.erase(result.size() - s.size());
                stopped = true; break;
            }
        }

        if (!stopped) {
            g_utf8_cache += buf;
            if (is_valid_utf8(g_utf8_cache)) {
                safe_result += g_utf8_cache;
                jstring js = env->NewStringUTF(g_utf8_cache.c_str());
                if (!js) {
                    g_utf8_cache.clear();
                    break;
                }
                bool ok = env->CallBooleanMethod(callback, onToken, js);
                env->DeleteLocalRef(js);
                g_utf8_cache.clear();
                if (env->ExceptionCheck()) {
                    env->ExceptionClear();
                    break;
                }
                if (!ok) break;
            }

            nb.token[0] = tok;
            nb.pos[0]   = n_past++;
            if (llama_decode(g_ctx, nb) != 0) break;
        }
    }

    if (!g_utf8_cache.empty() && is_valid_utf8(g_utf8_cache)) {
        safe_result += g_utf8_cache;
    }
    g_utf8_cache.clear();
    llama_batch_free(nb);
    llama_sampler_free(chain);
    return env->NewStringUTF(strip_stops(safe_result).c_str());
}

JNIEXPORT void JNICALL Java_com_nomad_engine_LlamaEngine_unloadModel(JNIEnv*, jclass) {
    g_stop.store(true);
    g_utf8_cache.clear();
    if (g_ctx) llama_free(g_ctx);
    if (g_model) llama_free_model(g_model);
    g_ctx = nullptr; g_model = nullptr;
    llama_backend_free();
}

JNIEXPORT jboolean JNICALL Java_com_nomad_engine_LlamaEngine_isModelLoaded(JNIEnv*, jclass) {
    return (g_model && g_ctx) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_nomad_engine_LlamaEngine_stopGeneration(JNIEnv*, jclass) {
    g_stop.store(true);
    g_utf8_cache.clear();
}

JNIEXPORT jstring JNICALL Java_com_nomad_engine_LlamaEngine_getModelInfo(JNIEnv* env, jclass) {
    if (!g_model) return env->NewStringUTF("{}");
    char buf[512];
    snprintf(buf, sizeof(buf), R"({"n_params":%lld,"n_ctx_train":%d,"n_embd":%d,"n_layer":%d})",
             (long long)llama_model_n_params(g_model), llama_n_ctx_train(g_model),
             llama_n_embd(g_model), llama_n_layer(g_model));
    return env->NewStringUTF(buf);
}

} // extern "C"
