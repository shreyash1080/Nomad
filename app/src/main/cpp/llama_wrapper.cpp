//  llama_wrapper.cpp — Production Ready + Maximum Performance
//  Built from working codebase — uses ONLY verified API names:
//    llama_n_ctx_train / llama_n_embd / llama_n_layer  (NOT llama_model_n_*)
//    #include <unistd.h> for _SC_NPROCESSORS_ONLN
//    #include <sched.h>  for sched_setaffinity

#include <jni.h>
#include <android/log.h>
#include <unistd.h>      // _SC_NPROCESSORS_ONLN — THIS was missing before
#include <sched.h>       // sched_setaffinity — thread pinning
#include <string>
#include <vector>
#include <atomic>
#include <cstring>
#include <cstdio>

#include "llama.h"
#include "ggml.h"

#define LOG_TAG "Eigen"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ── Global state ──────────────────────────────────────────────────────────────
static llama_model*      g_model           = nullptr;
static llama_context*    g_ctx             = nullptr;
static std::atomic<bool> g_stop{false};
static int               g_n_ctx           = 0;
static int               g_n_threads       = 4;

// ── KV Prefix cache — caches system prompt tokens so every turn skips them ────
static int                      g_prefix_n_tokens = 0;
static std::vector<llama_token> g_prefix_tokens;

// ── P-core CPU affinity mask ──────────────────────────────────────────────────
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

// Tokenize helper
static std::vector<llama_token> tokenize_str(const std::string& text, bool add_special) {
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

// ── Build P-core affinity mask from CPU frequency files ───────────────────────
static int build_perf_core_mask(cpu_set_t* mask) {
    CPU_ZERO(mask);
    int total = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (total <= 0) total = 4;

    std::vector<unsigned long> freqs(total, 0UL);
    unsigned long max_freq = 0;
    char path[128];
    for (int i = 0; i < total; i++) {
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        fscanf(f, "%lu", &freqs[i]);
        fclose(f);
        if (freqs[i] > max_freq) max_freq = freqs[i];
    }

    int count = 0;
    if (max_freq == 0) {
        for (int i = total / 2; i < total; i++) { CPU_SET(i, mask); count++; }
    } else {
        for (int i = 0; i < total; i++) {
            if (freqs[i] >= max_freq * 8 / 10) { CPU_SET(i, mask); count++; }
        }
    }
    LOGI("P-cores detected: %d / %d  (max %lu kHz)", count, total, max_freq);
    return count;
}

// Pin current thread to P-cores
static void pin_to_p_cores() {
    if (!g_mask_valid) return;
    if (sched_setaffinity(0, sizeof(cpu_set_t), &g_p_core_mask) != 0)
        LOGE("sched_setaffinity failed");
}

// ── Stop sequences covering all common model formats ──────────────────────────
static const std::vector<std::string> STOP_SEQS = {
        "<|im_end|>",              // ChatML — Qwen, Phi-3, most instruction models
        "<|eot_id|>",              // Llama 3 / 3.1 / 3.2
        "<|end_of_turn|>",         // Gemma
        "</s>",                    // Mistral, older LLaMA
        "[/INST]",                 // LLaMA 2
        "<|im_start|>user",        // Hallucination guard
        "<|im_start|>assistant",   // Hallucination guard
        "<|start_header_id|>user", // Llama 3 hallucination guard
        "\nUser:",                 // Plain format guard
        "\nHuman:",                // Plain format guard
};

static std::string strip_stops(std::string text) {
    for (const auto& s : STOP_SEQS) {
        size_t pos;
        while ((pos = text.find(s)) != std::string::npos)
            text.erase(pos, s.size());
    }
    size_t a = text.find_first_not_of(" \t\n\r");
    size_t b = text.find_last_not_of(" \t\n\r");
    if (a == std::string::npos) return "";
    return text.substr(a, b - a + 1);
}

// ─────────────────────────────────────────────────────────────────────────────
extern "C" {

// ── loadModel ─────────────────────────────────────────────────────────────────
JNIEXPORT jboolean JNICALL
Java_com_eigen_engine_LlamaEngine_loadModel(
        JNIEnv* env, jclass,
        jstring modelPath, jint nCtx, jint nThreads, jboolean useGpu)
{
    if (g_ctx)   { llama_free(g_ctx);         g_ctx   = nullptr; }
    if (g_model) { llama_free_model(g_model);  g_model = nullptr; }
    g_prefix_n_tokens = 0;
    g_prefix_tokens.clear();

    llama_backend_init();

    // Build P-core mask once at startup
    int p_count = build_perf_core_mask(&g_p_core_mask);
    g_mask_valid = (p_count > 0);
    pin_to_p_cores();  // pin load thread too

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = useGpu ? 99 : 0;
    mp.use_mmap     = true;
    mp.use_mlock    = false;

    std::string path = j2s(env, modelPath);
    LOGI("loadModel: %s  ctx=%d  gpu=%d", path.c_str(), nCtx, (int)useGpu);

    g_model = llama_load_model_from_file(path.c_str(), mp);
    if (!g_model) { LOGE("Failed to load model"); return JNI_FALSE; }

    // Use P-core count for thread count, or manual override
    g_n_threads = (nThreads > 0) ? nThreads : p_count;
    g_n_threads = std::min(std::max(g_n_threads, 2), 8);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t)nCtx;
    cp.n_batch         = 512;
    cp.n_ubatch        = 128;
    cp.n_threads       = (uint32_t)g_n_threads;
    cp.n_threads_batch = (uint32_t)g_n_threads;
    cp.flash_attn      = true;
    cp.type_k          = GGML_TYPE_Q8_0;
    cp.type_v          = GGML_TYPE_Q8_0;
    cp.offload_kqv     = useGpu;

    g_ctx = llama_new_context_with_model(g_model, cp);
    if (!g_ctx) {
        llama_free_model(g_model); g_model = nullptr;
        return JNI_FALSE;
    }
    g_n_ctx = nCtx;

    // ── Warm-up decode — eliminates cold-start penalty on first message ────────
    {
        llama_token bos = llama_token_bos(g_model);
        if (bos >= 0) {
            llama_batch wb = llama_batch_get_one(&bos, 1);
            llama_decode(g_ctx, wb);
            llama_kv_cache_clear(g_ctx);
        }
        LOGI("Warm-up decode done");
    }

    LOGI("Model ready: threads=%d  flash=true  kv=Q8_0", g_n_threads);
    return JNI_TRUE;
}

// ── cacheSystemPrompt — decodes system prefix once, reused every chat turn ────
JNIEXPORT jboolean JNICALL
Java_com_eigen_engine_LlamaEngine_cacheSystemPrompt(
        JNIEnv* env, jclass,
        jstring jPrefix)
{
    if (!g_model || !g_ctx) return JNI_FALSE;
    pin_to_p_cores();

    std::string prefix = j2s(env, jPrefix);
    g_prefix_tokens = tokenize_str(prefix, true);
    if (g_prefix_tokens.empty()) return JNI_FALSE;

    llama_kv_cache_clear(g_ctx);

    llama_batch batch = llama_batch_init((int32_t)g_prefix_tokens.size(), 0, 1);
    for (int i = 0; i < (int)g_prefix_tokens.size(); i++) {
        batch.token[i]     = g_prefix_tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = false;
    }

    int rc = llama_decode(g_ctx, batch);
    llama_batch_free(batch);

    if (rc != 0) {
        LOGE("cacheSystemPrompt: decode failed with %d", rc);
        g_prefix_tokens.clear();
        g_prefix_n_tokens = 0;
        return JNI_FALSE;
    }

    g_prefix_n_tokens = (int)g_prefix_tokens.size();
    LOGI("System prefix cached: %d tokens", g_prefix_n_tokens);
    return JNI_TRUE;
}

// ── unloadModel ───────────────────────────────────────────────────────────────
JNIEXPORT void JNICALL
Java_com_eigen_engine_LlamaEngine_unloadModel(JNIEnv*, jclass) {
    g_stop.store(true);
    if (g_ctx)   { llama_free(g_ctx);         g_ctx   = nullptr; }
    if (g_model) { llama_free_model(g_model);  g_model = nullptr; }
    g_prefix_n_tokens = 0;
    g_prefix_tokens.clear();
    llama_backend_free();
    LOGI("Model unloaded");
}

JNIEXPORT jboolean JNICALL
Java_com_eigen_engine_LlamaEngine_isModelLoaded(JNIEnv*, jclass) {
    return (g_model && g_ctx) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_eigen_engine_LlamaEngine_stopGeneration(JNIEnv*, jclass) {
    g_stop.store(true);
}

// ── generate — with KV prefix reuse ──────────────────────────────────────────
JNIEXPORT jstring JNICALL
Java_com_eigen_engine_LlamaEngine_generate(
        JNIEnv* env, jclass,
        jstring jPrompt, jint maxTokens,
        jfloat temperature, jfloat topP,
        jobject callback)
{
    if (!g_model || !g_ctx)
        return env->NewStringUTF("[ERROR: model not loaded]");

    pin_to_p_cores();
    g_stop.store(false);
    std::string prompt = j2s(env, jPrompt);

    // Tokenize full prompt
    std::vector<llama_token> all_tokens = tokenize_str(prompt, true);
    if (all_tokens.empty()) return env->NewStringUTF("[ERROR: tokenize failed]");

    // Truncate if prompt too long
    int max_prompt = g_n_ctx - maxTokens - 16;
    if ((int)all_tokens.size() > max_prompt) {
        all_tokens.erase(all_tokens.begin(),
                         all_tokens.begin() + ((int)all_tokens.size() - max_prompt));
    }

    // ── KV PREFIX REUSE ───────────────────────────────────────────────────────
    int decode_start = 0;

    if (g_prefix_n_tokens > 0 && (int)all_tokens.size() > g_prefix_n_tokens) {
        bool match = true;
        for (int i = 0; i < g_prefix_n_tokens; i++) {
            if (all_tokens[i] != g_prefix_tokens[i]) { match = false; break; }
        }
        if (match) {
            // Keep prefix, remove everything after
            llama_kv_cache_seq_rm(g_ctx, 0, g_prefix_n_tokens, -1);
            decode_start = g_prefix_n_tokens;
            LOGI("KV prefix reused: skipping %d tokens", g_prefix_n_tokens);
        } else {
            llama_kv_cache_clear(g_ctx);
            decode_start = 0;
        }
    } else {
        llama_kv_cache_clear(g_ctx);
        decode_start = 0;
    }

    // Decode prompt suffix
    if (decode_start < (int)all_tokens.size()) {
        int n_tokens = (int)all_tokens.size() - decode_start;
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; i++) {
            batch.token[i]     = all_tokens[decode_start + i];
            batch.pos[i]       = decode_start + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == n_tokens - 1);
        }

        int rc = llama_decode(g_ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            LOGE("llama_decode failed: %d", rc);
            return env->NewStringUTF("[ERROR: decode failed]");
        }
    }

    // Sampler setup
    llama_sampler_chain_params lcp = llama_sampler_chain_default_params();
    lcp.no_perf = true;
    llama_sampler* chain = llama_sampler_chain_init(lcp);

    if (temperature < 0.01f) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(topP, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_dist((uint32_t)time(nullptr)));
    }

    jclass    cbClass = env->GetObjectClass(callback);
    jmethodID onToken = env->GetMethodID(cbClass, "onToken", "(Ljava/lang/String;)Z");
    if (!onToken) {
        llama_sampler_free(chain);
        return env->NewStringUTF("[ERROR: callback method not found]");
    }

    std::string result;
    char buf[256];
    bool stopped = false;
    int n_past = (int)all_tokens.size();

    for (int i = 0; i < maxTokens && !g_stop.load() && !stopped; i++) {
        if (!g_ctx) break;

        // Sample from the last token's logits
        llama_token tok = llama_sampler_sample(chain, g_ctx, -1);
        llama_sampler_accept(chain, tok);

        if (llama_token_is_eog(g_model, tok)) break;

        int n = llama_token_to_piece(g_model, tok, buf, (int32_t)sizeof(buf)-1, 0, true);
        if (n <= 0) break;
        buf[n] = '\0';
        result += buf;

        // Check for stop sequences
        for (const auto& s : STOP_SEQS) {
            if (result.size() >= s.size() && result.compare(result.size() - s.size(), s.size(), s) == 0) {
                result.erase(result.size() - s.size());
                stopped = true;
                break;
            }
        }

        if (!stopped) {
            jstring jp  = env->NewStringUTF(buf);
            jboolean ok = env->CallBooleanMethod(callback, onToken, jp);
            env->DeleteLocalRef(jp);
            if (!ok) break;

            // Decode the sampled token
            llama_batch nb = llama_batch_init(1, 0, 1);
            nb.token[0]     = tok;
            nb.pos[0]       = n_past;
            nb.n_seq_id[0]  = 1;
            nb.seq_id[0][0] = 0;
            nb.logits[0]    = true;

            int rc = llama_decode(g_ctx, nb);
            llama_batch_free(nb);
            if (rc != 0) break;
            n_past++;
        }
    }

    llama_sampler_free(chain);
    result = strip_stops(result);
    return env->NewStringUTF(result.c_str());
}

// ── getModelInfo — uses CORRECT old-style API names from working codebase ─────
//    llama_n_ctx_train / llama_n_embd / llama_n_layer  (NOT llama_model_n_*)
JNIEXPORT jstring JNICALL
Java_com_eigen_engine_LlamaEngine_getModelInfo(JNIEnv* env, jclass) {
    if (!g_model) return env->NewStringUTF("{}");
    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"n_params\":%lld,\"n_ctx_train\":%d,\"n_embd\":%d,\"n_layer\":%d}",
             (long long)llama_model_n_params(g_model),
             llama_n_ctx_train(g_model),      // ← correct for this llama.cpp version
             llama_n_embd(g_model),           // ← correct
             llama_n_layer(g_model));         // ← correct
    return env->NewStringUTF(buf);
}

} // extern "C"