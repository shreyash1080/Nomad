//  llama_wrapper.cpp
//  JNI bridge between Android Kotlin and llama.cpp

#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>

#include "llama.h"
#include "ggml.h"

#define LOG_TAG "PocketLLM"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// --- Global state ---
static llama_model*   g_model   = nullptr;
static llama_context* g_ctx     = nullptr;
static std::atomic<bool> g_stop_generation{false};

// Helper to convert jstring to std::string
static std::string jstring_to_std(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    llama_backend_init();
    LOGI("llama_backend_init called in JNI_OnLoad");
    return JNI_VERSION_1_6;
}

// ------ loadModel --------------------------------------------------------
JNIEXPORT jboolean JNICALL
Java_com_pocketllm_engine_LlamaEngine_loadModel(
        JNIEnv* env, jclass clazz,
        jstring modelPath,
        jint    nCtx,
        jint    nThreads,
        jboolean useGpu)
{
    // Free any existing model
    if (g_ctx)   { llama_free(g_ctx);         g_ctx   = nullptr; }
    if (g_model) { llama_free_model(g_model);  g_model = nullptr; }

    llama_backend_init();

    // Model params
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = useGpu ? 99 : 0;
    mparams.use_mmap     = true;
    mparams.use_mlock    = false;

    std::string path = jstring_to_std(env, modelPath);
    LOGI("Loading model: %s (GPU=%d)", path.c_str(), useGpu);

    g_model = llama_load_model_from_file(path.c_str(), mparams);
    if (!g_model) {
        LOGE("Failed to load model from %s", path.c_str());
        return JNI_FALSE;
    }

    // Context params
    // Optimization: limit threads to 4-6 for better thermal/core affinity on mobile
    uint32_t actual_threads = (nThreads > 6) ? 6 : (uint32_t)nThreads;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t)nCtx;
    cparams.n_threads       = (int32_t)actual_threads;
    cparams.n_threads_batch = (int32_t)actual_threads;
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.offload_kqv     = useGpu;

    g_ctx = llama_new_context_with_model(g_model, cparams);
    if (!g_ctx) {
        LOGE("Failed to create context");
        llama_free_model(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    LOGI("Model loaded successfully. Context size: %d", nCtx);
    return JNI_TRUE;
}

// ------ unloadModel -------------------------------------------------------
JNIEXPORT void JNICALL
Java_com_pocketllm_engine_LlamaEngine_unloadModel(JNIEnv* /* env */, jclass clazz)
{
    g_stop_generation.store(true);
    if (g_ctx)   { llama_free(g_ctx);         g_ctx   = nullptr; }
    if (g_model) { llama_free_model(g_model);  g_model = nullptr; }
    llama_backend_free();
    LOGI("Model unloaded");
}

// ------ isModelLoaded -----------------------------------------------------
JNIEXPORT jboolean JNICALL
Java_com_pocketllm_engine_LlamaEngine_isModelLoaded(JNIEnv* /* env */, jclass clazz)
{
    return (g_model != nullptr && g_ctx != nullptr) ? JNI_TRUE : JNI_FALSE;
}

// ------ stopGeneration ----------------------------------------------------
JNIEXPORT void JNICALL
Java_com_pocketllm_engine_LlamaEngine_stopGeneration(JNIEnv* /* env */, jclass clazz)
{
    g_stop_generation.store(true);
}

// ------ generate ----------------------------------------------------------
// Streams tokens back via a Kotlin callback (TokenCallback.onToken)
JNIEXPORT jstring JNICALL
Java_com_pocketllm_engine_LlamaEngine_generate(
        JNIEnv*  env,
        jclass clazz,
        jstring  jPrompt,
        jint     maxTokens,
        jfloat   temperature,
        jfloat   topP,
        jobject  callback)        // TokenCallback interface
{
    if (!g_model || !g_ctx) {
        LOGE("Model not loaded");
        return env->NewStringUTF("[ERROR: model not loaded]");
    }

    g_stop_generation.store(false);

    std::string prompt = jstring_to_std(env, jPrompt);

    // Tokenize prompt
    std::vector<llama_token> tokens(prompt.size() + 64);
    int n_tokens = llama_tokenize(
        g_model,
        prompt.c_str(),
        (int32_t)prompt.size(),
        tokens.data(),
        (int32_t)tokens.size(),
        /*add_special=*/true,
        /*parse_special=*/true
    );

    if (n_tokens < 0) {
        LOGE("Tokenization failed");
        return env->NewStringUTF("[ERROR: tokenization failed]");
    }
    tokens.resize(n_tokens);

    // Evaluate prompt
    auto t_start = std::chrono::high_resolution_clock::now();
    llama_kv_cache_clear(g_ctx);

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)n_tokens);
    if (llama_decode(g_ctx, batch) != 0) {
        LOGE("llama_decode failed for prompt");
        return env->NewStringUTF("[ERROR: decode failed]");
    }
    auto t_prompt_done = std::chrono::high_resolution_clock::now();
    LOGI("Prompt eval: %.3f s", std::chrono::duration<double>(t_prompt_done - t_start).count());

    // Get callback method
    jclass   cbClass  = env->GetObjectClass(callback);
    jmethodID onToken = env->GetMethodID(cbClass, "onToken", "(Ljava/lang/String;)Z");

    // Sampler chain
    auto sparams         = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(topP, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string full_response;
    char piece_buf[256];

    for (int i = 0; i < maxTokens && !g_stop_generation.load(); i++) {
        llama_token token_id = llama_sampler_sample(chain, g_ctx, -1);

        if (llama_token_is_eog(g_model, token_id)) break;

        // Decode token to text
        int n = llama_token_to_piece(g_model, token_id, piece_buf, sizeof(piece_buf) - 1, 0, true);
        if (n < 0) break;
        piece_buf[n] = '\0';

        full_response += piece_buf;

        // Send token to Kotlin callback
        jstring jPiece = env->NewStringUTF(piece_buf);
        jboolean cont  = env->CallBooleanMethod(callback, onToken, jPiece);
        env->DeleteLocalRef(jPiece);

        if (!cont) break;   // callback returned false → user stopped

        // Decode next token
        llama_batch next_batch = llama_batch_get_one(&token_id, 1);
        if (llama_decode(g_ctx, next_batch) != 0) break;
    }

    llama_sampler_free(chain);

    return env->NewStringUTF(full_response.c_str());
}

// ------ getModelInfo ------------------------------------------------------
JNIEXPORT jstring JNICALL
Java_com_pocketllm_engine_LlamaEngine_getModelInfo(JNIEnv* env, jclass clazz)
{
    if (!g_model) return env->NewStringUTF("{}");

    char buf[512];
    snprintf(buf, sizeof(buf),
        "{\"n_params\":%lld,\"n_ctx_train\":%d,\"n_embd\":%d,\"n_layer\":%d}",
        (long long)llama_model_n_params(g_model),
             llama_n_ctx_train(g_model),
             llama_n_embd(g_model),
             llama_n_layer(g_model)
    );
    return env->NewStringUTF(buf);
}

} // extern "C"
