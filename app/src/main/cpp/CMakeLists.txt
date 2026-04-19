cmake_minimum_required(VERSION 3.22.1)
project("pocketllm")

# ─── llama.cpp source (cloned into src/main/cpp/llama.cpp/) ───────────────────
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

# Build options
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)

# Enable Vulkan GPU backend on Android
if(ANDROID)
    set(LLAMA_VULKAN ON CACHE BOOL "" FORCE)
    find_library(VULKAN_LIB vulkan)
endif()

# Add llama.cpp as subdirectory
add_subdirectory(${LLAMA_DIR} llama_build)

# ─── JNI Wrapper ──────────────────────────────────────────────────────────────
add_library(
    pocketllm_jni
    SHARED
    llama_wrapper.cpp
)

target_include_directories(pocketllm_jni PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)

target_link_libraries(pocketllm_jni
    llama
    ggml
    android
    log
    ${VULKAN_LIB}
)

target_compile_options(pocketllm_jni PRIVATE
    -O3
    -DNDEBUG
    -std=c++17
)
