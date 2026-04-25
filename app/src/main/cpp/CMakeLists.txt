cmake_minimum_required(VERSION 3.22.1)
project("nomad")

set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

# ── llama.cpp build options ────────────────────────────────────────────────────
set(GGML_VULKAN          OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

# ── Performance: ARM NEON + dotprod + fp16 ─────────────────────────────────────
# These flags unlock NEON vectorized matrix multiply — biggest CPU speedup
if(ANDROID AND CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    set(GGML_CPU_AARCH64      ON  CACHE BOOL "" FORCE)  # ARM64-optimized GGML kernels
    set(GGML_NEON             ON  CACHE BOOL "" FORCE)  # NEON SIMD
    add_compile_options(
        -march=armv8.2-a+dotprod+fp16   # dotprod = fast int8 matmul, fp16 = half precision
        -mcpu=cortex-a55                # conservative baseline (all ARMv8.2 phones)
        -O3
        -ffast-math                     # allow approximate FP (big speedup, fine for LLM)
        -fno-finite-math-only           # GGML requires INF/NaN for some operations
        -funroll-loops
        -fomit-frame-pointer
    )
endif()

# ── Vulkan GPU backend ─────────────────────────────────────────────────────────
if(FALSE) # Disabled due to missing vulkan-shaders-gen in environment
    find_library(VULKAN_LIB vulkan)
    if(VULKAN_LIB)
        set(GGML_VULKAN ON CACHE BOOL "" FORCE)
        message(STATUS "Vulkan found — GPU acceleration enabled")
    else()
        message(WARNING "Vulkan not found — CPU only")
    endif()
endif()

add_subdirectory(${LLAMA_DIR} llama_build)

# ── JNI wrapper ───────────────────────────────────────────────────────────────
add_library(nomad_jni SHARED llama_wrapper.cpp)

target_include_directories(nomad_jni PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)

target_link_libraries(nomad_jni
    llama
    ggml
    android
    log
    ${VULKAN_LIB}
)

target_compile_options(nomad_jni PRIVATE
    -O3
    -ffast-math
    -std=c++17
)