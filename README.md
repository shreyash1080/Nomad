# Nomad 🛰️

Nomad is a high-performance, **100% offline** LLM client for Android. Inspired by the simplicity of Gemini, it allows you to run state-of-the-art language models (like Llama 3.2) locally on your device with advanced file analysis capabilities.

## ✨ Features

- **Pure Black UI**: Minimalist, high-contrast design optimized for OLED screens.
- **Offline Intelligence**: No internet required. Your data never leaves your device.
- **Advanced File Analysis**:
  - **Images**: OCR powered by Google ML Kit.
  - **PDFs**: Full text extraction using `PdfRenderer` and OCR for scanned documents.
  - **Office (DOCX/XLSX)**: Direct XML parsing for fast, library-free document reading.
- **Dynamic Engine**: Automatically switches between Llama 3 and ChatML prompt formats.
- **RAM Guard**: Intelligent memory monitoring to prevent system OOM (Out-Of-Memory) crashes.

## 🛠️ Architecture

- **Engine**: C++ `llama.cpp` core with a custom JNI wrapper for maximum performance.
- **UI**: 100% Jetpack Compose for a fluid, responsive experience across all screen sizes.
- **Processing**: Kotlin Coroutines for non-blocking file analysis and streaming responses.

## 📜 Licensing & Open Source

Nomad leverages several open-source technologies:

1.  **[llama.cpp](https://github.com/ggerganov/llama.cpp)** (MIT License) - The core inference engine.
2.  **[Google ML Kit](https://developers.google.com/ml-kit)** (Google APIs Terms) - Used for on-device Text Recognition (OCR).
3.  **[Jetpack Compose](https://developer.android.com/jetpack/compose)** (Apache 2.0) - Modern Android toolkit for the UI.
4.  **[Kotlin Coroutines](https://github.com/Kotlin/kotlinx.coroutines)** (Apache 2.0) - Asynchronous programming.

## 🚀 For Developers

### Prerequisites
- Android Studio Jellyfish or newer.
- NDK (Side by side) installed via SDK Manager.
- A physical device with at least 6GB RAM is recommended for 3B models.

### Build
1. Clone the repository.
2. Run `./setup_llama.sh` (or manually sync the `llama.cpp` submodule).
3. Open in Android Studio and click **Run**.

---
*Developed with ❤️ for privacy and local AI.*
