# Nomad 🛰️

Nomad is a high-performance, **100% offline** LLM client for Android.It allows you to run state-of-the-art language models locally on your device with advanced file analysis capabilities.

## ✨ Features

- **Offline Intelligence**: No internet required. Your data never leaves your device.
- **Advanced File Analysis**:
  - **Images**: OCR powered by Google ML Kit.
  - **PDFs**: Full text extraction using `PdfRenderer` and OCR for scanned documents.
  - **Office (DOCX/XLSX)**: Direct XML parsing for fast, library-free document reading.
- **Dynamic Engine**: Automatically switches between Llama 3 and ChatML prompt formats.
- **Web Search**: Empowers the local model with real-time web context for improved response accuracy.
- **RAM Guard**: Intelligent memory monitoring to prevent system OOM.

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
## ⚖️ License & Attribution

Nomad is an open-source client, but the underlying models and libraries are governed by their respective creators. By using this software, you agree to the following terms:

### 🤖 Model Licenses
| Model Family | License | Commercial Usage | Key Requirement |
| :--- | :--- | :--- | :--- |
| **Llama 3.2** | [Llama 3.2 Community](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) | Free < 700M MAU | Must follow Meta's Acceptable Use Policy. |
| **Gemma 2** | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) | Free | Must include Google Gemma attribution. |
| **Phi-3** | [MIT License](https://opensource.org/license/mit) | Free | Include Microsoft copyright notice. |
| **Qwen 2.5** | [Qwen License](https://github.com/QwenLM/Qwen2.5/blob/main/LICENSE) | Free < 200M MAU | Requires agreement for extremely large scale use. |
| **Mistral 7B** | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Free | Include Mistral AI attribution. |

### ⚙️ Core Components
- **Inference Engine**: [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT License). Copyright (c) 2023 Georgi Gerganov.
- **OCR/Analysis**: [Google ML Kit](https://developers.google.com/ml-kit/terms) (Google APIs Terms).
- **Web Retrieval**: [DuckDuckGo](https://duckduckgo.com/privacy) (Subject to DDG Terms of Service).

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
