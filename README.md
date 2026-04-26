# Nomad 🛰️

**Nomad** is a high-performance, **privacy-first** LLM client for Android. It allows you to run state-of-the-art language models locally on your device with advanced file analysis and real-time capabilities.

[![Latest Release](https://img.shields.io/github/v/release/shreyash1080/Nomad?label=latest%20build&color=white&style=flat-square)](https://github.com/shreyash1080/Nomad/releases)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)](#)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](#)
[![Platform](https://img.shields.io/badge/platform-Android-3DDC84?style=flat-square&logo=android)](#)

---

## Download Latest APK:
 [<img src="https://img.shields.io/badge/Download-Latest%20APK-black?style=for-the-badge&logo=android&logoColor=white" height="40">](https://github.com/shreyash1080/Nomad/releases/download/Nomad-apk/Nomad.apk)

> Experience true local intelligence. No sign-ups, no subscriptions, and no tracking.
> **Note:** A physical device with at least **6GB RAM** is recommended for optimal performance with 3B+ parameter models.

---

## ✨ Features

### 🔐 Privacy & Performance
- **Offline Intelligence**: **100%** on-device processing. Your data never leaves your device.
- **RAM Guard**: An intelligent memory monitoring system that prevents **Out-Of-Memory (OOM)** crashes by managing model offloading.
- **Pure Black UI**: A minimalist, high-contrast interface optimized for **OLED** screens.

### 📄 Advanced File Analysis
- **Images**: High-speed OCR powered by **Google ML Kit**.
- **PDFs**: Full text extraction using `PdfRenderer` and OCR support for scanned documents.
- **Office (DOCX/XLSX)**: Direct XML parsing for fast, library-free reading.

### 🌐 Real-time Awareness
- **Web Search**: Supplements the offline AI with real-time web context via **DuckDuckGo** to provide accurate and up-to-date answers.

---

## 🛠️ Architecture

Nomad is built with a focus on speed and modern Android standards:

- **Core Engine**: C++ **`llama.cpp`** core integrated via a custom **JNI wrapper**.
- **Dynamic Formatting**: Automatically switches between **Llama 3** and **ChatML** prompt formats.
- **Modern UI**: **100% Jetpack Compose** for a fluid, responsive experience across all screen sizes.
- **Asynchronous**: Built on **Kotlin Coroutines** for non-blocking file processing and streaming responses.

---

## ⚖️ License & Attribution

Nomad is an open-source client, but the underlying models and libraries are governed by their respective creators. By using this software, you agree to the following terms:

### 🤖 Model Licenses

| Model Family | License | Commercial Usage | Key Requirement |
| :--- | :--- | :--- | :--- |
| **Llama 3.2** | [Llama 3.2 Community](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) | Free < 700M MAU | Must follow Meta's Acceptable Use Policy. |
| **Gemma 2** | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) | Free | Must include Google Gemma attribution. |
| **Phi-3** | [MIT License](https://opensource.org/license/mit) | Free | Include Microsoft copyright notice. |
| **Qwen 2.5** | [Qwen License](https://github.com/QwenLM/Qwen2.5/blob/main/LICENSE) | Free < 200M MAU | Requires agreement for large scale use. |
| **Mistral 7B** | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Free | Include Mistral AI attribution. |

### ⚙️ Core Components
- **Inference Engine**: [llama.cpp](https://github.com/ggerganov/llama.cpp) (**MIT License**). Copyright (c) 2023 Georgi Gerganov.
- **OCR/Analysis**: [Google ML Kit](https://developers.google.com/ml-kit/terms) (**Google APIs Terms**).
- **Web Retrieval**: [DuckDuckGo](https://duckduckgo.com/privacy) (Subject to **DDG Terms of Service**).

---

## 🚀 For Developers

### Prerequisites
- **Android Studio Jellyfish** or newer.
- **NDK (Side by side)** installed via the SDK Manager.
- A **physical device** (Emulators are not recommended for LLM inference).

### Build Instructions
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/shreyash1080/Nomad.git](https://github.com/shreyash1080/Nomad.git)

2. Run the setup script:
   ./setup_llama.sh
   (This syncs the llama.cpp core submodule).

3. Open in Android Studio and click Run.

Developed with ❤️ for privacy and local AI.
