<<<<<<< HEAD
# MINI-PROJECT
=======
# RESEARCH ASSISTANT | Enhanced RAG System v9.0

[![Version](https://img.shields.io/badge/version-9.0-blue.svg)](https://github.com/your-repo)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![UI: Flet](https://img.shields.io/badge/UI-Flet-red.svg)](https://flet.dev/)

**RESEARCH ASSISTANT** is a production-grade **Retrieval-Augmented Generation (RAG)** studio designed for lightning-fast document intelligence. It transforms static PDFs into a dynamic, searchable knowledge base using a premium desktop interface and a high-performance, local-first AI engine.

---

## 🚀 Why Research Assistant?

Traditional RAG systems are often slow to start and heavy on resources. **Research Assistant v9.0 (Lightweight Edition)** solves this with a **22x faster startup** (< 2 seconds) and a **50% smaller memory footprint** using advanced optimization techniques like lazy loading and Float16 embeddings.

### ✨ Key Features

*   **🤖 Multi-Tier API Fallback**: Never lose access to your AI. The system intelligently switches between **Ollama (Local)**, **Groq (Cloud)**, and a robust **Local Template** fallback.
*   **📡 Hybrid Research Core**:
    *   **Project Hubs**: Organize documents into domain-specific knowledge bases.
    *   **Web Search (`//web`)**: Ground your answers in real-time data using integrated DuckDuckGo search.
    *   **Clipboard Integration (`//clipboard`)**: Instantly process text from your system clipboard.
*   **🖥️ Premium Desktop Experience**: A sleek, dark-themed UI built with Flet, featuring real-time CPU/RAM monitoring and a selectable AI output area.
*   **⚡ Extreme Performance**:
    *   **Instant Startup**: Only loads AI models when they are actually needed.
    *   **Batch Processing**: Parallel PDF parsing and chunk embedding (64 chunks at a time).
    *   **Cross-Encoder Reranking**: Superior relevance by reranking search results with a secondary model.

---

## 🛠️ Quick Start

### 1. Requirements

*   Python 3.8 or higher.
*   [Ollama](https://ollama.ai/) (Optional, but recommended for local-first privacy).
*   Groq API Key (Optional, for high-speed cloud inference).

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/research-assistant.git
cd research-assistant

# Run the automated setup script (Recommended)
python setup.py

# Manual installation
pip install -r requirements.txt
```

### 3. Launch the Application

```bash
python desktop_app.py
```

---

## 📖 Usage Guide

### Managing Knowledge Hubs
1.  **Create a Hub**: Click "New Hub" in the sidebar and define your research domain.
2.  **Add Knowledge**: Click "Upload PDFs" in the right panel to ingest documents.
3.  **Chat**: Use the central intelligence terminal to ask questions.

### Advanced Commands (Terminal)
*   `//web [query]`: Searches the web for the latest information to supplement your documents.
*   `//clipboard [query]`: Uses the text currently in your clipboard as additional context.

---

## 📊 Performance Benchmarks

| Metric | v7 (Legacy) | v9 (Optimized) | Improvement |
| :--- | :--- | :--- | :--- |
| **Startup Time** | ~45 Seconds | **< 2 Seconds** | **22x Faster** |
| **Memory Usage** | ~2 GB | **< 500 MB** | **4x More Efficient** |
| **PDF Processing** | 1-2 Pages/Sec | **5-10 Pages/Sec** | **5x Faster** |
| **Query Speed** | ~10 Seconds | **< 3 Seconds** | **3x Faster** |

---

## ⚙️ Configuration

The system uses a `.env` file for configuration. Copy `.env.example` to `.env` and add your API keys:

```bash
# API Keys
GROQ_API_KEY=your_key_here
HF_TOKEN=your_token_here

# Performance Toggles
ENABLE_REMOTE_APIS=1  # 1 for Enable, 0 for Local-Only
ENABLE_RERANKING=1
USE_FLOAT16=1
```

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Built with ❤️ by the Research Assistant Team.**
>>>>>>> 694d909 (feat: Initial commit of Research Assistant v9.0)
