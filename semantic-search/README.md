<div align="center">

# 🔍 Semantic Search Engine

### Powered by Endee Vector Database & Groq LLM

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Endee%20Server-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/endeeio/endee-server)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-F55036?style=for-the-badge&logo=meta&logoColor=white)](https://groq.com/)

> **A Retrieval-Augmented Generation (RAG) pipeline that enables natural language Q&A over PDF documents using vector similarity search and LLM-powered answer synthesis.**

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [Usage](#-usage) · [Tech Stack](#-tech-stack)

<br>

<img src="assets/visual%20selection.png" alt="AI-Powered Information Retrieval Pipeline" width="700">

</div>

---

## 📌 Overview

Traditional keyword-based search fails when users phrase queries differently from the source text — searching for *"machine learning applications in healthcare"* won't surface a paragraph about *"AI-driven diagnosis tools for hospitals"*, even though they discuss the same topic.

This project addresses that gap with a **two-stage RAG approach**:

| Stage | What Happens |
|:------|:-------------|
| **1. Semantic Retrieval** | Documents are chunked, embedded into 768-dimensional vectors, and stored in the Endee vector database. User queries are matched by *meaning*, not keywords, using cosine similarity over HNSW indexes. |
| **2. LLM Answer Generation** | The top-k retrieved chunks are sent to a Groq-hosted LLM (`llama-3.1-8b-instant`) which synthesizes a clear, structured answer — grounded entirely in the source material. |

---

## 🏗 Architecture

```
                          ┌─────────────────────────┐
                          │    Streamlit Web UI     │
                          │     (localhost:8501)    │
                          └────────┬────────┬───────┘
                                   │        │
                        Upload PDF │        │ Search Query
                                   ▼        ▼
               ┌───────────────────────────────────────────┐
               │                 app.py                    │
               │          (Orchestration Layer)            │
               └───┬──────────┬──────────┬─────────────────┘
                   │          │          │
          ┌────────▼──-┐  ┌───▼───┐  ┌───▼────┐
          │  utils.py  │  │embed  │  │search  │
          │            │  │.py    │  │.py     │
          │ PDF Extract│  │       │  │        │
          │ + Chunking │  │Encode │  │ Query  │
          └────────┬───┘  └───┬───┘  └───┬────┘
                   │          │          │
                   └──────────▼──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   Endee Vector DB   │ ◄── Docker     (localhost:8080)
                   │                      │
                   |Index: semantic_search|
                   │  Dim: 768 │ Cosine   │
                   │  Precision: FLOAT32  │
                   └──────────┬──────────┘
                              │
                     Top-k Chunks
                              │
                   ┌──────────▼──────────┐
                   │      rag.py          │
                   │                      │
                   │  Groq LLM            │
                   │  llama-3.1-8b-instant│
                   │  (Answer Synthesis)  │
                   └──────────┬──────────┘
                              │
                      Structured Answer
                              │
                   ┌──────────▼──────────┐
                   │   User Interface     │
                   │  Answer + Sources    │
                   └─────────────────────┘
```

### Data Flow

<details>
<summary><b>📥 Ingestion Pipeline</b> (PDF → Vectors)</summary>

1. User uploads a PDF through the Streamlit sidebar
2. **Text Extraction** — `PyPDF2` extracts raw text from all pages
3. **Section-Aware Chunking** — Text is intelligently split into ~400-character chunks:
   - Document sections (Skills, Projects, Education, etc.) are auto-detected via header pattern matching
   - Each chunk is prefixed with its section label (e.g., `[Projects]`) to preserve semantic context
   - Contact information is auto-labeled using email/phone regex patterns
   - Duplicate chunks are eliminated via normalized text comparison
4. **Embedding** — Each chunk is encoded into a 768-dimensional vector using `multi-qa-mpnet-base-cos-v1` (a Q&A-optimized Sentence Transformer model)
5. **Storage** — Vectors are batch-upserted into the Endee index with original text stored as metadata

</details>

<details>
<summary><b>🔎 Search + RAG Pipeline</b> (Query → Answer)</summary>

1. User types a natural language question
2. The query is encoded into a 768-dim vector using the same embedding model
3. Endee performs approximate nearest neighbor search (HNSW algorithm, cosine similarity)
4. Top-k most relevant chunks are retrieved with similarity scores
5. Retrieved chunks + the original query are sent to Groq's LLM
6. The LLM generates a structured, cited answer grounded in the source material
7. Both the synthesized answer and expandable source chunks are displayed

</details>

---

## 🛠 Tech Stack

| Layer | Technology | Details |
|:------|:-----------|:--------|
| **Vector Database** | [Endee](https://github.com/endee-io/endee) | High-performance vector storage & HNSW-based similarity search (Docker) |
| **Embeddings** | [Sentence Transformers](https://www.sbert.net/) | `multi-qa-mpnet-base-cos-v1` — 768-dim, asymmetric Q&A-optimized |
| **LLM** | [Groq](https://groq.com/) | `llama-3.1-8b-instant` — ultra-low-latency inference for RAG |
| **PDF Parsing** | [PyPDF2](https://pypdf2.readthedocs.io/) | Robust text extraction from uploaded PDFs |
| **Frontend** | [Streamlit](https://streamlit.io/) | Interactive web UI with file upload & search |
| **Config** | [python-dotenv](https://pypi.org/project/python-dotenv/) | Environment variable management via `.env` |

---

## 🔌 How Endee is Used

Endee serves as the **core vector storage and retrieval engine**. The project interacts with it in four ways:

| Operation | SDK Method | Description |
|:----------|:-----------|:------------|
| **Connect** | `Endee()` + `set_base_url()` | Initialize the client, pointing at the Docker server. Supports both token-authenticated and open mode. |
| **Create Index** | `create_index()` | Create `semantic_search` index — 768 dimensions, cosine distance, FLOAT32 precision. |
| **Store Vectors** | `index.upsert()` | Batch-upsert chunk embeddings with metadata (original text, source file, chunk ID). |
| **Search** | `index.query()` | Approximate nearest neighbor search (HNSW) — returns top-k chunks ranked by cosine similarity. |

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version |
|:------------|:--------|
| Python | 3.10+ (3.12 recommended) |
| Docker Desktop | Latest |
| Groq API Key | Free — [console.groq.com](https://console.groq.com) |
| Disk Space | ~500 MB (for embedding model, auto-downloaded on first run) |
| GPU | **Not required** — embeddings run on CPU, LLM runs on Groq cloud |

### 1️⃣ Start the Endee Server

```bash
docker pull endeeio/endee-server:latest
docker run -d -p 8080:8080 --name endee-server \
  -e NDD_AUTH_TOKEN="" \
  endeeio/endee-server:latest
```

Verify the container is healthy:

```bash
docker ps
# STATUS should show "Up ... (healthy)"
```

### 2️⃣ Install Dependencies

```bash
cd semantic-search
pip install -r requirements.txt
```

### 3️⃣ Configure Environment

Create or edit the `.env` file:

```env
# Endee Vector Database
ENDEE_TOKEN=
ENDEE_BASE_URL=http://localhost:8080/api/v1
INDEX_NAME=semantic_search

# Embedding Model
EMBEDDING_MODEL=multi-qa-mpnet-base-cos-v1
TOP_K=5

# Groq LLM (RAG)
GROQ_API_KEY=<your-groq-api-key>
GROQ_MODEL=llama-3.1-8b-instant
```

### 4️⃣ Launch the Application

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.

---

## 💡 Usage

| Step | Action | What Happens |
|:----:|:-------|:-------------|
| **1** | Upload PDF(s) via the sidebar | Files are saved locally to `data/` |
| **2** | Click **"Process & Store"** | Text is extracted → chunked → embedded → stored in Endee |
| **3** | Type a question in the search box | Query is embedded and matched against stored vectors |
| **4** | Read the **generated answer** | The LLM synthesizes a structured response from relevant chunks |
| **5** | Expand **"Source Chunks"** | View the original text passages and their similarity scores |

---

## 📁 Project Structure

```
semantic-search/
│
├── app.py                 # Streamlit UI — orchestrates upload, search, and display
├── embed.py               # Embedding generation (Sentence Transformers) + Endee storage
├── search.py              # Vector similarity search against the Endee index
├── rag.py                 # RAG module — Groq LLM answer synthesis from chunks
├── utils.py               # PDF text extraction + section-aware intelligent chunking
│
├── .env                   # Configuration (Endee, model, Groq API key)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── data/                  # Auto-created directory for uploaded PDFs
```

---

## 📋 Requirements

```
endee>=0.1.16
sentence-transformers>=2.2.0
PyPDF2>=3.0.0
streamlit>=1.28.0
python-dotenv>=1.0.0
groq>=1.0.0
```

> **Note:** The Endee SDK requires `numpy>=2.2.4`, which needs **Python 3.10 or higher**.

---

## 🧹 Cleanup

Stop and remove the Endee Docker container when done:

```bash
docker stop endee-server
docker rm endee-server
```

---

<div align="center">

**Built with [Endee](https://endee.io) · [Groq](https://groq.com) · [Sentence Transformers](https://www.sbert.net/) · [Streamlit](https://streamlit.io/)**

</div>
