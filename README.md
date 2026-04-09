# 📄 RAG Project — PDF Question-Answering Application using LLM 

A production-style **Retrieval-Augmented Generation (RAG)** application that lets you upload PDF documents and ask natural-language questions about their content. Answers are grounded in the actual document text, not hallucinated by the LLM.

---

## 🎬 Demo

https://raw.githubusercontent.com/GnanaJeyam/rag_project/main/Rag_Walkthrough.mp4

---

## 📤 Uploading a PDF

```
┌──────────────┐  save file   ┌────────────┐   event   ┌───────────────┐
│  Streamlit UI│ ───────────▶ │  uploads/  │ ────────▶ │  Inngest (bg) │
│  (upload PDF)│              └────────────┘           │  rag_ingest   │
└──────────────┘                                       └───────┬───────┘
                                                               │
                                                    ┌──────────▼──────────┐
                                                    │  1. Parse PDF text  │
                                                    │     (PDFReader)     │
                                                    ├─────────────────────┤
                                                    │  2. Chunk text      │
                                                    │  (SentenceSplitter) │
                                                    ├─────────────────────┤
                                                    │  3. Embed chunks    │
                                                    │     (FastEmbed)     │
                                                    ├─────────────────────┤
                                                    │  4. Store vectors   │
                                                    │     (Qdrant)        │
                                                    └─────────────────────┘
```

1. **Upload** — User uploads a PDF via the Streamlit UI.
2. **Save** — The file is saved to the `uploads/` directory.
3. **Event** — A `rag/pdf_uploaded` event is fired to Inngest.
4. **Parse** — `PDFReader` (LlamaIndex) extracts raw text from each page.
5. **Chunk** — `SentenceSplitter` (LlamaIndex) splits text into overlapping 800-token chunks (200-token overlap).
6. **Embed** — `FastEmbed` (`BAAI/bge-small-en-v1.5`, 384-dim) converts each chunk into a vector embedding.
7. **Store** — Embeddings and their respective text payloads are upserted into a **Qdrant** vector collection.

---

## ❓ Answering Your Question

```
┌──────────────┐   event   ┌───────────────┐  query   ┌────────┐
│  Streamlit UI│ ────────▶ │  Inngest (bg) │ ───────▶ │ Qdrant │
│  (ask query) │           │  rag_query    │          │  (vec) │
└──────┬───────┘           └───────┬───────┘          └───┬────┘
       │                           │                      │
       │                           │  ◀── top-5 chunks ───┘
       │                           ▼
       │                   ┌──────────────┐
       │                   │ Gemini 2.5   │
       │                   │ Flash (LLM)  │
       │                   └──────┬───────┘
       │                          │
       │    ◀── grounded answer ──┘
       ▼
┌──────────────┐
│  Display     │
│  answer +    │
│  sources     │
└──────────────┘
```

1. **Question** — User types a question in the Streamlit UI and clicks Submit.
2. **Event** — A `rag/query_data` event is fired to Inngest.
3. **Embed** — The question is embedded with the same FastEmbed model used during ingestion.
4. **Retrieve** — Qdrant returns the top-5 most similar chunks (cosine similarity ≥ 0.45).
5. **Generate** — Retrieved chunks are sent as context to **Google Gemini 2.5 Flash**, which produces a grounded answer.
6. **Display** — The answer and source documents are shown in the UI.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit | Interactive web UI for upload & Q&A |
| **Orchestration** | Inngest + FastAPI | Event-driven background functions with retries |
| **PDF Parsing** | LlamaIndex (`PDFReader`, `SentenceSplitter`) | Text extraction & chunking |
| **Embeddings** | FastEmbed (`BAAI/bge-small-en-v1.5`) | Local, fast 384-dim text embeddings |
| **Vector Store** | Qdrant | Similarity search over document vectors |
| **LLM** | Google Gemini 2.5 Flash | Context-grounded answer generation |
| **Runtime** | Python 3.13, uv | Package & dependency management |

---

## Project Structure

```
rag_project/
├── streamlit_ui.py      # Streamlit frontend — file upload & question UI
├── inngest_handler.py   # Inngest functions — rag_ingest & rag_query (FastAPI server)
├── pdf_util.py          # PDF loading, chunking, embedding, and saving utilities
├── qdrant_helper.py     # Qdrant collection management, insert & search operations
├── pyproject.toml       # Project metadata & dependencies (uv / pip)
```

---

## Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **Qdrant** — running locally on port `6333`
  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```
- **Inngest Dev Server** — running locally on port `8288`
  ```bash
  npx inngest-cli@latest dev
  ```
- **Google Gemini API Key** — set in a `.env` file

---

## Setup & Installation

```bash
# Clone the repo
git clone <repo-url> && cd rag_project

# Install dependencies with uv
uv sync

# Create a .env file
echo "GOOGLE_API_KEY=your-gemini-api-key-here" > .env
```

---

## Running the Application

You need **four services** running simultaneously:

### 1. Qdrant Vector Database

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Inngest Dev Server

**Option A — via npx:**

```bash
npx inngest-cli@latest dev
```

**Option B — via Docker:**

```bash
docker run -p 8288:8288 inngest/inngest \
  inngest dev -u http://host.docker.internal:8000/api/inngest --no-discovery
```

### 3. Inngest + FastAPI backend

```bash
uv run uvicorn inngest_handler:app --reload --port 8000
```

### 4. Streamlit UI

```bash
uv run streamlit run streamlit_ui.py
```

Then open **http://localhost:8501** in your browser.

---

## Usage

1. **Upload a PDF** — drag or browse a PDF file in the upload widget.
2. **Wait for ingestion** — the spinner indicates the PDF is being parsed, chunked, embedded, and stored.
3. **Ask a question** — type a question about any uploaded document and click **Submit**.
4. **Read the answer** — the LLM-generated answer and source documents are displayed below.

---

## Key Design Decisions

- **LlamaIndex for parsing only** — LlamaIndex's `PDFReader` and `SentenceSplitter` are used purely for robust PDF text extraction and intelligent chunking. The LLM orchestration is handled separately via Inngest, giving full control over the RAG pipeline.
- **FastEmbed over SentenceTransformers** — FastEmbed is lightweight and runs locally without PyTorch, producing 384-dim embeddings efficiently.
- **Inngest for orchestration** — Event-driven background functions with built-in retries, step functions, and observability. Decouples the UI from heavy processing.
- **Qdrant for vector search** — Purpose-built vector database with cosine similarity, payload filtering, and local or cloud deployment options.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Google Gemini API key | *(required)* |
| `INNGEST_API_BASE` | Inngest dev server URL | `http://127.0.0.1:8288/v1` |

---

## License

This project is for personal/educational use.


