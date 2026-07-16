# RAG Chatbot — PDF Question Answering System

A Retrieval-Augmented Generation (RAG) system that enables natural language Q&A over internal PDF documents using bge-m3 embeddings and Gemini 3.5 Flash LLM.

---

## Project Structure

```
RAG_CHATBOT/
├── app/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # Environment variables and constants
│   ├── core/
│   │   ├── __init__.py
│   │   ├── chat.py             # Chat loop and response handling
│   │   ├── get_llm.py          # Gemini LLM initialization
│   │   └── prompts.py          # Prompt template
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py          # Token-aware text splitting
│   │   ├── ingest.py           # Ingestion pipeline
│   │   └── loader.py           # PDF loader
│   └── rag/
│       ├── __init__.py
│       ├── embedding.py        # HuggingFace embedding model
│       ├── retriever.py        # MMR-based retriever
│       └── vectorstore.py      # ChromaDB vectorstore
├── data/
│   └── Cam_nang.pdf            # Source document
├── vectorstore/                # ChromaDB persistent storage
├── venv/
├── .env
├── .env.example
├── main.py
├── README.md
└── requirements.txt
```

---

## Features

- PDF ingestion with PyMuPDF
- Token-aware chunking using HuggingFace tokenizer
- Multilingual embeddings with `BAAI/bge-m3`
- MMR-based retrieval for diverse and relevant context
- Gemini 3.5 Flash for answer generation
- Source page citation in every response

---

## Tech Stack

| Component | Library / Model |
|---|---|
| PDF parsing | PyMuPDF (`fitz`) |
| Text splitting | `langchain-text-splitters` |
| Embedding model | `BAAI/bge-m3` |
| Vector database | ChromaDB (`langchain-chroma`) |
| LLM | Gemini 3.5 Flash (`langchain-google-genai`) |
| Framework | LangChain |
| Tracing & Monitoring | LangSmith |
| GPU support | PyTorch CUDA 11.8 |

---

## Requirements

- Python 3.11.9
- CUDA 11.8 (optional, CPU fallback supported)

Install dependencies:

```bash
pip install -r requirements.txt
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

---

## Environment Variables

Create a `.env` file in the root directory: `Read .env.example`

---

## Usage

**Step 1 — Ingest PDF:**

```bash
python -m main.py
# Choose option 1
```

This will parse the PDF, split it into chunks, embed each chunk, and save to ChromaDB.

**Step 2 — Chat:**

```bash
python main.py
# Choose option 2
```

Type your question and receive an answer with source page references. Type `exit` to quit.

---

## How It Works

```
PDF file
   ↓ PyMuPDF
Raw text per page
   ↓ RecursiveCharacterTextSplitter (token-aware)
Chunks (700 tokens, 105 overlap)
   ↓ BAAI/bge-m3
Vector embeddings (1024 dimensions)
   ↓ ChromaDB
Persistent vectorstore

User question
   ↓ embed question
   ↓ MMR retrieval (fetch 15, return 5)
Relevant chunks
   ↓ build prompt
   ↓ Gemini 3.5 Flash
Answer + Source pages
```

---

## Configuration

All parameters are defined in `app/config/settings.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace embedding model |
| `LLM_MODEL` | `gemini-3.5-flash` | Gemini model name |
| `CHUNK_SIZE` | `700` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `150` | Overlapping tokens between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `TEMPERATURE` | `0.1` | LLM temperature |
| `MAX_OUTPUT_TOKENS` | `4096` | Max tokens in LLM response |

## Tracing with LangSmith

This project integrates LangSmith for tracing and performance monitoring.
Every retrieval and LLM call is automatically logged to your LangSmith project.

To enable tracing, add the following to your `.env` file: `Read .env.example`

Access your traces at https://smith.langchain.com