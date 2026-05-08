# RAG-Based AI Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot built using LangChain, ChromaDB, HuggingFace Embeddings, and Google Gemini API.

## Features

- PDF document ingestion
- Semantic chunking
- Vector database with ChromaDB
- Semantic search and retrieval
- Context-aware question answering
- Config-driven architecture

## Tech Stack

- Python
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Google Gemini API
- PyMuPDF

## Project Structure

rag-chatbot/
│
├── app/
├── data/
├── vectorstore/
├── .env
├── main.py
└── requirements.txt

## Usage

- Installation: pip install -r requirements.txt

- Ingest PDF: python main.py - Choose: 1

- Chat with PDF: python main.py - Choose: 2

## Future Improvements

- Conversational memory
- Reranking
- Query classification
- TOC extraction
- SQL Server integration

```bash