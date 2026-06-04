import os, torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import (
    HF_TOKEN,
    EMBEDDING_MODEL,
    VECTOR_DB_DIR,
    TOP_K,
    LLM_MODEL,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS)


def get_embedding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"token": HF_TOKEN,
                      "device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embedding

def create_vectorstore(chunks):
    embedding = get_embedding()
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=VECTOR_DB_DIR
    )
    return db

def load_vectorstore():
    embedding = get_embedding()
    db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embedding
    )
    return db

def get_retriever():
    db = load_vectorstore()
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K,
                       "fetch_k": TOP_K * 3,
                       "lambda_mult": 0.7}
    )
    return retriever

def get_llm():
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS
    )
    return llm