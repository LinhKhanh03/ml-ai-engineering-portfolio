from app.rag.embedding import get_embedding
from langchain_chroma import Chroma
from app.config import settings
from langchain_core.documents import Document

def create_vectorstore(chunks: list[Document]) ->Chroma:
    embedding = get_embedding()
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=settings.VECTOR_DB_DIR
    )
    return db

def load_vectorstore() -> Chroma:
    embedding = get_embedding()
    db = Chroma(
        embedding_function=embedding,
        persist_directory=settings.VECTOR_DB_DIR
    )
    return db