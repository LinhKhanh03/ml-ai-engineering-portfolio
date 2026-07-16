from app.rag.vectorstore import load_vectorstore
from app.config import settings
from langchain_core.vectorstores import VectorStoreRetriever

def get_retriever() -> VectorStoreRetriever:
    db = load_vectorstore()
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.TOP_K,
                       "fetch_k": settings.TOP_K * 3,
                       "lambda_mult": 0.7}
    )
    return retriever