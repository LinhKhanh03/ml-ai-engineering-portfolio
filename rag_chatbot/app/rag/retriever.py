from app.rag.vectorstore import load_vectorstore
from app.config import config
from langchain_core.vectorstores import VectorStoreRetriever

def get_retriever() -> VectorStoreRetriever:
    db = load_vectorstore()
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.TOP_K,
                       "fetch_k": config.TOP_K * 3,
                       "lambda_mult": 0.7}
    )
    return retriever