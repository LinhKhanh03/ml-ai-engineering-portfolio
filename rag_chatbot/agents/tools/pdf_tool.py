from langchain_core.tools import tool
from app.rag.retriever import get_retriever
from app.rag.reranker import reranker

retriever = get_retriever()


@tool
def search_pdf(question: str) -> str:
    """Tìm kiếm thông tin trong tài liệu PDF nội bộ của trường."""
    docs = retriever.invoke(question)
    docs_rerank = reranker(question, docs, top_n=3)
    return "\n".join([doc.page_content for doc in docs_rerank])