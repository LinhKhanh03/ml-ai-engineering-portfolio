from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from app.config import settings

rerank_model = CrossEncoder(settings.RERANK_MODEL, device='cpu')

def reranker(question: str, docs: list[Document], top_n = settings.TOP_N) -> list[Document]:
    pairs = []
    for document in docs:
        pairs.append((question, document.page_content))

    scores = rerank_model.predict(pairs)

    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    top_docs = []
    for score, document in scored_docs[:top_n]:
        top_docs.append(document)

    return top_docs
