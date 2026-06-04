from sentence_transformers import CrossEncoder
from langchain.schema import Document
from app.config import RERANK_MODEL
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
rerank_model = CrossEncoder(RERANK_MODEL, device=device)

def rerank_docs(question: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    pairs = [(question, doc.page_content) for doc in docs]
    scores = rerank_model.predict(pairs)

    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_n]] 
