from langchain_core.tools import tool
from app.core.rag_pipeline import get_retriever, get_llm
from app.core.reranker import rerank_docs
from app.core.prompts import build_prompt


@tool
def pdf_search(query: str) -> str:
    """Tra cứu thông tin trong Cẩm nang sinh viên (file PDF) của Trường Đại học Ngân hàng TP.HCM.
    Dùng tool này khi câu hỏi liên quan đến quy định, quy chế, hướng dẫn, thông tin chung
    được ghi trong cẩm nang sinh viên.
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)
    docs = rerank_docs(query, docs)

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = build_prompt(context=context, question=query, history="")

    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content