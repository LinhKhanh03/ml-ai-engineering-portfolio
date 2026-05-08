from app.rag_pipeline import (
    get_retriever,
    get_llm)


def chat():
    retriever = get_retriever()
    llm = get_llm()

    while True:
        question = input("\nAsk: ")
        if question.lower() == "exit":
            break

        docs = retriever.invoke(question)
        context = ""
        sources = []

        for i, doc in enumerate(docs):
            content = doc.page_content
            page = doc.metadata.get("page", "N/A")
            context += f"\n[Document {i+1} | Page {page}]\n{content}\n"
            sources.append(f"Document {i+1} - Page {page}")

        prompt = f"""
Bạn là AI assistant.
Hãy trả lời dựa trên context dưới đây.
Context:
{context}
Question:
{question}
"""
        response = llm.invoke(prompt)

        print("\nAnswer:")
        print(response.content)

        print("\nSources:")
        for s in sources:
            print("-", s)