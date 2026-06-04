from app.rag_pipeline import get_retriever, get_llm
from app.prompts import build_prompt
from app.memory import create_memory, load_history, save_turn
from app.reranker import rerank_docs


def chat():
    retriever = get_retriever()
    llm = get_llm()
    memory = create_memory(k=5)

    while True:
        question = input("\nAsk: ")
        if question.lower() == "exit":
            break

        history = load_history(memory)

        docs_before = retriever.invoke(question)
        docs_after = rerank_docs(question, docs_before, top_n=3)

        context = ""
        sources = []

        for i, doc in enumerate(docs_after):
            content = doc.page_content
            page = doc.metadata.get("page", "N/A")
            context += f"\n[Document {i+1} | Page {page}]\n{content}\n"
            sources.append(f"Document {i+1} - Page {page}")

        prompt = build_prompt(context, question, history)
        response = llm.invoke(prompt)

        save_turn(memory, question, response.content)

        print("\nAnswer:")
        print(response.content)

        print("\nSources:")
        for s in sources:
            print("-", s)