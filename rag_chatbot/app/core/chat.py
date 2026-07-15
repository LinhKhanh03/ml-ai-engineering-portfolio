from app.core.rag_pipeline import (
    get_retriever,
    get_llm)
from app.core.prompts import prompt

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

        prom = prompt(context, question)
        response = llm.invoke(prom)
        
        content = response.content

        if isinstance(content, list):
            response_text = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
        else:
            response_text = content

        print("\nAnswer:")
        print(response_text)

        print("\nSources:")
        for s in sources:
            print("-", s)