from app.core.prompts import prompt
from app.rag.retriever import get_retriever
from app.core.get_llm import get_llm

from app.rag.reranker import reranker
from app.core.memory import SimpleWindowMemory

def chat():
    retriever = get_retriever()
    llm = get_llm()

    memory = SimpleWindowMemory()

    while True:
        question = input("\nAsk: ")
        if question.lower() == "exit":
            break
            
        history = memory.load_history()

        docs = retriever.invoke(question)
        docs_rerank = reranker(question, docs)

        context = ""
        sources = []

        for i, doc in enumerate(docs_rerank):
            content = doc.page_content
            page = doc.metadata.get("page", "N/A")
            context += f"{content}\n"
            sources.append(f"Document {i+1} - Page {page}")

        prom = prompt(context, question, history)
        response = llm.invoke(prom)
        
        content = response.content

        if isinstance(content, list):
            response_text = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
        else:
            response_text = content

        memory.save_turn(question, response_text)    

        print("\nAnswer:")
        print(response_text)
        if response_text == "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp.":
            continue
        else:
            print("\nSources:")
            for s in sources:
                print("-", s)