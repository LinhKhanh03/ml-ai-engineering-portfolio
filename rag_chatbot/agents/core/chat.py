from app.core.memory import SimpleWindowMemory
from agents.graph.graph import build_graph


def chat_agent():
    graph = build_graph()
    memory = SimpleWindowMemory()

    while True:
        question = input("\nAsk: ")
        if question.lower() == "exit":
            break

        history = memory.load_history()

        result = graph.invoke({
            "question": question,
            "history": history,
            "tool_results": "",
            "response": ""
        })

        response_text = result["response"]
        memory.save_turn(question, response_text)

        print("\nAnswer:")
        print(response_text)