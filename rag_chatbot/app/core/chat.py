from app.graph import build_agent, ask_agent
from app.core.memory import create_memory, load_history, save_turn

def chat():
    print("*Multi-source RAG Agent - HUB*")
    print("Gõ 'exit' để thoát.\n")

    agent = build_agent()
    memory = create_memory()

    while True:
        question = input("CLIENT: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        history = load_history(memory)
        answer = ask_agent(agent, question, history)

        print(f"\nBOT: {answer}\n")
        save_turn(memory, question, answer)