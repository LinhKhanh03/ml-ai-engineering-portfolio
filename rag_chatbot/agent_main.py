from app.graph import build_agent, ask_agent
from app.core.memory import create_memory, load_history, save_turn


def main():
    print("=== Multi-source RAG Agent - HUB ===")
    print("Gõ 'exit' để thoát.\n")

    agent = build_agent()
    memory = create_memory()

    while True:
        question = input("Bạn: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        history = load_history(memory)
        answer = ask_agent(agent, question, history)

        print(f"\nBot: {answer}\n")
        save_turn(memory, question, answer)


if __name__ == "__main__":
    main()