from langchain.memory import ConversationBufferWindowMemory

def create_memory(k: int = 5) -> ConversationBufferWindowMemory:
    return ConversationBufferWindowMemory(
        k=k,
        human_prefix="User",
        ai_prefix="Bot",
        return_messages=False
    )

def load_history(memory: ConversationBufferWindowMemory) -> str:
    history = memory.load_memory_variables({})["history"]
    return history if history else "Chưa có lịch sử hội thoại."

def save_turn(memory: ConversationBufferWindowMemory, question: str, answer: str) -> None:
    memory.save_context(
        {"input": question},
        {"output": answer}
    )