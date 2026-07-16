from langchain_core.messages import HumanMessage, AIMessage
from app.config import settings


class SimpleWindowMemory:
    def __init__(self, k: int = settings.MEMORY_K):
        self.k = k
        self.messages = []

    def load_history(self) -> str:
        if not self.messages:
            return "Chưa có lịch sử hội thoại."

        history = ""
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                history += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history += f"Bot: {msg.content}\n"
        return history

    def save_turn(self, question: str, answer: str) -> None:
        self.messages.append(HumanMessage(content=question))
        self.messages.append(AIMessage(content=answer))
        if len(self.messages) > self.k * 2:
            self.messages = self.messages[2:]