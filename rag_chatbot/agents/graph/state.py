from typing import TypedDict


class AgentState(TypedDict):
    question: str
    history: str
    tool_results: str
    response: str