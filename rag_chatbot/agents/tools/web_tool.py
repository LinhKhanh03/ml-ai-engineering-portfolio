from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from app.config import settings

tavily = TavilySearch(
    max_results=3,
    topic="general",
    tavily_api_key=settings.TAVILY_API_KEY
)


@tool
def search_web(question: str) -> str:
    """Tìm kiếm thông tin mới nhất trên internet.
    Dùng khi câu hỏi về tin tức, xu hướng, hoặc thông tin ngoài tài liệu nội bộ."""
    result = tavily.invoke(question)
    return result