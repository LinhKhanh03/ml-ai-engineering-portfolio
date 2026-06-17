from langchain.agents import create_agent
from app.core.rag_pipeline import get_llm
from app.tools.pdf_tool import pdf_search
from app.tools.web_tool import hub_announcements
from app.tools.sql_tool import get_sql_tools, SQL_TOOL_PREFIX

AGENT_SYSTEM_PROMPT = f"""Bạn là trợ lý AI của Trường Đại học Ngân hàng TP.HCM (HUB).
Bạn có 3 nhóm công cụ:
1. pdf_search: tra cứu Cẩm nang sinh viên (quy định, quy chế, hướng dẫn).
2. Các công cụ SQL: tra cứu thông tin ngành, chuyên ngành, môn học, số tín chỉ.
3. hub_announcements: lấy thông báo mới nhất từ website hub.edu.vn.

{SQL_TOOL_PREFIX}

Trả lời bằng tiếng Việt, rõ ràng, dựa trên kết quả tool. Nếu không tool nào phù hợp,
hãy trả lời dựa trên hiểu biết chung và nói rõ đây không phải thông tin từ tài liệu/database/web."""


def build_agent():
    llm = get_llm()
    tools = [pdf_search, hub_announcements] + get_sql_tools()
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )
    return agent


def ask_agent(agent, question: str, history: str = "") -> str:
    if history:
        question = f"Lịch sử hội thoại:\n{history}\n\nCâu hỏi hiện tại: {question}"

    result = agent.invoke({
        "messages": [("user", question)]
    })
    return result["messages"][-1].content