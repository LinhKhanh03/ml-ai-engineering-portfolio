from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from app.core.get_llm import get_llm
from app.core.prompts import prompt
from agents.graph.state import AgentState
from agents.tools.pdf_tool import search_pdf
from agents.tools.sql_tool import search_sql
from agents.tools.web_tool import search_web

llm = get_llm()
tools = [search_pdf, search_sql, search_web]
llm_with_tools = llm.bind_tools(tools)
parser = StrOutputParser()


def router_node(state: AgentState) -> AgentState:
    question = state["question"]
    history = state["history"]

    response = llm_with_tools.invoke(f"""
Lịch sử hội thoại:
{history}

Câu hỏi: {question}

Hướng dẫn chọn tool:
- search_pdf: câu hỏi về tài liệu nội bộ, giới thiệu, lưu ý, quy chế, quy định, nội quy, hướng dẫn, hỗ trợ, thông tin tham khảo
- search_sql: câu hỏi về dữ liệu thương mại, đơn hàng, khách hàng, doanh thu, khu vực, sản phẩm, người bán
- search_web: câu hỏi về tin tức, xu hướng, thông tin bên ngoài tài liệu

Nếu câu hỏi liên quan đến nhiều nguồn, hãy gọi tất cả tool cần thiết.
""")

    tool_results = ""

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "search_pdf":
                result = search_pdf.invoke(tool_args)
            elif tool_name == "search_sql":
                result = search_sql.invoke(tool_args)
            elif tool_name == "search_web":
                result = search_web.invoke(tool_args)
            else:
                result = ""

            tool_results += f"\n[{tool_name}]\n{result}\n"
    else:
        content = response.content
        if isinstance(content, list) and len(content) > 0:
            first = content[0]
            tool_results = first["text"] if isinstance(first, dict) and "text" in first else str(first)
        else:
            tool_results = str(content)

    return {
        "question": state["question"],
        "history": state["history"],
        "tool_results": tool_results,
        "response": ""
    }


def synthesizer_node(state: AgentState) -> AgentState:
    prom = prompt(
        context=state["tool_results"],
        question=state["question"],
        history=state["history"]
    )

    response = llm.invoke(prom)
    response_text = parser.invoke(response)

    return {
        "question": state["question"],
        "history": state["history"],
        "tool_results": state["tool_results"],
        "response": response_text
    }


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()