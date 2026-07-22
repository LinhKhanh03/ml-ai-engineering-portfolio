from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from app.core.get_llm import get_llm
from app.config import settings

assert settings.DATABASE_URL is not None, "DATABASE_URL is not set in .env"

engine = create_engine(settings.DATABASE_URL)
llm = get_llm()

TABLE_DESCRIPTIONS = {
    "customers": "khách hàng",
    "orders": "đơn hàng",
    "order_items": "chi tiết đơn hàng",
    "order_payments": "thanh toán đơn hàng",
    "order_reviews": "đánh giá đơn hàng",
    "products": "sản phẩm",
    "sellers": "người bán",
    "geolocation": "vị trí địa lý",
    "product_category_name_translation": "danh mục sản phẩm"
}


def get_schema() -> str:
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """))
        rows = result.fetchall()

    schema = ""
    current_table = ""
    for row in rows:
        table_name, column_name, data_type = row
        if table_name != current_table:
            description = TABLE_DESCRIPTIONS.get(table_name, "")
            schema += f"\nTable: {table_name} ({description})\n"
            current_table = table_name
        schema += f"  - {column_name} ({data_type})\n"
    return schema


def run_query(sql: str) -> str:
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = result.fetchall()
        columns = list(result.keys())

    if not rows:
        return "Không có kết quả."

    output = " | ".join(columns) + "\n"
    output += "-" * 40 + "\n"
    for row in rows:
        output += " | ".join(str(value) for value in row) + "\n"
    return output


@tool
def search_sql(question: str) -> str:
    """Truy vấn dữ liệu thương mại điện tử Olist từ PostgreSQL.
    Dùng khi câu hỏi liên quan đến đơn hàng, doanh thu, sản phẩm, khách hàng, người bán."""
    schema = get_schema()

    prompt = f"""Dựa vào schema sau (tên bảng kèm mô tả tiếng Việt):
{schema}

Câu hỏi có thể bằng tiếng Việt hoặc tiếng Anh.
Hãy tự map từ ngữ tiếng Việt sang tên bảng và cột tiếng Anh tương ứng trong schema.
Viết câu SQL PostgreSQL để trả lời câu hỏi: {question}

Chỉ trả về câu SQL thuần túy, không giải thích, không markdown."""

    sql_response = llm.invoke(prompt)

    content = sql_response.content
    if isinstance(content, list) and len(content) > 0:
        first = content[0]
        sql_text = first["text"] if isinstance(first, dict) and "text" in first else str(first)
    else:
        sql_text = str(content)

    result = run_query(sql_text)
    return result