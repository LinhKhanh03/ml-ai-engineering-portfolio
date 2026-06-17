from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from app.config.config import SQL_CONNECTION_STRING
from app.core.rag_pipeline import get_llm

SQL_TOOL_PREFIX = """Bạn đang truy vấn database thông tin đào tạo của
Trường Đại học Ngân hàng TP.HCM, gồm 3 bảng:
- nganh(id, ten_nganh)
- chuyen_nganh(id, nganh_id, ten_chuyen_nganh)
- mon_hoc(id, chuyen_nganh_id, ten_mon, so_tin_chi)

Chỉ dùng các bảng này để trả lời câu hỏi về ngành, chuyên ngành, môn học, số tín chỉ."""


def get_sql_tools():
    db = SQLDatabase.from_uri(SQL_CONNECTION_STRING)
    llm = get_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()