# RAG_CHATBOT — Multi-source RAG Agent

Chatbot RAG cho Trường Đại học Ngân hàng TP.HCM (HUB), mở rộng từ pipeline RAG đơn (PDF Cẩm nang sinh viên) thành agent đa nguồn dữ liệu sử dụng LangGraph: PDF, SQL Server, và web scraping trang thông báo của trường.

## Tính năng

- Trả lời câu hỏi dựa trên Cẩm nang sinh viên (PDF) qua pipeline RAG với reranker.
- Tra cứu thông tin ngành, chuyên ngành, môn học, số tín chỉ từ SQL Server bằng ngôn ngữ tự nhiên.
- Lấy thông báo mới nhất từ hub.edu.vn/thong-bao.
- Agent tự động chọn nguồn dữ liệu phù hợp dựa trên câu hỏi (LangChain `create_agent`).

## Kiến trúc thư mục

```
RAG_CHATBOT/
├── app/
│   ├── config/
│   │   └── settings.py        # cấu hình chung: model, path, SQL, web
│   ├── core/
│   │   ├── rag_pipeline.py     # embedding, vectorstore, retriever, LLM
│   │   ├── reranker.py         # CrossEncoder reranker
│   │   ├── memory.py           # quản lý lịch sử hội thoại
│   │   ├── prompts.py          # prompt template cho RAG
│   │   └── chat.py             # build chat agent   
│   ├── ingestion/
│   │   ├── loader.py           # đọc PDF
│   │   ├── chunker.py          # chia chunk văn bản
│   │   └── ingest.py           # build vectorstore từ PDF
│   ├── tools/
│   │   ├── pdf_tool.py         # tool tra cứu PDF
│   │   ├── sql_tool.py         # tool truy vấn SQL Server
│   │   └── web_tool.py         # tool scrape hub.edu.vn
│   └── graph.py                # agent LangGraph gộp 3 tool
├── db/
│   └── seed_data.sql           # schema + dữ liệu mẫu SQL Server
├── data/
│   └── Cam_nang.pdf            # tài liệu nguồn
├── vectorstore/                 # ChromaDB, sinh ra sau khi ingest
├── main.py                      # entrypoint: ingest hoặc chat agent
├── requirements.txt
├── .gitignore
└── .env                          # biến môi trường, không commit
```

## Yêu cầu hệ thống

- Python 3.10.11
- GPU NVIDIA hỗ trợ CUDA 11.8 (không bắt buộc, có thể chạy CPU)
- SQL Server đã cài sẵn, có driver **ODBC Driver 18 for SQL Server**
- Tài khoản Google AI Studio (lấy `GOOGLE_API_KEY` cho Gemini)
- (Tuỳ chọn) HuggingFace token nếu model embedding yêu cầu xác thực

## Hướng dẫn cài đặt (Windows, VS Code terminal)

### 1. Clone hoặc mở project trong VS Code

Mở thư mục `RAG_CHATBOT` bằng VS Code, mở terminal tích hợp (`Ctrl + ~`), đảm bảo terminal đang chạy ở chế độ **cmd** (không phải PowerShell, để tránh khác cú pháp activate venv).

### 2. Tạo virtual environment

```cmd
python -m venv venv
```

### 3. Activate virtual environment

```cmd
venv\Scripts\activate
```

Sau khi activate, dòng lệnh sẽ hiện `(venv)` ở đầu.

### 4. Cài đặt thư viện

```cmd
pip install -r requirements.txt
```

Nếu gặp lỗi liên quan package legacy (ví dụ `langchain.memory` không tìm thấy), cài thêm:

```cmd
pip install langchain-classic==1.0.8
```

### 5. Tạo file `.env`

Tạo file `.env` ở thư mục gốc project với nội dung:

```
GOOGLE_API_KEY=your_google_api_key
HF_TOKEN=your_huggingface_token

SQL_SERVER=localhost,1433
SQL_DATABASE=hub_demo
SQL_USER=sa
SQL_PASSWORD=your_password
```

### 6. Tạo database và bảng mẫu trong SQL Server

Mở SQL Server Management Studio (SSMS) hoặc Azure Data Studio, tạo database (ví dụ `hub_demo`), sau đó chạy nội dung file `db/seed_data.sql` trong database đó để tạo 3 bảng `nganh`, `chuyen_nganh`, `mon_hoc` cùng dữ liệu mẫu.

### 7. Đặt file PDF nguồn

Đảm bảo file `data/Cam_nang.pdf` đã có trong thư mục `data/`. Nếu chưa, tạo thư mục và copy file vào:

```cmd
mkdir data
copy đường_dẫn_tới_file\Cam_nang.pdf data\Cam_nang.pdf
```

## Chạy chương trình

Toàn bộ chương trình chạy qua `main.py`:

```cmd
python -m main
```

Sau khi chạy, chương trình hiện menu lựa chọn:

```
1. Ingest dữ liệu (xây dựng vectorstore từ PDF)
2. Chat với AI Agent (PDF + SQL Server + Web)
```

### Lựa chọn 1 — Ingest

Chạy lựa chọn này **trước tiên, chỉ cần 1 lần** (hoặc mỗi khi đổi file PDF nguồn). Quá trình này đọc PDF, chia chunk, sinh embedding, và lưu vào `vectorstore/`.

### Lựa chọn 2 — Chat với AI Agent

Sau khi đã ingest, chọn mục này để bắt đầu hội thoại. Agent sẽ tự quyết định dùng tool PDF, SQL, hoặc web tuỳ theo câu hỏi. Gõ `exit` để thoát.

Ví dụ câu hỏi thử nghiệm:

```
- Quy định về học phí trong cẩm nang là gì?          (PDF)
- Ngành Hệ thống thông tin quản lý có chuyên ngành nào?  (SQL)
- Thông báo mới nhất của trường là gì?                (Web)
```

## Xử lý lỗi thường gặp

**CUDA out of memory**: nếu GPU dưới 6GB VRAM, mở `app/core/reranker.py` và `app/core/rag_pipeline.py`, đổi biến `device` thành `"cpu"` để chạy embedding/reranker trên CPU, dành VRAM cho các tác vụ khác.

**Lỗi kết nối SQL Server**: kiểm tra `SQL_SERVER`, `SQL_USER`, `SQL_PASSWORD` trong `.env`, và đảm bảo SQL Server cho phép xác thực SQL Server Authentication (không chỉ Windows Authentication), cùng việc đã cài đúng ODBC Driver 18.

**Web scraping trả rỗng**: cấu trúc HTML của hub.edu.vn/thong-bao có thể thay đổi theo thời gian; kiểm tra lại selector trong `app/tools/web_tool.py`.

## Công nghệ sử dụng

- **LangChain / LangGraph** — orchestration, agent, tool-calling
- **ChromaDB** — vector store
- **HuggingFace `intfloat/multilingual-e5-base`** — embedding đa ngôn ngữ
- **BAAI/bge-reranker-v2-m3** — reranking kết quả truy xuất
- **Google Gemini 2.5 Flash** — LLM sinh câu trả lời
- **SQL Server + SQLDatabaseToolkit** — truy vấn dữ liệu có cấu trúc bằng ngôn ngữ tự nhiên
- **BeautifulSoup4 + Requests** — web scraping
- **PyMuPDF** — đọc và trích xuất nội dung PDF

## Ghi chú phát triển

Đây là phiên bản demo nhằm minh hoạ khả năng kết hợp nhiều nguồn dữ liệu trong một agent RAG. Hướng phát triển tiếp theo có thể gồm: thêm UI (Streamlit/FastAPI), mở rộng schema SQL, bổ sung cache có TTL cho web scraping, và đánh giá chất lượng câu trả lời bằng RAGAS.