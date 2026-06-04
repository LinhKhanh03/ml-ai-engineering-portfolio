import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectorstore")
PDF_FILE = os.path.join(DATA_DIR, "Cam_nang.pdf")

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
LLM_MODEL = "gemini-2.5-flash"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
TOP_K = 5
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 4096