from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    EMBEDDING_MODEL
)

def token_len(text):
    return len(tokenizer.encode(text))

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_len
    )
    chunks = splitter.split_documents(documents)
    return chunks