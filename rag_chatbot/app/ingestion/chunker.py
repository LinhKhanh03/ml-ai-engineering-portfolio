from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config.settings import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from langchain_core.documents import Document

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    EMBEDDING_MODEL
)

def token_len(text) -> int:
    return len(tokenizer.encode(text))

def split_text(documents) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_len,
        separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
        ""]
    )
    chunks = splitter.split_documents(documents)
    return chunks