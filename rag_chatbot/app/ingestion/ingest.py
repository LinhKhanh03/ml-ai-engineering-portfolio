from app.ingestion.loader import load_pdf
from app.ingestion.chunker import split_text
from app.rag.vectorstore import create_vectorstore

def ingest():
    documents = load_pdf()
    print(f"Loaded {len(documents)} documents")

    chunks = split_text(documents)
    print(f"Created {len(chunks)} chunks")

    create_vectorstore(chunks)
    print("Vector DB created successfully!")