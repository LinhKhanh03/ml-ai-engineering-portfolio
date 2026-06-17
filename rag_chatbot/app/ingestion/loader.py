import fitz
from langchain_core.documents import Document
from app.config.config import PDF_FILE

def load_pdf():
    docs = []
    
    with fitz.open(PDF_FILE) as pdf:
        for i, page in enumerate(pdf):
            text = page.get_text()
            if not text.strip():
                continue
            
            docs.append(
                Document(
                    page_content=text,
                    metadata={"page": i + 1}
                )
            )
    return docs 