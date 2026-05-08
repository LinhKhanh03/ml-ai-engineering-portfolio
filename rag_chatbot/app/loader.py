import fitz
from langchain.schema import Document
from app.config import PDF_FILE


def load_pdf():
    docs = []
    pdf = fitz.open(PDF_FILE)

    for i, page in enumerate(pdf):
        text = page.get_text()
        docs.append(
            Document(
                page_content=text,
                metadata={"page": i + 1}
            )
        )
    return docs