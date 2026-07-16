from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings
import torch

def get_embedding() -> HuggingFaceEmbeddings:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding = HuggingFaceEmbeddings(
        model_name = settings.EMBEDDING_MODEL,
        model_kwargs = {"device": device,
                        "token": settings.HF_TOKEN},
        encode_kwargs = {"normalize_embeddings": True}
    )
    return embedding

