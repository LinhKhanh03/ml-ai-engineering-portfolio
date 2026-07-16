from langchain_huggingface import HuggingFaceEmbeddings
from app.config import config
import torch

def get_embedding() -> HuggingFaceEmbeddings:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding = HuggingFaceEmbeddings(
        model_name = config.EMBEDDING_MODEL,
        model_kwargs = {"device": device,
                        "token": config.HF_TOKEN},
        encode_kwargs = {"normalize_embeddings": True}
    )
    return embedding

