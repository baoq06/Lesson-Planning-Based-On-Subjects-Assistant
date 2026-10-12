import os
import numpy as np
from dotenv import load_dotenv
from src.data_layer.jina_embeddings import jina_embeddings

load_dotenv()


def embed_text(text: str, task: str = "retrieval.passage") -> list[float]:
    """Embed a single text using local Jina helper.
    
    task='retrieval.passage' khi embed document/chunk.
    task='retrieval.query' khi embed câu hỏi tìm kiếm.
    """
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise ValueError("JINA_API_KEY chưa được thiết lập trong biến môi trường (.env)")
    
    resp = jina_embeddings(input=text, model="jina-embeddings-v3", task=task, api_key=api_key)
    return resp.data[0].embedding

def batch_embed_text(
    texts: list[str],
    task: str = "retrieval.passage",
    model: str = "jina-embeddings-v3",
    api_key: str | None = None
) -> list[list[float]]:
    """Gửi một danh sách text để lấy embeddings trong 1 lần gọi API duy nhất."""
    if not texts:
        return []
    if api_key is None:
        api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise ValueError("JINA_API_KEY chưa được thiết lập.")
    resp = jina_embeddings(input=texts, model=model, task=task, api_key=api_key)
    return [item.embedding for item in resp.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))