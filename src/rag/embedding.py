import os
import numpy as np
from dotenv import load_dotenv
from src.rag.jina_embeddings import jina_embeddings

load_dotenv()

def embed_text(text:str, model="jina-embeddings-v3", api_key=None) -> list[float]:
    if api_key is None:
        api_key = os.getenv("JINA_API_KEY")
    resp = jina_embeddings(input=text, model=model, api_key=api_key)
    return resp.data[0].embedding
def batch_embed_text(input:list[str], model="jina-embeddings-v3", api_key=None) -> list[list[float]]:
    if api_key is None:
        api_key = os.getenv("JINA_API_KEY")
    resp = jina_embeddings(input=input, model=model, api_key=api_key)
    return [item.embedding for item in resp.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))