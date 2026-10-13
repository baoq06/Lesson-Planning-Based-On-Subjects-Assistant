# Week 3: RAG Foundations - Document Model, Chunking & Embeddings

## 1. General Information & Learning Objectives (A3.1 to A3.4)
- **A3.1**: Design the document models: `@dataclass` `Document` and `DocumentChunk` with rich metadata.
- **A3.2**: Build a `DocumentStore` that supports the basic operations: add, get, delete, list_all.
- **A3.3**: Apply text segmentation strategies (Fixed-size chunking with Overlap, Recursive splitting).
- **A3.4**: Generate semantic Vector Embeddings and compute Cosine Similarity to match document similarity.

## 2. Foundational Theory & System Architecture
### 2.1. Text Chunking Strategies
- **Fixed-size Chunking with Overlap**:
  Cut the text at a fixed length $L$ (for example 300 - 500 characters) with an overlap $O$ (for example 50 characters). The overlap ensures that sentences or ideas that fall on the boundary between two chunks do not lose their context.
- **Recursive Chunking**:
  Split the text using a hierarchy of separators: paragraph (`\n\n`), line (`\n`), sentence (`. `) and word (` `). Keeps complete paragraphs in the same chunk if the size allows.

### 2.2. Vector Embeddings & Cosine Similarity
- **Vector Embedding**: Maps natural text into a multi-dimensional vector space (for example 768 or 1536 dimensions). Sentences with close semantic meaning have close vectors.
- **Cosine Similarity Formula**:
  $$\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \sqrt{\sum_{i=1}^{n} b_i^2}}$$
  - Value $1.0$: the two vectors point in exactly the same direction (highest semantic similarity).
  - Value $0.0$: the two vectors are orthogonal (semantically unrelated).
  - Value $-1.0$: the two vectors point in exactly opposite directions.

## 3. Lab Exercises & Step-by-step Sample Code
### Exercise 3.1 — Document & Chunk Models (doc_model.py)
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class Document:
    content: str
    doc_id: str = ""
    title: str = ""
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    embedding: Optional[list[float]] = None
```

### Exercise 3.2 — Sliding Window Chunking (recursive_chunking.py)
```python
def chunk_document(doc: Document, chunk_size: int = 300, overlap: int = 50) -> list[DocumentChunk]:
    chunks = []
    text = doc.content
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(DocumentChunk(
            chunk_id=f"{doc.doc_id}_c{idx:03d}",
            doc_id=doc.doc_id,
            content=chunk_text,
            chunk_index=idx
        ))
        start += (chunk_size - overlap)
        idx += 1
    return chunks
```

### Exercise 3.3 — VectorStore & In-Memory Retrieval (lab3_vector_store.py)
```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    dot = np.dot(va, vb)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(dot / norm) if norm > 0 else 0.0

class VectorStore:
    def __init__(self):
        self.chunks: dict[str, DocumentChunk] = {}

    def search(self, query_vec: list[float], top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
        scores = []
        for c in self.chunks.values():
            if c.embedding:
                score = cosine_similarity(query_vec, c.embedding)
                scores.append((c, score))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
```

## 4. Common Pitfalls & Design Principles (Gotchas & Best Practices)
- **Lost-in-the-Middle pitfall**: A chunk size that is too large distracts the LLM with peripheral information; a chunk size that is too small makes the chunk lose the context of its paragraph.
- **Non-portable VectorStore path pitfall**: Caching the vector store with an absolute path makes the code fail on another computer; always use a relative path (Portable relative path).

## 5. Independent Tasks & Competency Assessment (Independent Labs)
- Build a Batch Embedding mechanism that groups many chunks into a single API call, which significantly reduces network latency and API cost.
