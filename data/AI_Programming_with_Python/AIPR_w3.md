# Tuần 3: Nền tảng RAG - Document Model, Chunking & Embeddings

## 1. Thông tin chung & Chuẩn đầu ra (Learning Objectives - A3.1 đến A3.4)
- **A3.1**: Thiết kế các mô hình tài liệu: `@dataclass` `Document` và `DocumentChunk` kèm metadata phong phú.
- **A3.2**: Xây dựng kho lưu trữ `DocumentStore` hỗ trợ đầy đủ các thao tác cơ bản: add, get, delete, list_all.
- **A3.3**: Áp dụng các chiến lược phân đoạn văn bản (Fixed-size chunking kèm Overlap, Recursive splitting).
- **A3.4**: Sinh Vector Embeddings ngữ nghĩa và tính toán Cosine Similarity để so khớp độ tương đồng tài liệu.

## 2. Lý thuyết nền tảng & Kiến trúc hệ thống
### 2.1. Chiến lược Chunking văn bản (Text Chunking Strategies)
- **Fixed-size Chunking with Overlap**:
  Cắt văn bản theo độ dài cố định $L$ (ví dụ: 300 - 500 ký tự) kèm khoảng gối đầu $O$ (ví dụ: 50 ký tự). Khoảng overlap đảm bảo các câu hoặc ý nghĩa nằm ở biên giữa hai chunk không bị cắt đứt gãy ngữ cảnh.
- **Recursive Chunking**:
  Tách văn bản dựa trên hệ thống phân cấp dấu phân cách: đoạn văn (`\n\n`), dòng (`\n`), câu (`. `), và từ khóa (` `). Giữ cho các đoạn văn hoàn chỉnh ở cùng một chunk nếu kích thước cho phép.

### 2.2. Vector Embeddings & Độ tương đồng Cosine (Cosine Similarity)
- **Vector Embedding**: Ánh xạ văn bản tự nhiên thành không gian vector nhiều chiều (ví dụ: 768 hoặc 1536 chiều). Các câu có ý nghĩa ngữ nghĩa gần nhau sẽ có khoảng cách vector gần nhau.
- **Cosine Similarity Formula**:
  $$\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \sqrt{\sum_{i=1}^{n} b_i^2}}$$
  - Giá trị bằng $1.0$: Hai vector chỉ cùng hướng hoàn toàn (đồng nghĩa cao nhất).
  - Giá trị bằng $0.0$: Hai vector trực giao (không liên quan ngữ nghĩa).
  - Giá trị bằng $-1.0$: Hai vector ngược hướng hoàn toàn.

## 3. Các bài tập Lab & Mã nguồn mẫu từng bước (Step-by-step Exercises)
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

## 4. Các bẫy thường gặp & Nguyên tắc thiết kế (Gotchas & Best Practices)
- **Bẫy Lost-in-the-Middle**: Đặt chunk size quá lớn khiến LLM bị phân tâm bởi các thông tin râu ria; đặt quá nhỏ khiến chunk bị mất ngữ cảnh của đoạn văn.
- **Bẫy Không chuẩn hóa đường dẫn VectorStore**: Lưu cache vector store bằng đường dẫn tuyệt đối sẽ khiến code không chạy được khi đem sang máy tính khác; bắt buộc dùng đường dẫn tương đối (Portable relative path).

## 5. Nhiệm vụ tự học & Đánh giá năng lực (Independent Labs)
- Xây dựng cơ chế Batch Embedding để gom nhiều chunk trong một lời gọi API, giúp giảm thiểu độ trễ mạng và chi phí API đáng kể.
