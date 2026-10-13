from typing import Optional
from rank_bm25 import BM25Okapi
from src.schemas import DocumentChunk, RetrievalResult
from src.rag.embedding import embed_text, cosine_similarity
from src.rag.vector_store import VectorStore


def normalise_scores(scores: list[float]) -> list[float]:
    """Chuẩn hóa điểm số về khoảng [0, 1] bằng Min-Max Scaling (theo bài W4 của thầy)."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)  # Tất cả bằng nhau -> gán 1.0
    return [(s - min_s) / (max_s - min_s) for s in scores]


def build_bm25_index(chunks: list[DocumentChunk]) -> Optional[BM25Okapi]:
    """Xây dựng chỉ mục BM25 từ danh sách DocumentChunk."""
    if not chunks:
        return None
    tokenized_corpus = [c.content.lower().split() for c in chunks]
    return BM25Okapi(tokenized_corpus)


class HybridRetriever:
    """Class điều phối truy xuất đa chế độ (Dense, BM25, Hybrid) cho RAG Engine."""

    def __init__(
        self,
        vector_store: VectorStore,
        w_bm25: float = 0.3,
        w_vector: float = 0.7,
    ):
        self.vector_store = vector_store
        self.w_bm25 = w_bm25
        self.w_vector = w_vector
        self._bm25_index: Optional[BM25Okapi] = None
        self._cached_chunks_count: int = 0

    def _get_bm25_index(self, chunks: list[DocumentChunk]) -> Optional[BM25Okapi]:
        """Tự động cache BM25 index để tránh build lại liên tục mỗi khi query."""
        if self._bm25_index is None or len(chunks) != self._cached_chunks_count:
            self._bm25_index = build_bm25_index(chunks)
            self._cached_chunks_count = len(chunks)
        return self._bm25_index

    def search_dense(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Chế độ 1: Chỉ tìm kiếm ngữ nghĩa theo Vector Cosine Similarity (Dense-only)."""
        raw_results = self.vector_store.search(query=query, top_k=top_k)
        return [
            RetrievalResult(chunk=chunk, score=score, mode="dense")
            for chunk, score in raw_results
        ]

    def search_bm25(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Chế độ phụ trợ: Chỉ tìm kiếm từ khóa theo BM25 (Sparse-only)."""
        chunks = list(self.vector_store.chunks.values())
        if not chunks:
            return []

        bm25 = self._get_bm25_index(chunks)
        if not bm25:
            return []

        tokenized_query = query.lower().split()
        scores = list(bm25.get_scores(tokenized_query))
        norm_scores = normalise_scores(scores)

        ranked = sorted(enumerate(norm_scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievalResult(chunk=chunks[idx], score=score, mode="bm25")
            for idx, score in ranked
        ]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 3,
        w_bm25: Optional[float] = None,
        w_vector: Optional[float] = None,
    ) -> list[RetrievalResult]:
        """
        Chế độ 2: Kết hợp BM25 (từ khóa) + Vector Cosine (ngữ nghĩa).
        Công thức của thầy: hybrid_score = w_bm25 * bm25_norm + w_vector * vector_score
        """
        w_b = w_bm25 if w_bm25 is not None else self.w_bm25
        w_v = w_vector if w_vector is not None else self.w_vector

        chunks = list(self.vector_store.chunks.values())
        if not chunks:
            return []

        # 1. Điểm BM25
        bm25 = self._get_bm25_index(chunks)
        if bm25:
            tokenized_query = query.lower().split()
            bm25_scores = list(bm25.get_scores(tokenized_query))
            bm25_norm = normalise_scores(bm25_scores)
        else:
            bm25_norm = [0.0] * len(chunks)

        # 2. Điểm Vector (bảo vệ an toàn nếu embedding is None)
        try:
            query_vec = embed_text(query, task="retrieval.query")
            vector_scores = []
            for c in chunks:
                if c.embedding is not None:
                    vector_scores.append(cosine_similarity(query_vec, c.embedding))
                else:
                    vector_scores.append(0.0)
        except Exception:
            # Nếu mất mạng hoặc lỗi API embedding -> tự động lùi về chỉ dùng BM25
            vector_scores = [0.0] * len(chunks)
            w_b = 1.0
            w_v = 0.0

        # 3. Kết hợp điểm (Score Fusion)
        hybrid_scores = [
            w_b * b + w_v * v
            for b, v in zip(bm25_norm, vector_scores)
        ]

        # 4. Sắp xếp giảm dần và lấy top_k
        ranked = sorted(enumerate(hybrid_scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievalResult(chunk=chunks[idx], score=score, mode="hybrid")
            for idx, score in ranked
        ]

    def search(
        self,
        query: str,
        top_k: int = 3,
        mode: str = "hybrid",
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Hàm giao diện chính hỗ trợ cờ 'mode' theo đúng Rubric Week 4:
        - mode='dense': Tìm kiếm thuần Vector Cosine
        - mode='hybrid': Kết hợp BM25 + Vector
        - mode='bm25': Tìm kiếm thuần từ khóa BM25
        Trả về danh sách (DocumentChunk, score) để tương thích với các module khác.
        """
        if mode == "dense":
            results = self.search_dense(query=query, top_k=top_k)
        elif mode == "bm25":
            results = self.search_bm25(query=query, top_k=top_k)
        else:
            results = self.search_hybrid(query=query, top_k=top_k)

        return [(r.chunk, r.score) for r in results]
