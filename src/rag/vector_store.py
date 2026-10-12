import json
import os
from datetime import datetime
from typing import Optional
from src.schemas import Document, DocumentChunk
from src.rag.chunker import chunk_document
from src.rag.embedding import embed_text, batch_embed_text, cosine_similarity


class DocumentStore:
    def __init__(self):
        self._docs: dict[str, Document] = {}
        self._chunks: dict[str, DocumentChunk] = {}

    def add_document(self, doc: Document) -> str:
        # Chuẩn hóa ID dạng doc_0001 nếu chưa có
        if not doc.doc_id:
            doc.doc_id = f"doc_{len(self._docs) + 1:04d}"
        
        self._docs[doc.doc_id] = doc
        chunks = chunk_document(doc)
        for chunk in chunks:    
            self._chunks[chunk.chunk_id] = chunk
        return doc.doc_id

    def add_chunk(self, chunk: DocumentChunk) -> str:
        if not chunk.doc_id:
            chunk.doc_id = f"doc_{len(self._docs) + 1:04d}"
        if not chunk.chunk_id:
            chunk.chunk_id = f"chunk_{len(self._chunks) + 1:04d}"
        self._chunks[chunk.chunk_id] = chunk
        return chunk.chunk_id
        
    def delete(self, doc_id: str) -> bool:
        if doc_id not in self._docs:
            return False
        del self._docs[doc_id]
        # Xóa các chunk liên quan
        self._chunks = {cid: c for cid, c in self._chunks.items() if c.doc_id != doc_id}
        return True

    def list_documents(self) -> list[Document]:
        return list(self._docs.values())

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self._docs.get(doc_id)

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self._chunks.get(chunk_id)

    def get_chunks_by_doc(self, doc_id: str) -> list[DocumentChunk]:
        return [c for c in self._chunks.values() if c.doc_id == doc_id]

    def count(self) -> dict:
        return {
            "Documents": len(self._docs),
            "Chunks": len(self._chunks)
        }

    @property
    def documents(self):
        return self._docs

    @property
    def chunks(self):
        return self._chunks


class VectorStore(DocumentStore):
    """A DocumentStore with embedding-based semantic search."""

    def add_document(self, document: Document) -> str:
        if not document.doc_id:
            document.doc_id = f"doc_{len(self._docs) + 1:04d}"
        self._docs[document.doc_id] = document
        
        chunks = chunk_document(document)
        for chunk in chunks:
            self.add_chunk(chunk)
        return document.doc_id

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Thêm hàng loạt tài liệu với batch embedding tối ưu tốc độ (1 API call)."""
        doc_ids = []
        all_new_chunks: list[DocumentChunk] = []

        for doc in documents:
            if not doc.doc_id:
                doc.doc_id = f"doc_{len(self._docs) + 1:04d}"
            self._docs[doc.doc_id] = doc
            doc_ids.append(doc.doc_id)

            chunks = chunk_document(doc)
            for c in chunks:
                if not c.doc_id:
                    c.doc_id = doc.doc_id
                if not c.chunk_id:
                    c.chunk_id = f"chunk_{len(self._chunks) + len(all_new_chunks) + 1:04d}"
                all_new_chunks.append(c)

        # Batch embed các chunks chưa có embedding
        chunks_to_embed = [c for c in all_new_chunks if c.embedding is None]
        if chunks_to_embed:
            embeddings = batch_embed_text([c.content for c in chunks_to_embed], task="retrieval.passage")
            for c, emb in zip(chunks_to_embed, embeddings):
                c.embedding = emb

        for c in all_new_chunks:
            self._chunks[c.chunk_id] = c

        return doc_ids

    def add_chunk(self, chunk: DocumentChunk) -> str:
        """Thêm chunk và tự động sinh embedding nếu chưa có."""
        if chunk.embedding is None:
            chunk.embedding = embed_text(chunk.content, task="retrieval.passage")
        return super().add_chunk(chunk)

    def search(self, query: str, top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
        """Tìm kiếm top_k chunks có độ tương đồng ngữ nghĩa cao nhất với query."""
        query_embedding = embed_text(query, task="retrieval.query")
        similarities = []
        for chunk in self.chunks.values():
            if chunk.embedding is not None:
                score = cosine_similarity(query_embedding, chunk.embedding)
                similarities.append((chunk, score))
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    def search_with_threshold(
        self, query: str, top_k: int = 3, min_score: float = 0.5
    ) -> list[tuple[DocumentChunk, float]]:
        """Tìm kiếm có điều kiện lọc theo ngưỡng điểm tối thiểu."""
        all_results = self.search(query, top_k=len(self.chunks))
        filtered_results = [(chunk, score) for chunk, score in all_results if score >= min_score]
        return filtered_results[:top_k]

    def save_to_disk(self, file_path: str = "data/vector_store/store.json") -> None:
        """Lưu toàn bộ Documents và Chunks (kèm Embeddings) xuống file JSON trên đĩa."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data = {
            "documents": [
                {
                    "content": d.content,
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "source": d.source,
                    "metadata": d.metadata,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                }
                for d in self._docs.values()
            ],
            "chunks": [
                {
                    "content": c.content,
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "metadata": c.metadata,
                    "embedding": c.embedding,
                }
                for c in self._chunks.values()
            ],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_disk(self, file_path: str = "data/vector_store/store.json") -> bool:
        """Nạp Documents và Chunks (kèm Embeddings) từ file JSON đã lưu trên đĩa."""
        if not os.path.exists(file_path):
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._docs.clear()
            self._chunks.clear()

            for d_info in data.get("documents", []):
                created_at = None
                if d_info.get("created_at"):
                    try:
                        created_at = datetime.fromisoformat(d_info["created_at"])
                    except Exception:
                        pass
                doc = Document(
                    content=d_info["content"],
                    doc_id=d_info["doc_id"],
                    title=d_info.get("title", ""),
                    source=d_info.get("source", ""),
                    metadata=d_info.get("metadata", {}),
                    created_at=created_at,
                )
                self._docs[doc.doc_id] = doc

            for c_info in data.get("chunks", []):
                chunk = DocumentChunk(
                    content=c_info["content"],
                    chunk_id=c_info["chunk_id"],
                    doc_id=c_info["doc_id"],
                    chunk_index=c_info["chunk_index"],
                    metadata=c_info.get("metadata", {}),
                    embedding=c_info.get("embedding"),
                )
                self._chunks[chunk.chunk_id] = chunk

            return len(self._chunks) > 0
        except Exception:
            return False

    def load_or_build(
        self, documents: list[Document], file_path: str = "data/vector_store/store.json"
    ) -> None:
        """Tự động nạp từ cache trên đĩa nếu có, nếu chưa thì tính toán và lưu cache."""
        if self.load_from_disk(file_path):
            return
        self.add_documents(documents)
        self.save_to_disk(file_path)