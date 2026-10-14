import json
import os
from datetime import datetime
from typing import Optional
from src.schemas import Document, DocumentChunk
from dataclasses import dataclass, field, asdict
from src.rag.chunker import chunk_document
from src.rag.embedding import embed_text, batch_embed_text, cosine_similarity

@dataclass
class DocumentStore:
    _docs: dict[str, Document] = field(default_factory=dict)
    _chunks: dict[str, DocumentChunk] = field(default_factory=dict)

    def add_document(self, doc: Document) -> str:
        if not doc.doc_id:
            doc.doc_id = f"doc_{len(self._docs) + 1:04d}"
        
        self._docs[doc.doc_id] = doc
        _chunks = chunk_document(doc)
        for chunk in _chunks:    
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

@dataclass
class VectorStore(DocumentStore):
    def add_document(self, document: Document) -> str:
        if not document.doc_id:
            document.doc_id = f"doc_{len(self._docs) + 1:04d}"
        self._docs[document.doc_id] = document
        
        _chunks = chunk_document(document)
        for chunk in _chunks:
            self.add_chunk(chunk)
        return document.doc_id

    def add_chunk(self, chunk: DocumentChunk) -> str:
        if chunk.embedding is None:
            chunk.embedding = embed_text(chunk.content)
        return super().add_chunk(chunk)

    def add_documents_batch(self, documents: list[Document], batch_size: int = 16) -> None:
        all_chunks: list[DocumentChunk] = []
        for doc in documents:
            if not doc.doc_id:
                doc.doc_id = f"doc_{len(self._docs) + 1:04d}"
            self._docs[doc.doc_id] = doc
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)
        chunks_to_embed = [c for c in all_chunks if c.embedding is None]
        
        for i in range(0, len(chunks_to_embed), batch_size):
            batch = chunks_to_embed[i : i + batch_size]
            texts = [c.content for c in batch]
            embeddings = batch_embed_text(texts)
            for chunk, emb in zip(batch, embeddings):
                chunk.embedding = emb

        for chunk in all_chunks:
            if not chunk.chunk_id:
                chunk.chunk_id = f"chunk_{len(self._chunks) + 1:04d}"
            self._chunks[chunk.chunk_id] = chunk

    def search(self, query: str, top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
        query_embedding = embed_text(query)
        similarities = []
        for chunk in self._chunks.values():
            if chunk.embedding is not None:
                score = cosine_similarity(query_embedding, chunk.embedding)
                similarities.append((chunk, score))
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    def search_with_threshold(
        self, query: str, top_k: int = 3, min_score: float = 0.5
    ) -> list[tuple[DocumentChunk, float]]:
        all_results = self.search(query, top_k=len(self._chunks))
        filtered_results = [(chunk, score) for chunk, score in all_results if score >= min_score]
        return filtered_results[:top_k]

    def batch_search(self, queries: list[str], top_k: int = 3) -> list[list[tuple[DocumentChunk, float]]]:
        """
        Search for multiple queries at once.

        More efficient than calling search() in a loop because
        it can embed all queries in a single API call.

        Returns:
            List of results per query
        """
        # Your code here
        results = []
        api_key = os.getenv("JINA_API_KEY")
        query_embeddings = batch_embed_text(queries)

        for query_embedding in query_embeddings:
            temp = []
            for key, value in self._chunks.items():
                score = cosine_similarity(query_embedding, value.embedding)
                temp.append((value, score))
            temp.sort(key=lambda x:x[1], reverse=True)
            results.append(temp[:top_k])

        return results

    def save(self, path: str) -> None:
        """Save the store to a JSON file (excluding embeddings to save space)."""
        data = {}
        for id, chunk in self._chunks.items():
            chunk_data = asdict(chunk)
            # chunk_data.pop("embedding", None)  # Kept embeddings for search
            data[id] = chunk_data

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load the store from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        self._chunks = {}

        for key, chunk_data in data.items():
            chunk = DocumentChunk(**chunk_data)
            self._chunks[key] = chunk