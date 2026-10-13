import re
from typing import Literal
from src.schemas import Document, DocumentChunk
from src.rag.embedding import batch_embed_text, cosine_similarity


# fixed size chunking
def fixed_size_chunking(
    doc: Document, chunk_size: int = 200, overlap: int = 40
) -> list[DocumentChunk]:
    """Cắt văn bản theo kích thước ký tự cố định và độ gối đầu overlap."""
    text = doc.content.strip()
    chunks: list[DocumentChunk] = []
    chunk_index = 0
    start = 0

    while start < len(text):
        if overlap >= chunk_size:
            break
        end = start + chunk_size
        chunk_text = text[start:end]

        chunk_id = f"{doc.doc_id}_chunk_{chunk_index}" if doc.doc_id else f"chunk_{chunk_index}"
        chunks.append(
            DocumentChunk(
                content=chunk_text,
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                chunk_index=chunk_index,
                metadata=doc.metadata,
            )
        )
        chunk_index += 1
        start += chunk_size - overlap
        if end >= len(text):
            break

    return chunks


# recursive chunking
def recursive_chunking(
    doc: Document,
    chunk_size: int = 400,
    overlap: int = 50,
    separators: list[str] | None = None,
) -> list[DocumentChunk]:
    """Cắt văn bản ưu tiên theo ranh giới đoạn văn (\n\n), dòng (\n), câu (. ), rồi đến từ."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    def _split_text(text: str, seps: list[str]) -> list[str]:
        if not text:
            return []
        if not seps or len(text) <= chunk_size:
            return [text]

        sep = seps[0]
        splits = text.split(sep)
        good_splits: list[str] = []
        current_chunk = ""

        for part in splits:
            part_with_sep = part if not current_chunk else sep + part
            if len(current_chunk) + len(part_with_sep) <= chunk_size:
                current_chunk += part_with_sep
            else:
                if current_chunk:
                    good_splits.append(current_chunk.strip())
                if len(part) > chunk_size:
                    # Nếu 1 phần vẫn quá dài, gọi đệ quy với separator tiếp theo
                    good_splits.extend(_split_text(part, seps[1:]))
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            good_splits.append(current_chunk.strip())
        return good_splits

    text = doc.content.strip()
    raw_chunks = _split_text(text, separators)

    # Đóng gói thành DocumentChunk
    chunks: list[DocumentChunk] = []
    for idx, content in enumerate(raw_chunks):
        if not content:
            continue
        chunk_id = f"{doc.doc_id}_chunk_{idx}" if doc.doc_id else f"chunk_{idx}"
        chunks.append(
            DocumentChunk(
                content=content,
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                chunk_index=idx,
                metadata=doc.metadata,
            )
        )
    return chunks


# senmatic chunking
def split_into_sentences(text: str) -> list[str]:
    """Tách đoạn văn thành các câu dựa trên dấu chấm, hỏi, than hoặc ngắt dòng."""
    # Regex bắt ranh giới kết thúc câu: sau dấu . ? ! kèm khoảng trắng hoặc xuống dòng
    sentences = re.split(r"(?<=[.?!])\s+|\n+", text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunking(
    doc: Document,
    similarity_threshold: float = 0.65,
    max_chunk_size: int = 600,
    min_sentences: int = 1,
) -> list[DocumentChunk]:

    text = doc.content.strip()
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    # Nếu chỉ có 1-2 câu, không cần tính toán phức tạp
    if len(sentences) <= min_sentences:
        chunk_id = f"{doc.doc_id}_chunk_0" if doc.doc_id else "chunk_0"
        return [DocumentChunk(content=text, chunk_id=chunk_id, doc_id=doc.doc_id, chunk_index=0, metadata=doc.metadata)]

    # 1. Batch embed toàn bộ các câu (1 lần gọi API duy nhất)
    embeddings = batch_embed_text(sentences, task="retrieval.passage")

    # 2. Tính độ tương đồng giữa câu i và câu i+1
    similarities: list[float] = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # 3. Gom nhóm các câu thành các chunk
    chunks: list[DocumentChunk] = []
    current_sentences: list[str] = [sentences[0]]
    current_length = len(sentences[0])
    chunk_index = 0

    for i in range(len(similarities)):
        sim = similarities[i]
        next_sentence = sentences[i + 1]
        next_len = len(next_sentence)

        # Điều kiện ngắt chunk:
        # - Ý nghĩa thay đổi (sim < threshold)
        # - HOẶC độ dài cộng thêm vượt quá max_chunk_size
        should_split = (sim < similarity_threshold) or (current_length + next_len > max_chunk_size)

        if should_split and len(current_sentences) >= min_sentences:
            chunk_text = " ".join(current_sentences)
            chunk_id = f"{doc.doc_id}_chunk_{chunk_index}" if doc.doc_id else f"chunk_{chunk_index}"
            chunks.append(
                DocumentChunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    chunk_index=chunk_index,
                    metadata=doc.metadata,
                )
            )
            chunk_index += 1
            current_sentences = [next_sentence]
            current_length = next_len
        else:
            current_sentences.append(next_sentence)
            current_length += next_len + 1

    # Thêm phần còn lại vào chunk cuối
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunk_id = f"{doc.doc_id}_chunk_{chunk_index}" if doc.doc_id else f"chunk_{chunk_index}"
        chunks.append(
            DocumentChunk(
                content=chunk_text,
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                chunk_index=chunk_index,
                metadata=doc.metadata,
            )
        )

    return chunks


# =====================================================================
# Bộ điều phối (Dispatcher) cho toàn bộ dự án
# =====================================================================
def chunk_document(
    doc: Document,
    strategy: Literal["fixed", "recursive", "semantic"] = " recursive",
    **kwargs,
) -> list[DocumentChunk]:
    """Hàm giao diện chung: cho phép chọn chiến lược chunking tùy ý."""
    if strategy == "fixed":
        chunk_size = kwargs.get("chunk_size", 200)
        overlap = kwargs.get("overlap", 40)
        return fixed_size_chunking(doc, chunk_size=chunk_size, overlap=overlap)
    elif strategy == "semantic":
        threshold = kwargs.get("similarity_threshold", 0.65)
        max_size = kwargs.get("max_chunk_size", 600)
        return semantic_chunking(doc, similarity_threshold=threshold, max_chunk_size=max_size)
    else:  # recursive (mặc định)
        chunk_size = kwargs.get("chunk_size", 400)
        overlap = kwargs.get("overlap", 50)
        return recursive_chunking(doc, chunk_size=chunk_size, overlap=overlap)