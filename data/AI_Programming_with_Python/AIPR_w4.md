# Tuần 4: Xây dựng RAG Pipeline & Tìm kiếm lai (Hybrid Search)

## 1. Thông tin chung & Chuẩn đầu ra (Learning Objectives - A4.1 đến A4.4)
- **A4.1**: Thiết kế Grounding System Prompt: Ép buộc LLM chỉ trả lời DUY NHẤT dựa trên context được cung cấp.
- **A4.2**: Lắp ráp pipeline RAG hoàn chỉnh 5 giai đoạn: Query -> Retrieve -> Build Prompt -> LLM Call -> Return Answer.
- **A4.3**: Quản lý và định dạng nguồn trích dẫn (Citations) `[doc_id]` trong câu trả lời để người dùng kiểm chứng.
- **A4.4**: Xử lý toàn diện các trường hợp biên (Edge Cases): Ngữ cảnh rỗng, câu hỏi ngoài lề, prompt vượt ngưỡng token.

## 2. Lý thuyết nền tảng & Kiến trúc hệ thống
### 2.1. Động lực của Tìm kiếm lai (Why Hybrid Search?)
- **Sparse Retrieval (BM25 Okapi)**:
  - Dựa trên tần suất từ (Term Frequency - TF) và độ nghịch đảo tần suất văn bản (Inverse Document Frequency - IDF).
  - Cực kỳ mạnh khi tìm kiếm các từ khóa chính xác, mã định danh (`DOC001`, `LP_W04`), thuật ngữ hiếm, từ viết tắt.
  - Nhược điểm: Bị lỗi khi người dùng dùng từ đồng nghĩa hoặc câu hỏi diễn đạt khác đi (Vocabulary Mismatch Problem).
- **Dense Retrieval (Vector Embedding)**:
  - Nắm bắt ngữ nghĩa tương đồng (Semantic meaning), câu hỏi diễn đạt tự nhiên.
  - Nhược điểm: Kém chính xác với các từ khóa kỹ thuật số hiệu hoặc thuật ngữ quá đặc thù.
- **Score Fusion với Min-Max Normalisation**:
  Do điểm BM25 không bị chặn trên trong khi điểm Vector nằm trong $[0, 1]$, cần chuẩn hóa điểm số trước khi hợp nhất:
  $$\text{norm}(S) = \frac{S - S_{\min}}{S_{\max} - S_{\min}}$$
  $$\text{Score}_{\text{hybrid}} = w_{\text{bm25}} \times \text{norm}(BM25) + w_{\text{vector}} \times \text{Vector}$$
  (Trọng số chuẩn của Thầy: $w_{\text{bm25}} = 0.4$, $w_{\text{vector}} = 0.6$).

### 2.2. Kiến trúc Grounding Prompt & Anti-Hallucination Guardrail
Prompt đóng vai trò là "bức tường lửa" duy nhất ngăn LLM tự bịa:
- Phải có chỉ dẫn rõ ràng: *"Chỉ trả lời dựa trên Context dưới đây. Nếu không đủ dữ liệu, bắt buộc từ chối: 'Tôi không tìm thấy đủ thông tin trong tài liệu bài giảng'."*
- Bắt buộc gắn tag trích dẫn mã đoạn: `[LP_W04_chunk_0]` sau mỗi luận điểm trả lời.

## 3. Các bài tập Lab & Mã nguồn mẫu từng bước (Step-by-step Exercises)
### Exercise 4.1 — Grounding Prompt Template (ex1_1_rag_prompt.py)
```python
RAG_SYSTEM_PROMPT = """Bạn là trợ lý học tập môn AIPR.
QUY TẮC CỐT LÕI:
1. Trả lời câu hỏi của sinh viên CHỈ dựa trên phần CONTEXT được cung cấp dưới đây.
2. Tuyệt đối không tự suy diễn hoặc dùng kiến thức ngoài tài liệu.
3. Nếu CONTEXT không chứa thông tin để trả lời, bắt buộc nói:
   "Nội dung này không nằm trong tài liệu bài giảng môn học."
4. Luôn trích dẫn nguồn bài học theo định dạng [LP_W...] ở cuối câu trả lời.

CONTEXT:
{context}
"""
```

### Exercise 4.2 — Thuật toán BM25 & Score Fusion (ex2_1_hybrid_search.py)
```python
from rank_bm25 import BM25Okapi

def normalise_scores(scores: list[float]) -> list[float]:
    if not scores or max(scores) == min(scores):
        return [1.0] * len(scores)
    s_min, s_max = min(scores), max(scores)
    return [(s - s_min) / (s_max - s_min) for s in scores]

def hybrid_search(query: str, chunks: list, bm25: BM25Okapi, vector_scores: list[float], top_k=3, w_b=0.4, w_v=0.6):
    tokenized_query = query.lower().split()
    bm25_raw = bm25.get_scores(tokenized_query)
    bm25_norm = normalise_scores(list(bm25_raw))

    fused_scores = [w_b * b + w_v * v for b, v in zip(bm25_norm, vector_scores)]
    ranked = sorted(enumerate(fused_scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunks[i], score) for i, score in ranked]
```

### Exercise 4.3 — RAG Pipeline v2 Complete (ex2_2_rag_pipeline_v2.py)
- Tích hợp nạp dữ liệu, thực hiện Hybrid Search, ghép Context vào prompt, gọi Groq Chat Completion và in kết quả kèm citations.

## 4. Các bẫy thường gặp & Nguyên tắc thiết kế (Gotchas & Best Practices)
- **Bẫy Chia cho 0 khi Min-Max**: Khi tất cả các chunk có điểm BM25 bằng nhau ($S_{\max} == S_{\min}$), công thức chuẩn hóa sẽ bị lỗi `ZeroDivisionError`; bắt buộc kiểm tra điều kiện biên để gán mảng điểm mặc định.
- **Bẫy Context Overflow**: Ghép quá nhiều chunk (`top_k` quá lớn) sẽ làm tràn token của LLM hoặc gây ra hiện tượng Lost-in-the-Middle.

## 5. Nhiệm vụ tự học & Đánh giá năng lực (Independent Labs)
- Cài đặt cơ chế kiểm tra câu hỏi rỗng hoặc câu hỏi ngoài lề để từ chối ngay trước khi gọi LLM nhằm tiết kiệm tài nguyên hệ thống.
