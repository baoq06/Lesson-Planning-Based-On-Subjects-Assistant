# Tuần 5: Đánh giá RAG và Đo lường định lượng (Evaluation & Groundedness)

## 1. Thông tin chung & Chuẩn đầu ra (Learning Objectives - A5.1 đến A5.4)
- **A5.1**: Đo lường định lượng 3 chỉ số then chốt: Faithfulness (Groundedness), Answer Relevance, Context Precision.
- **A5.2**: Xây dựng Evaluation Pipeline tự động với Testset đa nhóm và kỹ thuật LLM-as-a-judge.
- **A5.3**: Chẩn đoán lỗi RAG qua ma trận chỉ số: Phân biệt lỗi do Retrieval (kém tìm kiếm) hay lỗi do Generation (LLM Hallucination).
- **A5.4**: Đưa ra quyết định cải tiến RAG dựa trên số liệu thực nghiệm thay vì phỏng đoán.

## 2. Lý thuyết nền tảng & Kiến trúc hệ thống
### 2.1. Vấn đề "Confident, Fluent, and Wrong" trong RAG
Một hệ thống RAG không có tầng đánh giá sẽ dễ đánh lừa người dùng: câu trả lời nghe rất trôi chảy, tự tin nhưng thực chất là thông tin bịa đặt (Hallucination) do:
- **Retrieval Failure**: Bộ tìm kiếm trả về các chunk sai hoặc không đủ thông tin.
- **Generation Failure**: Bộ tìm kiếm trả về đúng chunk nhưng LLM bỏ qua context và tự ý trả lời dựa trên dữ liệu pre-train bên ngoài.

### 2.2. Khung đánh giá RAGAS & Các chỉ số định lượng
- **Faithfulness (Độ trung thực / Groundedness)**:
  Tỷ lệ phần trăm các luận điểm (Claims) trong câu trả lời có bằng chứng trực tiếp trong context đã truy xuất:
  $$\text{Faithfulness} = \frac{|\text{Số Claims được chứng minh trong Context}|}{|\text{Tổng số Claims đưa ra}|}$$
  - Điểm $1.0$: Toàn bộ câu trả lời hoàn toàn dựa trên bài giảng.
  - Điểm $<0.8$: Hệ thống đang có hiện tượng bịa đặt thông tin.
- **Out-of-Scope Refusal Rate (Tỷ lệ từ chối chuẩn xác)**:
  Đo lường trên tập câu hỏi ngoài chương trình. Bắt buộc hệ thống phải đạt $100\%$ từ chối, không được trả lời các chủ đề ngoài môn học (nấu ăn, thể thao, tin tức).
- **Citation Precision**:
  Đo lường độ chính xác của các mã trích dẫn bài học `[LP_W...]`. Đảm bảo các chunk được trích dẫn thực sự nằm trong top-k mà retriever đã tìm thấy.

### 2.3. Ma trận chẩn đoán nguyên nhân lỗi RAG (Diagnostic Matrix)
| Faithfulness | Context Relevance | Chẩn đoán nguyên nhân | Giải pháp cải tiến |
| :---: | :---: | :--- | :--- |
| **THẤP** | **CAO** | LLM Hallucination (Tìm đúng nhưng bịa thêm) | Tăng cường Grounding Prompt, hạ temperature = 0.0, dùng Critic Verifier |
| **THẤP** | **THẤP** | Retrieval Gap (Tìm sai chunk dẫn đến trả lời sai) | Chuyển sang Hybrid Search, chỉnh chunk size/overlap, tinh chỉnh $w_{bm25}$ |
| **CAO** | **THẤP** | LLM chỉ tóm tắt đúng context rác được đưa vào | Tăng ngưỡng lọc điểm tương đồng (min_score threshold) |
| **CAO** | **CAO** | RAG Pipeline hoạt động hoàn hảo | Duy trì và mở rộng testset |

## 3. Các bài tập Lab & Mã nguồn mẫu từng bước (Step-by-step Exercises)
### Exercise 5.1 — Thiết kế Testset chuẩn hóa (data/testset.jsonl)
Testset được chia làm 3 nhóm câu hỏi rõ rệt:
1. **Factual (Truy xuất trực tiếp)**: Câu hỏi có câu trả lời nằm trọn trong 1 chunk (ví dụ: công thức cosine similarity).
2. **Synthesis (Tổng hợp kiến thức)**: Câu hỏi yêu cầu gom thông tin từ nhiều tuần học (ví dụ: so sánh công cụ tuần 4 và tuần 7).
3. **Out-of-Scope (Ngoài chương trình)**: Câu hỏi cố tình nằm ngoài phạm vi môn học để kiểm tra cơ chế từ chối an toàn.

### Exercise 5.2 — Kỹ thuật LLM-as-a-judge đo Faithfulness
```python
FAITHFULNESS_JUDGE_PROMPT = """Bạn là giám khảo đánh giá RAG.
Nhiệm vụ: Đối chiếu từng luận điểm trong CÂU TRẢ LỜI với NGỮ CẢNH.
Nếu có bất kỳ chi tiết nào không được chứng minh bởi ngữ cảnh -> Đánh giá 0.
Nếu toàn bộ đều có nguồn gốc trong ngữ cảnh -> Đánh giá 1.

NGỮ CẢNH: {context}
CÂU TRẢ LỜI: {answer}

Trả về định dạng JSON: {"score": 0 hoặc 1, "unsupported_claims": [...]}
"""
```

### Exercise 5.3 — Tính toán Refusal Rate tự động
```python
def compute_refusal_metrics(predictions: list[dict]) -> float:
    refusals = 0
    out_of_scope = [p for p in predictions if p["type"] == "out_of_scope"]
    for item in out_of_scope:
        ans = item["answer"].lower()
        if any(kw in ans for kw in ["ngoài phạm vi", "không nằm trong", "tôi chỉ hỗ trợ môn học"]):
            refusals += 1
    return refusals / len(out_of_scope) if out_of_scope else 1.0
```

## 4. Các bẫy thường gặp & Nguyên tắc thiết kế (Gotchas & Best Practices)
- **Bẫy Đánh giá bằng mắt (Vibe Checking)**: Không được đánh giá RAG bằng việc "thử vài câu thấy mượt là được"; bắt buộc phải chạy testset tự động và báo cáo chỉ số định lượng.
- **Bẫy Judge Bias**: Khi dùng LLM làm giám khảo, LLM có xu hướng thiên vị các câu trả lời dài dòng hơn câu trả lời ngắn; prompt giám khảo cần định nghĩa tiêu chí cụ thể và ép buộc output dạng JSON.

## 5. Nhiệm vụ tự học & Đánh giá năng lực (Independent Labs)
- Xây dựng file `tests/test_evaluation_pipeline.py` tự động chạy 12 test cases và xuất bảng số liệu Benchmark chuẩn xác.
