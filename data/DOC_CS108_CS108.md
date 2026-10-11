# [DOC_CS108] AI Programming with Python (Lập trình AI với Python)

**doc_id**: DOC_CS108
**title**: AI Programming with Python
**course_code**: CS108
**credits**: 3 tín chỉ
**date_version**: 2024-2025 / v2.1
**owner**: Khoa Công nghệ Thông tin
**prerequisites**: Introduction to Programming (DOC_CS101)
**assessment_scheme**: Chuyên cần: 10%, Lab bài tập tuần: 20%, Giữa kỳ: 20%, Đồ án Agentic RAG cuối kỳ: 50%

---

## 1. Mô tả học phần
Học phần trọng tâm về phát triển ứng dụng AI hiện đại: Cấu trúc dữ liệu AI (@dataclass, typing), Tích hợp LLM API, Vector Embeddings, Xây dựng hệ thống RAG Hybrid (BM25 + Cosine), Đo lường đánh giá Faithfulness, Vòng lặp Agent ReAct và Giao thức Model Context Protocol (MCP).

## 2. Chuẩn đầu ra học phần (Course Learning Outcomes - CLO)
- CLO1: Sử dụng dataclass, Type hints và quản lý gọi LLM API an toàn (retry, timeout, prompt separation).
- CLO2: Xây dựng pipeline Hybrid RAG kết hợp BM25 và Vector Similarity kèm chiến lược chunking hợp lý.
- CLO3: Đánh giá độ trung thực (Faithfulness) và hiệu năng của hệ thống RAG qua bộ testset thực nghiệm.
- CLO4: Tự xây dựng vòng lặp Agent ReAct và triển khai MCP Server hỗ trợ tool discovery động qua stdio.

## 3. Kế hoạch giảng dạy chi tiết theo tuần (Lesson Plan)
### Tuần 1: Python Data Structures for AI: List, Dict, Set, @dataclass, Type hints, JSON serialisation
- **Tài liệu học tập / Đọc trước**: Slide Week 1
- **Hoạt động & Đánh giá**: Định nghĩa Schema dữ liệu bài học bằng @dataclass chuẩn PEP 557

### Tuần 2: LLM API Access & Integration: Quản lý API Key, Retry logic, Timeout, Tách prompt template
- **Tài liệu học tập / Đọc trước**: Slide Week 2
- **Hoạt động & Đánh giá**: Viết client gọi mô hình LLM với cơ chế xử lý lỗi HTTP an toàn

### Tuần 3: RAG Foundations: Documents, Chunking strategies, Embeddings và Vector Stores
- **Tài liệu học tập / Đọc trước**: Slide Week 3
- **Hoạt động & Đánh giá**: Xây dựng kho lưu trữ vector nhúng văn bản và tính Cosine Similarity

### Tuần 4: Building the RAG Pipeline: Dense Search, BM25 Keyword Search, Chuẩn hóa và Hybrid Fusion (0.3 BM25 + 0.7 Vector)
- **Tài liệu học tập / Đọc trước**: Slide Week 4
- **Hoạt động & Đánh giá**: Cài đặt thuật toán Hybrid Search BM25 kết hợp vector embedding

### Tuần 5: RAG Evaluation: Faithfulness (Groundedness), Relevance, Calibration (no_answer) và Đo lường Token USAGE
- **Tài liệu học tập / Đọc trước**: Slide Week 5
- **Hoạt động & Đánh giá**: Xây dựng bộ testset 12 câu đánh giá độ trung thực và tỷ lệ từ chối đúng

### Tuần 6: Agent Loop, ReAct Pattern & Tool Calling: MAX_STEPS, Error handling dạng chuỗi, Side-effect và Verify
- **Tài liệu học tập / Đọc trước**: Slide Week 6
- **Hoạt động & Đánh giá**: Tự viết vòng lặp Agent ReAct không dùng framework có sẵn

### Tuần 7: Model Context Protocol (MCP): Viết MCP Server chạy qua stdio, Dynamic Tool Discovery, Multi-Agent Handoff
- **Tài liệu học tập / Đọc trước**: Slide Week 7 & Multi-Agent.pdf
- **Hoạt động & Đánh giá**: Viết MCP Server công bố tool ghi trạng thái và client discover động

### Tuần 8: Hoàn thiện đồ án, Viết Báo cáo chuẩn 8-9 trang, Đo đạc nghiệm thu và Bảo vệ thử
- **Tài liệu học tập / Đọc trước**: Yêu cầu thực hiện đồ án
- **Hoạt động & Đánh giá**: Kiểm thử lệnh chạy một chạm Makefile và chuẩn bị bảo vệ


---
*Đề cương này là văn bản chính thức của Nhà trường áp dụng cho niên khóa hiện hành. Mọi thay đổi phải được Hội đồng Khoa phê duyệt.*
