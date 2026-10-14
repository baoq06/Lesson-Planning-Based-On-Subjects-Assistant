---
doc_id: POLICY_AIPR
title: Quy chế môn học AIPR
version: 1.0
date: 2026-10-12
owner: AIPR Instructor
---
# Quy chế Môn học: Lập trình AI với Python (AIPR Course Policy)

## 1. Thông tin môn học & Quy định chung
- **Tên môn học:** Lập trình AI với Python (AI Programming with Python - AIPR).
- **Mã môn học:** AIPR_24110FIE
- **Thời lượng:** 7.5 tuần (2 buổi / tuần, 4 giờ / buổi) -> 60 giờ học tập.
- **Kênh thông báo & Hỗ trợ:** Toàn bộ thông báo bài tập, deadline và tài liệu học tập được cập nhật trên cổng LMS của trường và hệ thống Trợ lý học tập AI AIPR (AIPR Learning Assistant).

## 2. Cơ cấu Đánh giá & Thang điểm (Grading Scheme)
Điểm tổng kết môn học được tính trên thang điểm 10 theo cơ cấu trọng số như sau:

- **1. Điểm Chuyên cần (5%):**
  - Đánh giá mức độ tham dự đầy đủ, đúng giờ tại các buổi lên lớp và tham gia tương tác trực tiếp (Recall Sprints, Q&A).
  - Vắng quá 20% tổng số buổi học sẽ bị cấm thi/không đủ điều kiện bảo vệ đồ án cuối kỳ.

- **2. Điểm Bài tập thực hành Lab (15%):**
  - Tổng hợp kết quả các bài tập thực hành trên lớp từ Tuần 1 đến Tuần 7 (`ex1_1`, `ex1_2`, `ex2_1`, `lab_sheet`).
  - Sinh viên hoàn thiện mã nguồn và vượt qua toàn bộ kiểm thử tự động (Unit Tests) theo từng tuần.

- **3. Điểm Bài kiểm tra định kỳ (30%):**
  - Gồm 2 bài kiểm tra lớn (mỗi bài chiếm 15%):
    - *Bài Test 1 (Giữa kỳ - Tuần 4):* Đánh giá kỹ năng xây dựng Pipeline RAG, Chunking, Embeddings và Hybrid Search (BM25 + Vector).
    - *Bài Test 2 (Tuần 7):* Đánh giá kỹ năng lập trình Agentic Loop, Function Calling và giao thức Model Context Protocol (MCP).

- **4. Đồ án Cuối kỳ & Vấn đáp (50%):**
  - Xây dựng một hệ thống Agentic AI hoàn chỉnh kết hợp đầy đủ: RAG Engine, MCP Tools (có Side Effects kiểm chứng được), và mô hình Đa tác tử (Producer-Critic Verifier).
  - Điểm số bao gồm: Đánh giá hồ sơ nộp bài, tỷ lệ vượt qua Unit Test / Benchmark định lượng, và phần **Vấn đáp cá nhân** trực tiếp với Giảng viên.

## 3. Quy định Đồ án Cuối kỳ & Hình thức Thực hiện
- **Hình thức làm bài:**
  - Sinh viên bắt buộc làm theo nhóm từ **2 đến 3 sinh viên/nhóm**.
  - Đồ án phải đáp ứng đầy đủ tất cả các tiêu chí kỹ thuật và nghiệp vụ do Giảng viên yêu cầu.
  - Mỗi thành viên trong nhóm phải có phần công việc rõ ràng, được ghi nhận trong bảng phân chia trách nhiệm (Workload Breakdown).
- **Vấn đáp cá nhân (Oral Defense):**
  - Giảng viên sẽ tổ chức vấn đáp trực tiếp từng sinh viên độc lập.
  - Điểm số cuối cùng của mỗi thành viên được đánh giá dựa trên mức độ hiểu sâu mã nguồn, khả năng giải trình thiết kế kiến trúc và khối lượng đóng góp thực tế (không cào bằng điểm nhóm).
- **Hồ sơ nộp bài chuẩn (Deliverables Bundle):**
  Khi nộp bài cuối kỳ, mỗi nhóm bắt buộc phải đóng gói đầy đủ các thành phần sau:
  1. **Source Code:** Mã nguồn hoàn chỉnh kèm link GitHub Repository hoặc file nén `.zip`.
  2. **Báo cáo đồ án (`report.docx`):** Tài liệu văn bản mô tả bài toán, kiến trúc hệ thống, sơ đồ luồng dữ liệu, thực nghiệm và kết quả.
  3. **Nhật ký thực thi (`logs/`):** Thư mục lưu vết toàn bộ log chạy thực tế của hệ thống (Agent loop trace, MCP calls, retriever logs).
  4. **Bộ kiểm thử chuẩn (`testset.jsonl`):** Tập dữ liệu benchmark tối thiểu 30-50 test cases dùng để đo lường định lượng (Faithfulness, Hit Rate, Refusal Rate).
  5. **Báo cáo sử dụng AI (`AI_USAGE.md`):** Tài liệu kê khai minh bạch việc sử dụng công cụ GenAI trong quá trình làm đồ án.
  6. **Hướng dẫn triển khai (`README.md`):** Tài liệu hướng dẫn cài đặt môi trường, cấu hình biến môi trường và lệnh chạy demo chi tiết.

## 4. Chính sách Nộp bài trễ hạn & Gia hạn Deadline (Late Submission & Extension Policy)
- **Chính sách Nộp bài trễ thông thường:**
  - Nộp trễ từ 1 đến 24 giờ (Ngày thứ 1): **Trừ 10%** tổng số điểm của bài nộp.
  - Nộp trễ từ 25 đến 48 giờ (Ngày thứ 2): **Trừ 20%** tổng số điểm của bài nộp.
  - Nộp trễ từ 49 đến 72 giờ (Ngày thứ 3): **Trừ 30%** tổng số điểm của bài nộp.
  - Nộp trễ quá 72 giờ (Sau 3 ngày): **Nhận 0 điểm** cho bài tập đó. Hệ thống tự động đóng cổng nộp bài.
- **Chính sách Gia hạn Deadline (Extension Policy):**
  - **Điều kiện duy nhất:** Nhà trường và môn học **CHỈ xem xét gia hạn cho trường hợp bất khả kháng về y tế** (ốm đau, nhập viện có Giấy xác nhận / Bệnh án hợp lệ từ bác sĩ hoặc cơ sở y tế).
  - **Thời gian gia hạn tối đa:** Không quá **5 ngày** kể từ hạn chót ban đầu.
  - **Quy trình gửi yêu cầu:** Sinh viên phải gửi đơn kèm ảnh chụp minh chứng y tế thông qua hệ thống Trợ lý học tập hoặc email Giảng viên trước khi hết hạn nộp bài.

## 5. Liêm chính Học thuật & Quy định Sử dụng GenAI (Academic Integrity & GenAI Policy)
Môn học khuyến khích sinh viên tận dụng sức mạnh của các mô hình ngôn ngữ lớn (LLM), nhưng bắt buộc phải tuân thủ nghiêm ngặt tính trung thực học thuật:
- **Quy định sử dụng GenAI hợp lệ:**
  - Sinh viên **ĐƯỢC PHÉP** sử dụng các công cụ LLM (ChatGPT, Claude, Groq, Gemini, Copilot) để: tra cứu khái niệm lý thuyết, giải thích lỗi lập trình, tối ưu thuật toán hoặc đóng vai trò phản biện (Code Review).
  - Mọi trường hợp có sử dụng GenAI bắt buộc phải ghi nhận vào file `AI_USAGE.md` (bao gồm: tên công cụ, đường link/phiên bản model, câu lệnh prompt chính đã dùng và mô tả đoạn code được sinh/tối ưu).
  - Sinh viên phải tự tay làm chủ và hiểu 100% từng dòng mã nguồn nộp lên.
- **Hành vi vi phạm & Chế tài xử lý:**
  - Nghiêm cấm hành vi sao chép mã nguồn của nhóm khác hoặc nhờ người ngoài viết hộ bài.
  - Trong buổi Vấn đáp cá nhân, nếu sinh viên **không giải thích được mã nguồn** do AI sinh ra, hoặc bị phát hiện đạo văn:
    - Bài nộp sẽ bị **TRỪ 100% ĐIỂM (Nhận 0 điểm)** ngay lập tức.
    - Chuyển hồ sơ vi phạm liêm chính học thuật lên Hội đồng Kỷ luật Khoa để xử lý kỷ luật môn học.

## 6. Quy trình Phúc khảo & Khiếu nại Điểm số
- Sau khi Giảng viên công bố điểm từng bài tập/kiểm tra, sinh viên có thời hạn **03 ngày làm việc** để gửi yêu cầu phúc khảo nếu phát hiện sai sót trong quá trình chấm điểm.
- Đơn phúc khảo phải chỉ rõ câu hỏi, phần code hoặc tiêu chí chấm cần xem xét lại kèm lý do cụ thể.
