# [DOC_CS103] Database System (Hệ cơ sở dữ liệu)

**doc_id**: DOC_CS103
**title**: Database System
**course_code**: CS103
**credits**: 3 tín chỉ
**date_version**: 2024-2025 / v2.1
**owner**: Khoa Công nghệ Thông tin
**prerequisites**: Introduction to Programming (DOC_CS101)
**assessment_scheme**: Chuyên cần: 10%, Đồ án nhóm CSDL: 20%, Thi giữa kỳ: 30%, Thi cuối kỳ: 40%

---

## 1. Mô tả học phần
Nghiên cứu mô hình thực thể mối kết hợp (ERD), mô hình quan hệ, đại số quan hệ, ngôn ngữ SQL chuẩn, chuẩn hóa cơ sở dữ liệu (1NF, 2NF, 3NF, BCNF) và giao dịch ACID.

## 2. Chuẩn đầu ra học phần (Course Learning Outcomes - CLO)
- CLO1: Thiết kế mô hình dữ liệu quan hệ từ bài toán thực tế qua lược đồ ERD.
- CLO2: Viết truy vấn SQL phức tạp (JOIN, GROUP BY, Subquery, Window function).
- CLO3: Chuẩn hóa lược đồ quan hệ đạt dạng chuẩn 3NF và BCNF.

## 3. Kế hoạch giảng dạy chi tiết theo tuần (Lesson Plan)
### Tuần 1: Tổng quan về hệ cơ sở dữ liệu và Kiến trúc 3 tầng ANSI-SPARC
- **Tài liệu học tập / Đọc trước**: Chương 1 Tổng quan CSDL
- **Hoạt động & Đánh giá**: Cài đặt DBMS MySQL / PostgreSQL / SQL Server

### Tuần 2: Mô hình thực thể mối kết hợp (ERD) và Lược đồ thực thể mở rộng (EER)
- **Tài liệu học tập / Đọc trước**: Chương 2 Mô hình ERD
- **Hoạt động & Đánh giá**: Thiết kế ERD cho hệ thống quản lý thư viện

### Tuần 3: Mô hình quan hệ và Chuyển đổi từ mô hình ERD sang Lược đồ quan hệ
- **Tài liệu học tập / Đọc trước**: Chương 3 Mô hình quan hệ
- **Hoạt động & Đánh giá**: Xác định khóa chính, khóa ngoại và ràng buộc toàn vẹn

### Tuần 4: Đại số quan hệ: Các phép toán chọn, chiếu, kết, tích descartes, hợp, giao
- **Tài liệu học tập / Đọc trước**: Chương 4 Đại số quan hệ
- **Hoạt động & Đánh giá**: Biểu diễn câu hỏi truy vấn dưới dạng biểu thức đại số quan hệ

### Tuần 5: Ngôn ngữ SQL: Định nghĩa dữ liệu (DDL - CREATE, ALTER, DROP) và Toàn vẹn
- **Tài liệu học tập / Đọc trước**: Chương 5 SQL DDL
- **Hoạt động & Đánh giá**: Tạo database và các bảng có khóa ngoại, check constraint

### Tuần 6: Ngôn ngữ SQL: Thao tác dữ liệu (DML - SELECT, INSERT, UPDATE, DELETE)
- **Tài liệu học tập / Đọc trước**: Chương 5 SQL DML
- **Hoạt động & Đánh giá**: Viết câu lệnh truy vấn có INNER JOIN, LEFT JOIN

### Tuần 7: Thi giữa kỳ lý thuyết và viết câu lệnh SQL (60 phút)
- **Tài liệu học tập / Đọc trước**: Phòng máy thi
- **Hoạt động & Đánh giá**: Đánh giá thiết kế ERD và câu truy vấn SQL

### Tuần 8: Truy vấn SQL nâng cao: Gom nhóm GROUP BY, HAVING, Subquery, Correlated subquery
- **Tài liệu học tập / Đọc trước**: Chương 5 SQL Nâng cao
- **Hoạt động & Đánh giá**: Viết truy vấn tìm sinh viên có điểm cao nhất mỗi lớp

### Tuần 9: Lý thuyết thiết kế CSDL: Phụ thuộc hàm và các hệ tiên đề Armstrong
- **Tài liệu học tập / Đọc trước**: Chương 6 Phụ thuộc hàm
- **Hoạt động & Đánh giá**: Tìm bao đóng của tập thuộc tính và tìm tất cả khóa của quan hệ

### Tuần 10: Chuẩn hóa dữ liệu: Dạng chuẩn 1NF, 2NF, 3NF và Boyce-Codd (BCNF)
- **Tài liệu học tập / Đọc trước**: Chương 6 Chuẩn hóa
- **Hoạt động & Đánh giá**: Phân rã lược đồ quan hệ bảo toàn thông tin và phụ thuộc hàm

### Tuần 11: Giao dịch (Transaction), tính chất ACID và Điều khiển đồng thời (Concurrency Control)
- **Tài liệu học tập / Đọc trước**: Chương 7 Giao dịch & Khóa
- **Hoạt động & Đánh giá**: Mô phỏng tình huống Dirty Read, Non-repeatable Read

### Tuần 12: Báo cáo đồ án môn học thiết kế CSDL thực tế và Ôn tập cuối kỳ
- **Tài liệu học tập / Đọc trước**: Buổi vấn đáp đồ án nhóm
- **Hoạt động & Đánh giá**: Bảo vệ mô hình CSDL và chạy demo truy vấn


---
*Đề cương này là văn bản chính thức của Nhà trường áp dụng cho niên khóa hiện hành. Mọi thay đổi phải được Hội đồng Khoa phê duyệt.*
