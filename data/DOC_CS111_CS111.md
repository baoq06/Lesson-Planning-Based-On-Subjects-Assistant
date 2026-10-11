# [DOC_CS111] Information Security (An toàn thông tin)

**doc_id**: DOC_CS111
**title**: Information Security
**course_code**: CS111
**credits**: 3 tín chỉ
**date_version**: 2024-2025 / v2.1
**owner**: Khoa Mạng máy tính & An toàn thông tin
**prerequisites**: Operating System (DOC_CS106), Networking Essentials (DOC_NET101)
**assessment_scheme**: Chuyên cần: 10%, Lab thực hành bảo mật: 20%, Thi giữa kỳ: 20%, Đồ án kiểm thử thâm nhập cuối kỳ: 50%

---

## 1. Mô tả học phần
Nghiên cứu nguyên lý bảo mật CIA triad, mật mã học cổ điển và hiện đại (AES, RSA, ECC), hàm băm, chữ ký số, chứng chỉ số PKI, kiểm thử xâm nhập (Penetration Testing), an ninh mạng và lỗ hổng OWASP Top 10.

## 2. Chuẩn đầu ra học phần (Course Learning Outcomes - CLO)
- CLO1: Vận dụng mã hóa đối xứng, bất đối xứng và chữ ký số bảo vệ thông tin.
- CLO2: Phân tích và phòng chống các lỗ hổng web phổ biến (SQL Injection, XSS, CSRF).
- CLO3: Thực hành rà quét lỗ hổng và kiểm thử an ninh mạng theo quy chuẩn đạo đức.

## 3. Kế hoạch giảng dạy chi tiết theo tuần (Lesson Plan)
### Tuần 1: Nguyên lý cơ bản của An toàn thông tin: Tam giác CIA, Mô hình đe dọa và Rủi ro
- **Tài liệu học tập / Đọc trước**: Chương 1 Tổng quan ATTT
- **Hoạt động & Đánh giá**: Phân tích tình huống vi phạm dữ liệu thực tế

### Tuần 2: Mật mã học cổ điển: Mã dịch chuyển Caesar, Vigenere, Phân tích tần suất chữ cái
- **Tài liệu học tập / Đọc trước**: Chương 2 Mật mã cổ điển
- **Hoạt động & Đánh giá**: Giải mã văn bản mã hóa bằng kỹ thuật phân tích thống kê

### Tuần 3: Mật mã khóa đối xứng hiện đại: DES, 3DES, AES và các chế độ hoạt động (CBC, GCM)
- **Tài liệu học tập / Đọc trước**: Chương 3 Mã hóa đối xứng
- **Hoạt động & Đánh giá**: Cài đặt mã hóa và giải mã file bằng thư viện OpenSSL / PyCryptodome

### Tuần 4: Mật mã khóa công khai: Thuật toán RSA, Trao đổi khóa Diffie-Hellman và ECC
- **Tài liệu học tập / Đọc trước**: Chương 4 Mã hóa công khai
- **Hoạt động & Đánh giá**: Tính toán cặp khóa RSA với số nguyên tố nhỏ và mã hóa chuỗi

### Tuần 5: Hàm băm một chiều (MD5, SHA-256), Mã xác thực thông điệp (HMAC) và Chữ ký số
- **Tài liệu học tập / Đọc trước**: Chương 5 Hàm băm & Chữ ký
- **Hoạt động & Đánh giá**: Ký số văn bản điện tử và kiểm tra tính toàn vẹn

### Tuần 6: Hạ tầng khóa công khai (PKI), Chứng thư số X.509 và Giao thức SSL/TLS
- **Tài liệu học tập / Đọc trước**: Chương 6 PKI & TLS
- **Hoạt động & Đánh giá**: Tạo chứng chỉ SSL tự ký và cấu hình máy chủ HTTPS Nginx

### Tuần 7: Thi giữa kỳ lý thuyết và thuật toán mật mã (60 phút)
- **Tài liệu học tập / Đọc trước**: Phòng thi Giữa kỳ
- **Hoạt động & Đánh giá**: Đánh giá AES, RSA, Hàm băm và PKI

### Tuần 8: Xác thực và Kiểm soát truy cập: Mật khẩu, MFA, OAuth 2.0, JWT, Mô hình RBAC
- **Tài liệu học tập / Đọc trước**: Chương 7 Xác thực
- **Hoạt động & Đánh giá**: Xây dựng luồng xác thực hai bước TOTP trên ứng dụng

### Tuần 9: Các lỗ hổng ứng dụng Web (OWASP Top 10): SQL Injection và Phòng chống
- **Tài liệu học tập / Đọc trước**: Chương 8 Lỗ hổng Web
- **Hoạt động & Đánh giá**: Thực hành khai thác và vá lỗ hổng SQLi trên môi trường DVWA

### Tuần 10: Lỗ hổng Cross-Site Scripting (XSS), CSRF và Cơ chế bảo vệ Same-Origin Policy
- **Tài liệu học tập / Đọc trước**: Chương 8 Web Security
- **Hoạt động & Đánh giá**: Khai thác Stored XSS và phòng vệ bằng Content Security Policy (CSP)

### Tuần 11: Tường lửa (Firewall), Hệ thống phát hiện/ngăn ngừa xâm nhập (IDS/IPS - Snort)
- **Tài liệu học tập / Đọc trước**: Chương 9 An ninh mạng
- **Hoạt động & Đánh giá**: Cấu hình luật Snort phát hiện hành vi quét cổng Nmap

### Tuần 12: Báo cáo đồ án kiểm thử thâm nhập (Pen-test Report) và Tổng kết môn học
- **Tài liệu học tập / Đọc trước**: Hội đồng chấm đồ án
- **Hoạt động & Đánh giá**: Trình bày báo cáo đánh giá rủi ro và giải pháp khắc phục


---
*Đề cương này là văn bản chính thức của Nhà trường áp dụng cho niên khóa hiện hành. Mọi thay đổi phải được Hội đồng Khoa phê duyệt.*
