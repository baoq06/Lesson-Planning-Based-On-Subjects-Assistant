# [DOC_CS105] Computer Architecture and Assembly Language (Kiến trúc máy tính và Hợp ngữ)

**doc_id**: DOC_CS105
**title**: Computer Architecture and Assembly Language
**course_code**: CS105
**credits**: 3 tín chỉ
**date_version**: 2024-2025 / v2.1
**owner**: Khoa Công nghệ Thông tin
**prerequisites**: Basic Electronics (DOC_EE101)
**assessment_scheme**: Chuyên cần: 10%, Lab thực hành Assembly: 20%, Thi giữa kỳ: 30%, Thi cuối kỳ: 40%

---

## 1. Mô tả học phần
Nghiên cứu cấu trúc bên trong máy tính: Tập lệnh ISA (MIPS/x86), đường truyền dữ liệu (Datapath), điều khiển, kỹ thuật ống dẫn (Pipelining), hệ thống phân cấp bộ nhớ Cache và lập trình Hợp ngữ.

## 2. Chuẩn đầu ra học phần (Course Learning Outcomes - CLO)
- CLO1: Lập trình bằng hợp ngữ Assembly x86 hoặc MIPS thực thi các thuật toán cơ bản.
- CLO2: Phân tích hoạt động của đường dẫn dữ liệu đơn chu kỳ và đa chu kỳ.
- CLO3: Tính toán hiệu năng CPU và cơ chế phân cấp bộ nhớ Cache hit/miss.

## 3. Kế hoạch giảng dạy chi tiết theo tuần (Lesson Plan)
### Tuần 1: Tổng quan về Kiến trúc máy tính, Mô hình Von Neumann và Hiệu năng CPU
- **Tài liệu học tập / Đọc trước**: Chương 1 Tổng quan phần cứng
- **Hoạt động & Đánh giá**: Công thức tính thời gian thực thi CPU Time

### Tuần 2: Kiến trúc tập lệnh (ISA): Định dạng lệnh MIPS (R-type, I-type, J-type)
- **Tài liệu học tập / Đọc trước**: Chương 2 Tập lệnh MIPS
- **Hoạt động & Đánh giá**: Biến đổi lệnh C sang mã máy nhị phân MIPS

### Tuần 3: Lập trình Hợp ngữ Assembly cơ bản: Phép toán số học, logic và nạp/lưu bộ nhớ (lw, sw)
- **Tài liệu học tập / Đọc trước**: Chương 2 Lập trình MARS/SPIM
- **Hoạt động & Đánh giá**: Cài đặt môi trường mô phỏng MARS MIPS

### Tuần 4: Rẽ nhánh và Vòng lặp trong Hợp ngữ (beq, bne, slt, j)
- **Tài liệu học tập / Đọc trước**: Chương 2 Cấu trúc điều khiển
- **Hoạt động & Đánh giá**: Viết vòng lặp for và while bằng hợp ngữ

### Tuần 5: Thủ tục và hàm trong Assembly: Quy ước thanh ghi, Ngăn xếp Stack và jal/jr
- **Tài liệu học tập / Đọc trước**: Chương 2 Gọi hàm & Stack
- **Hoạt động & Đánh giá**: Cài đặt hàm tính giai thừa bằng đệ quy trong Assembly

### Tuần 6: Số học máy tính: Biểu diễn số nguyên có dấu bù 2 và số thực dấu phẩy động IEEE 754
- **Tài liệu học tập / Đọc trước**: Chương 3 Số học máy tính
- **Hoạt động & Đánh giá**: Chuyển đổi số thực sang định dạng 32-bit IEEE 754

### Tuần 7: Thi giữa kỳ trên máy tính (Lập trình Assembly MIPS 60 phút)
- **Tài liệu học tập / Đọc trước**: Phòng máy thi
- **Hoạt động & Đánh giá**: Chấm bài code Assembly

### Tuần 8: Thiết kế bộ xử lý (Datapath): Đường dữ liệu đơn chu kỳ (Single-Cycle Datapath)
- **Tài liệu học tập / Đọc trước**: Chương 4 Datapath
- **Hoạt động & Đánh giá**: Theo dõi tín hiệu điều khiển của lệnh lw, sw, add

### Tuần 9: Kỹ thuật đường ống (Pipelining): Các giai đoạn IF, ID, EX, MEM, WB
- **Tài liệu học tập / Đọc trước**: Chương 4 Pipeline
- **Hoạt động & Đánh giá**: Xác định các xung đột Pipeline: Structural, Data, Control Hazard

### Tuần 10: Xung đột dữ liệu và Giải pháp: Chuyển tiếp dữ liệu (Forwarding) và Tạm dừng (Stalling)
- **Tài liệu học tập / Đọc trước**: Chương 4 Hazard Resolution
- **Hoạt động & Đánh giá**: Vẽ biểu đồ thời gian thực thi Pipeline

### Tuần 11: Hệ thống phân cấp bộ nhớ: Bộ nhớ đệm Cache (Direct-Mapped, Set-Associative)
- **Tài liệu học tập / Đọc trước**: Chương 5 Bộ nhớ Cache
- **Hoạt động & Đánh giá**: Tính toán địa chỉ Tag, Index, Offset và tỷ lệ Miss Penalty

### Tuần 12: Bộ nhớ ảo (Virtual Memory), Phân trang (Paging) và Ôn tập thi cuối kỳ
- **Tài liệu học tập / Đọc trước**: Chương 5 Bộ nhớ ảo
- **Hoạt động & Đánh giá**: Phân tích bảng trang Page Table và TLB


---
*Đề cương này là văn bản chính thức của Nhà trường áp dụng cho niên khóa hiện hành. Mọi thay đổi phải được Hội đồng Khoa phê duyệt.*
