# [DOC_CS106] Operating System (Hệ điều hành)

**doc_id**: DOC_CS106
**title**: Operating System
**course_code**: CS106
**credits**: 3 tín chỉ
**date_version**: 2024-2025 / v2.1
**owner**: Khoa Công nghệ Thông tin
**prerequisites**: Computer Architecture and Assembly Language (DOC_CS105)
**assessment_scheme**: Chuyên cần: 10%, Lab thực hành Linux: 20%, Thi giữa kỳ: 30%, Thi cuối kỳ: 40%

---

## 1. Mô tả học phần
Nghiên cứu cấu trúc hệ điều hành: Quản lý tiến trình (Process), tiểu trình (Thread), lập lịch CPU, đồng bộ hóa (Semaphore, Mutex), Deadlock, quản lý bộ nhớ và hệ thống tập tin.

## 2. Chuẩn đầu ra học phần (Course Learning Outcomes - CLO)
- CLO1: Phân tích hoạt động của tiến trình, luồng và các thuật toán lập lịch CPU.
- CLO2: Giải quyết bài toán miền găng và đồng bộ hóa tiến trình bằng Semaphore/Mutex.
- CLO3: Hiểu cơ chế phân trang (Paging), bộ nhớ ảo và các thuật toán thay thế trang (FIFO, LRU).

## 3. Kế hoạch giảng dạy chi tiết theo tuần (Lesson Plan)
### Tuần 1: Tổng quan về Hệ điều hành: Chức năng, Kiến trúc Kernel (Monolithic vs Microkernel)
- **Tài liệu học tập / Đọc trước**: Chương 1 Tổng quan OS
- **Hoạt động & Đánh giá**: Tìm hiểu System Call và cài đặt máy ảo Linux

### Tuần 2: Tiến trình (Process): Khái niệm, Trạng thái, Khối điều khiển PCB, Lệnh fork(), exec()
- **Tài liệu học tập / Đọc trước**: Chương 2 Tiến trình
- **Hoạt động & Đánh giá**: Viết chương trình C trên Linux gọi fork() tạo tiến trình con

### Tuần 3: Tiểu trình (Thread): Đa luồng (Multithreading), Thư viện Pthreads trong C/Linux
- **Tài liệu học tập / Đọc trước**: Chương 2 Đa luồng
- **Hoạt động & Đánh giá**: Lập trình nhân ma trận đa luồng dùng POSIX Threads

### Tuần 4: Lập lịch CPU: Tiêu chí lập lịch, Thuật toán FCFS, SJF, Priority, Round Robin (RR)
- **Tài liệu học tập / Đọc trước**: Chương 3 Lập lịch CPU
- **Hoạt động & Đánh giá**: Tính thời gian chờ trung bình (Average Waiting Time) của các thuật toán

### Tuần 5: Đồng bộ hóa tiến trình (Process Synchronization): Miền găng (Critical Section)
- **Tài liệu học tập / Đọc trước**: Chương 4 Đồng bộ hoá
- **Hoạt động & Đánh giá**: Giải pháp Peterson và kiểm tra phần cứng Test-and-Set

### Tuần 6: Công cụ đồng bộ: Mutex Locks, Semaphores và Bài toán Bounded-Buffer, Diners
- **Tài liệu học tập / Đọc trước**: Chương 4 Semaphore
- **Hoạt động & Đánh giá**: Cài đặt bài toán Producer - Consumer dùng Semaphore

### Tuần 7: Thi giữa kỳ tập trung (Lập lịch CPU & Đồng bộ hóa tiến trình 60 phút)
- **Tài liệu học tập / Đọc trước**: Phòng máy thi
- **Hoạt động & Đánh giá**: Đánh giá nội dung các tuần 1-6

### Tuần 8: Tắc nghẽn (Deadlock): 4 điều kiện cần, Đồ thị cấp phát tài nguyên, Giải thuật Banker
- **Tài liệu học tập / Đọc trước**: Chương 5 Deadlock
- **Hoạt động & Đánh giá**: Chạy giải thuật Banker kiểm tra trạng thái an toàn (Safe State)

### Tuần 9: Quản lý bộ nhớ chính: Cấp phát liên tục, Phân đoạn (Segmentation) và Phân trang (Paging)
- **Tài liệu học tập / Đọc trước**: Chương 6 Quản lý bộ nhớ
- **Hoạt động & Đánh giá**: Tính địa chỉ logic sang địa chỉ vật lý qua Page Table

### Tuần 10: Bộ nhớ ảo (Virtual Memory): Phân trang theo yêu cầu (Demand Paging), Page Fault
- **Tài liệu học tập / Đọc trước**: Chương 7 Bộ nhớ ảo
- **Hoạt động & Đánh giá**: Tính số lỗi trang với các thuật toán FIFO, LRU, Optimal

### Tuần 11: Hệ thống tập tin (File System): Cấu trúc thư mục, Inode trong Linux, Lập lịch ổ đĩa
- **Tài liệu học tập / Đọc trước**: Chương 8 File System
- **Hoạt động & Đánh giá**: Mô phỏng thuật toán lập lịch đầu đọc đĩa FCFS, SSTF, SCAN

### Tuần 12: Bảo mật hệ điều hành và Ôn tập thi kết thúc học phần
- **Tài liệu học tập / Đọc trước**: Chương 9 Bảo mật OS
- **Hoạt động & Đánh giá**: Giải đáp các dạng bài tập tổng hợp


---
*Đề cương này là văn bản chính thức của Nhà trường áp dụng cho niên khóa hiện hành. Mọi thay đổi phải được Hội đồng Khoa phê duyệt.*
