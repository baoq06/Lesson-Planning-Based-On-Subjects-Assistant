# [DOC_CS109] Artifical Intelligence (Trí tuệ nhân tạo)

**doc_id**: DOC_CS109
**title**: Artifical Intelligence
**course_code**: CS109
**credits**: 3 tín chỉ
**date_version**: 2024-2025 / v2.1
**owner**: Khoa Công nghệ Thông tin
**prerequisites**: Discrete Mathematics and Graph Theory (DOC_MATH104), Data Structures and Algorithms (DOC_CS102)
**assessment_scheme**: Chuyên cần: 10%, Lab thuật toán: 20%, Thi giữa kỳ: 20%, Đồ án nhóm cuối kỳ: 50%

---

## 1. Mô tả học phần
Nghiên cứu các phương pháp trí tuệ nhân tạo kinh điển: Không gian trạng thái, Tìm kiếm mù (BFS, DFS, UCS), Tìm kiếm Heuristic (A*, IDA*), Bài toán thỏa mãn ràng buộc (CSP), Tìm kiếm đối kháng (Minimax, Alpha-Beta Pruning), Biểu diễn tri thức và Học máy cơ bản.

## 2. Chuẩn đầu ra học phần (Course Learning Outcomes - CLO)
- CLO1: Mô hình hóa bài toán thực tế dưới dạng không gian trạng thái.
- CLO2: Cài đặt thuật toán A* với hàm Heuristic chấp nhận được (Admissible).
- CLO3: Áp dụng thuật toán Minimax và cắt tỉa Alpha-Beta vào trò chơi đối kháng (Cờ caro, Cờ vua).

## 3. Kế hoạch giảng dạy chi tiết theo tuần (Lesson Plan)
### Tuần 1: Tổng quan về Trí tuệ nhân tạo, Định nghĩa Agent thông minh và Môi trường (PEAS)
- **Tài liệu học tập / Đọc trước**: Chương 1 Tổng quan AI
- **Hoạt động & Đánh giá**: Phân loại môi trường bài toán tự lái xe, chơi cờ

### Tuần 2: Giải quyết vấn đề bằng Tìm kiếm: Tìm kiếm mù (BFS, DFS, Uniform Cost Search)
- **Tài liệu học tập / Đọc trước**: Chương 2 Tìm kiếm không thông tin
- **Hoạt động & Đánh giá**: Cài đặt bài toán 8-Puzzle bằng BFS và UCS

### Tuần 3: Tìm kiếm Heuristic (Informed Search): Greedy Best-First Search và Thuật toán A*
- **Tài liệu học tập / Đọc trước**: Chương 3 Tìm kiếm có thông tin
- **Hoạt động & Đánh giá**: Thiết kế hàm Heuristic khoảng cách Manhattan cho A*

### Tuần 4: Tính chấp nhận được (Admissibility) và Tính nhất quán (Consistency) của Heuristic
- **Tài liệu học tập / Đọc trước**: Chương 3 Tính chất Heuristic
- **Hoạt động & Đánh giá**: Chứng minh tính tối ưu của thuật toán A*

### Tuần 5: Bài toán thỏa mãn ràng buộc (CSP - Constraint Satisfaction Problems): Backtracking, AC-3
- **Tài liệu học tập / Đọc trước**: Chương 4 CSP
- **Hoạt động & Đánh giá**: Giải bài toán Sudoku và bài toán xếp 8 quân hậu

### Tuần 6: Tìm kiếm đối kháng: Trò chơi đối kháng 2 người, Thuật toán Minimax
- **Tài liệu học tập / Đọc trước**: Chương 5 Game Playing
- **Hoạt động & Đánh giá**: Cài đặt bot chơi Tic-Tac-Toe bằng Minimax

### Tuần 7: Cắt tỉa Alpha-Beta (Alpha-Beta Pruning) và Đánh giá trạng thái tĩnh
- **Tài liệu học tập / Đọc trước**: Chương 5 Alpha-Beta
- **Hoạt động & Đánh giá**: Tối ưu hóa bot cờ caro gom nhánh không khả thi

### Tuần 8: Thi giữa kỳ lý thuyết và thuật toán AI (60 phút)
- **Tài liệu học tập / Đọc trước**: Phòng máy thi
- **Hoạt động & Đánh giá**: Đánh giá A* và Alpha-Beta

### Tuần 9: Biểu diễn tri thức và Lập luận logic: Mạng ngữ nghĩa, Hệ chuyên gia dựa trên luật
- **Tài liệu học tập / Đọc trước**: Chương 6 Biểu diễn tri thức
- **Hoạt động & Đánh giá**: Xây dựng hệ luật suy diễn tiến (Forward Chaining) chẩn đoán bệnh

### Tuần 10: Lý thuyết quyết định và Quá trình quyết định Markov (MDP - Markov Decision Process)
- **Tài liệu học tập / Đọc trước**: Chương 7 MDP
- **Hoạt động & Đánh giá**: Thuật toán Value Iteration tìm chính sách tối ưu

### Tuần 11: Nhập môn Học tăng cường (Reinforcement Learning): Q-Learning
- **Tài liệu học tập / Đọc trước**: Chương 8 Q-Learning
- **Hoạt động & Đánh giá**: Huấn luyện agent thoát mê cung bằng bảng Q-Table

### Tuần 12: Báo cáo đồ án nhóm Trí tuệ nhân tạo và Tổng kết môn học
- **Tài liệu học tập / Đọc trước**: Hội đồng chấm đồ án
- **Hoạt động & Đánh giá**: Trình diễn bot game / ứng dụng tìm đường và bảo vệ


---
*Đề cương này là văn bản chính thức của Nhà trường áp dụng cho niên khóa hiện hành. Mọi thay đổi phải được Hội đồng Khoa phê duyệt.*
