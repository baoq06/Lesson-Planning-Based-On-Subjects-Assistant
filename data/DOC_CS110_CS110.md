# [DOC_CS110] Cloud Coputing (Điện toán đám mây)

**doc_id**: DOC_CS110
**title**: Cloud Coputing
**course_code**: CS110
**credits**: 3 tín chỉ
**date_version**: 2024-2025 / v2.1
**owner**: Khoa Công nghệ Thông tin
**prerequisites**: Operating System (DOC_CS106), Networking Essentials (DOC_NET101)
**assessment_scheme**: Chuyên cần: 10%, Lab AWS/GCP: 20%, Thi giữa kỳ: 20%, Đồ án triển khai hạ tầng cuối kỳ: 50%

---

## 1. Mô tả học phần
Nghiên cứu các mô hình dịch vụ IaaS, PaaS, SaaS, kiến trúc ảo hóa, container hóa (Docker), điều phối container (Kubernetes), hạ tầng dưới dạng mã (Terraform) và các dịch vụ đám mây công cộng AWS/Azure/GCP.

## 2. Chuẩn đầu ra học phần (Course Learning Outcomes - CLO)
- CLO1: Phân biệt các mô hình điện toán đám mây và tính toán chi phí TCO.
- CLO2: Đóng gói ứng dụng web thành Docker container và triển khai trên cụm máy chủ.
- CLO3: Thiết kế kiến trúc chịu tải cao (High Availability, Auto-scaling, Load Balancer) trên đám mây.

## 3. Kế hoạch giảng dạy chi tiết theo tuần (Lesson Plan)
### Tuần 1: Khái niệm Điện toán đám mây, 5 đặc trưng NIST, Mô hình dịch vụ IaaS, PaaS, SaaS
- **Tài liệu học tập / Đọc trước**: Chương 1 Tổng quan Cloud
- **Hoạt động & Đánh giá**: Đăng ký tài khoản AWS Free Tier và khảo sát Console

### Tuần 2: Ảo hóa (Virtualization): Hypervisor loại 1 và loại 2, So sánh Ảo hóa với Container
- **Tài liệu học tập / Đọc trước**: Chương 2 Công nghệ ảo hóa
- **Hoạt động & Đánh giá**: Cài đặt và cấu hình máy ảo EC2 trên AWS

### Tuần 3: Mạng và Bảo mật đám mây: Virtual Private Cloud (VPC), Subnet, Security Group, IAM
- **Tài liệu học tập / Đọc trước**: Chương 3 Mạng Cloud
- **Hoạt động & Đánh giá**: Thiết lập VPC với Public Subnet và Private Subnet

### Tuần 4: Dịch vụ lưu trữ đám mây: Block Storage (EBS), Object Storage (S3), File Storage (EFS)
- **Tài liệu học tập / Đọc trước**: Chương 4 Lưu trữ Cloud
- **Hoạt động & Đánh giá**: Cấu hình S3 Static Website Hosting và thiết lập chính sách Bucket Policy

### Tuần 5: Công nghệ Container hóa: Docker cơ bản, Dockerfile, Docker Image và Docker Hub
- **Tài liệu học tập / Đọc trước**: Chương 5 Docker
- **Hoạt động & Đánh giá**: Viết Dockerfile đóng gói ứng dụng web Python Flask

### Tuần 6: Docker Compose và Quản lý nhiều container liên kết (Multi-container)
- **Tài liệu học tập / Đọc trước**: Chương 5 Docker Compose
- **Hoạt động & Đánh giá**: Triển khai ứng dụng gồm Web App + Nginx + CSDL PostgreSQL bằng compose

### Tuần 7: Thi giữa kỳ trên máy (Thực hành cấu hình hạ tầng AWS 60 phút)
- **Tài liệu học tập / Đọc trước**: Phòng máy thi
- **Hoạt động & Đánh giá**: Đánh giá VPC, EC2, IAM và S3

### Tuần 8: Cân bằng tải (Load Balancer) và Tự động co giãn (Auto Scaling Group)
- **Tài liệu học tập / Đọc trước**: Chương 6 Khả năng chịu lỗi
- **Hoạt động & Đánh giá**: Mô phỏng bài kiểm tra tải Stress-test kích hoạt tạo thêm máy chủ

### Tuần 9: Giới thiệu Điều phối container với Kubernetes (K8s): Pod, Service, Deployment
- **Tài liệu học tập / Đọc trước**: Chương 7 Kubernetes
- **Hoạt động & Đánh giá**: Cài đặt Minikube và triển khai tệp YAML Deployment

### Tuần 10: Serverless Computing: AWS Lambda, API Gateway và Kiến trúc hướng sự kiện
- **Tài liệu học tập / Đọc trước**: Chương 8 Serverless
- **Hoạt động & Đánh giá**: Xây dựng hàm Lambda tự động xử lý và nén ảnh khi upload lên S3

### Tuần 11: Hạ tầng dưới dạng mã (Infrastructure as Code - IaC) với Terraform
- **Tài liệu học tập / Đọc trước**: Chương 9 IaC
- **Hoạt động & Đánh giá**: Viết mã Terraform tự động khởi tạo cụm máy chủ và VPC

### Tuần 12: Báo cáo đồ án triển khai hệ thống phân tán trên đám mây và Tổng kết
- **Tài liệu học tập / Đọc trước**: Hội đồng chấm đồ án
- **Hoạt động & Đánh giá**: Bảo vệ kiến trúc hệ thống và kiểm tra khả năng phục hồi lỗi


---
*Đề cương này là văn bản chính thức của Nhà trường áp dụng cho niên khóa hiện hành. Mọi thay đổi phải được Hội đồng Khoa phê duyệt.*
