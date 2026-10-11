# [DOC_CS112] Internet of Things (Internet vạn vật (IoT))

**doc_id**: DOC_CS112
**title**: Internet of Things
**course_code**: CS112
**credits**: 3 tín chỉ
**date_version**: 2024-2025 / v2.1
**owner**: Khoa Công nghệ Thông tin
**prerequisites**: Basic Electronics (DOC_EE101), Networking Essentials (DOC_NET101)
**assessment_scheme**: Chuyên cần: 10%, Lab vi điều khiển: 20%, Thi giữa kỳ: 20%, Đồ án sản phẩm IoT cuối kỳ: 50%

---

## 1. Mô tả học phần
Nghiên cứu kiến trúc hệ thống IoT, vi điều khiển nhúng (ESP32, Raspberry Pi), cảm biến và cơ cấu chấp hành, giao thức truyền thông IoT (MQTT, CoAP, BLE, LoRaWAN) và điện toán đám mây cho IoT.

## 2. Chuẩn đầu ra học phần (Course Learning Outcomes - CLO)
- CLO1: Lập trình vi điều khiển ESP32 đọc dữ liệu từ các cảm biến môi trường.
- CLO2: Truyền nhận dữ liệu qua giao thức MQTT lên IoT Cloud Broker.
- CLO3: Thiết kế và hiện thực hóa một giải pháp Smart Home hoặc Nông nghiệp thông minh.

## 3. Kế hoạch giảng dạy chi tiết theo tuần (Lesson Plan)
### Tuần 1: Tổng quan về Internet of Things: Kiến trúc 4 lớp IoT, Ứng dụng thực tế và Thách thức
- **Tài liệu học tập / Đọc trước**: Chương 1 Tổng quan IoT
- **Hoạt động & Đánh giá**: Khảo sát các nền tảng IoT phổ biến (ThingSpeak, Adafruit IO, Blynk)

### Tuần 2: Phần cứng IoT: Giới thiệu vi điều khiển ESP32, Các chân GPIO, ADC, DAC, PWM
- **Tài liệu học tập / Đọc trước**: Chương 2 Phần cứng ESP32
- **Hoạt động & Đánh giá**: Cài đặt môi trường Arduino IDE và nạp chương trình chớp tắt LED

### Tuần 3: Giao tiếp với cảm biến: Cảm biến số, tương tự (DHT11, Quang trở LDR, Cảm biến mưa)
- **Tài liệu học tập / Đọc trước**: Chương 3 Cảm biến
- **Hoạt động & Đánh giá**: Đọc nhiệt độ độ ẩm và hiển thị lên màn hình LCD I2C / OLED

### Tuần 4: Điều khiển cơ cấu chấp hành: Rơ-le đóng ngắt thiết bị 220V, Động cơ Servo, Còi báo
- **Tài liệu học tập / Đọc trước**: Chương 4 Cơ cấu chấp hành
- **Hoạt động & Đánh giá**: Lập trình bật quạt tự động khi nhiệt độ vượt ngưỡng

### Tuần 5: Kết nối không dây: Wi-Fi và Bluetooth Low Energy (BLE) trên vi điều khiển ESP32
- **Tài liệu học tập / Đọc trước**: Chương 5 Kết nối không dây
- **Hoạt động & Đánh giá**: Cấu hình ESP32 ở chế độ Wi-Fi Station và Access Point

### Tuần 6: Giao thức truyền thông IoT: Giao thức MQTT (Broker, Publisher, Subscriber, QoS)
- **Tài liệu học tập / Đọc trước**: Chương 6 Giao thức MQTT
- **Hoạt động & Đánh giá**: Cài đặt Mosquitto MQTT Broker trên Raspberry Pi / máy tính

### Tuần 7: Thi giữa kỳ trên máy (Lập trình ESP32 kết nối cảm biến và xuất dữ liệu 60 phút)
- **Tài liệu học tập / Đọc trước**: Phòng máy thi
- **Hoạt động & Đánh giá**: Đánh giá kỹ năng lập trình nhúng phần cứng

### Tuần 8: Xây dựng Dashboard giám sát IoT trên Web sử dụng Node-RED và WebSockets
- **Tài liệu học tập / Đọc trước**: Chương 7 Dashboard
- **Hoạt động & Đánh giá**: Thiết kế giao diện bật tắt đèn và đồ thị nhiệt độ thời gian thực

### Tuần 9: Lưu trữ dữ liệu IoT chuỗi thời gian (Time-series Database: InfluxDB) và Grafana
- **Tài liệu học tập / Đọc trước**: Chương 8 Dữ liệu IoT
- **Hoạt động & Đánh giá**: Đẩy dữ liệu cảm biến vào InfluxDB và trực quan hóa trên Grafana

### Tuần 10: Giao thức truyền thông diện rộng tầm xa năng lượng thấp: Giới thiệu LoRa và LoRaWAN
- **Tài liệu học tập / Đọc trước**: Chương 9 Công nghệ LoRa
- **Hoạt động & Đánh giá**: Gửi gói tin từ LoRa Node về Gateway khoảng cách 1km

### Tuần 11: Bảo mật trong hệ thống IoT: Mã hóa đường truyền MQTT TLS và Xác thực thiết bị
- **Tài liệu học tập / Đọc trước**: Chương 10 An toàn IoT
- **Hoạt động & Đánh giá**: Cấu hình chứng chỉ bảo mật kết nối với AWS IoT Core

### Tuần 12: Báo cáo đồ án sản phẩm IoT hoàn chỉnh (Smart Home/Smart Garden) và Tổng kết
- **Tài liệu học tập / Đọc trước**: Hội đồng chấm đồ án
- **Hoạt động & Đánh giá**: Trưng bày mô hình phần cứng hoạt động thực tế


---
*Đề cương này là văn bản chính thức của Nhà trường áp dụng cho niên khóa hiện hành. Mọi thay đổi phải được Hội đồng Khoa phê duyệt.*
