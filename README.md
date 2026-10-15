# AIPR Teaching & Lesson Planning Assistant
Trợ lý AI hỗ trợ học tập, tra cứu quy chế môn học và lập kế hoạch bài giảng cho môn **AIPR (AI Programming with Python)** theo mô hình **ReAct Agent** tích hợp **MCP Server (Model Context Protocol)** và **Hybrid RAG (Dense + BM25)**.
---
## 1. Cài đặt & Khởi chạy nhanh (Lệnh duy nhất)
Chỉ cần chạy một dòng lệnh sau để cài đặt môi trường và khởi động giao diện trợ lý:
```bash
make setup && make demo
```

- make setup: Cài đặt/cập nhật toàn bộ thư viện phụ thuộc (groq, mcp, rank_bm25, rich,...).
- make demo: Chạy vòng lặp tương tác Agent trực tiếp trên Terminal CLI.
