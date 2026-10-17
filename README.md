# Lesson Planning Based On Subjects Assistant

Trợ lý học tập cho sinh viên môn **AIPR (AI Programming with Python)** — đề tài T3: *hỏi nội dung môn → trả lời có trích nguồn → từ chối nếu ngoài chương trình*.

Tên hệ thống có nghĩa là trợ lý hoạt động **dựa trên lesson plan (bài giảng từng tuần) của môn học**: giúp sinh viên học từ bài giảng, **không** soạn giáo án.

Hệ thống gồm:
- **Agent ReAct** tự viết (`src/agent/loop.py`), gọi LLM qua Groq.
- **MCP server** tự viết (`src/mcp/mcp_server.py`), chạy qua **stdio**; agent khám phá tool động bằng `list_tools()`.
- **RAG** trên bài giảng tuần 1–6, hai chế độ truy vấn: **dense** (cosine) và **hybrid** (0.3 × BM25 + 0.7 × cosine).
- Tool **ghi** (soạn đơn xin gia hạn ra file) luôn đi kèm bước **verify** (đọc lại file).

---

## 1. Chạy nhanh

### Yêu cầu
- Python **3.11+**
- `make` (Linux/macOS có sẵn; Windows: cài `make` hoặc chạy trực tiếp các lệnh ở mục 3)
- API key của **Groq** (LLM) và **Jina** (embedding)

### Bước 1 — Tạo file `.env`

```bash
cp .env.example .env            # Linux / macOS
Copy-Item .env.example .env     # Windows PowerShell
```

Mở `.env` và điền key:

```
GROQ_API_KEY=<key Groq>
JINA_API_KEY=<key Jina>
```

### Bước 2 — Chạy demo (lệnh duy nhất)

```bash
make setup && make demo
```

### Bước 3 — Tái lập toàn bộ số liệu đánh giá

```bash
make eval
```

> **Lần chạy đầu tiên** hệ thống sẽ embed toàn bộ bài giảng (khoảng 60 lời gọi Jina, mất vài phút) và lưu vào `data/vector_store_cache.json`. Các lần sau đọc lại cache, không embed lại.

---

## 2. Biến môi trường

| Biến | Bắt buộc | Dùng cho |
|---|---|---|
| `GROQ_API_KEY` | Có | Gọi LLM cho agent và cho giám khảo chấm điểm trong `make eval` |
| `JINA_API_KEY` | Có | Embedding bài giảng và câu hỏi (tool `search_lesson_content`) |

File `.env` đã nằm trong `.gitignore`, không commit key lên repo.

---

## 3. Các lệnh

| Lệnh | Lệnh Python tương đương | Mô tả / Công việc |
|---|---|---|
| `make setup` | `python -m pip install --upgrade pip && pip install -r requirements.txt` | Cài đặt toàn bộ thư viện cần thiết |
| `make demo` | `python -m src.agent.loop` | Chạy chương trình tương tác trực tiếp qua terminal (Single Agent ReAct) |
| `make multi` | `python -m src.agent.team_critic` | Chạy hệ thống Multi-Agent (Producer + Critic CLI) |
| `make second_host` | `python scripts/test_second_host.py` | Kiểm thử MCP Server từ Host/Client thứ hai độc lập (chứng minh tính tương thích) |
| `make eval` | `python -m tests.eval && python -m tests.compare_configs` | Đánh giá benchmark định lượng hệ thống (kèm so sánh 2 cấu hình Hybrid vs Dense) |
| `make eval ARGS="--resume"` | `python -m tests.eval --resume && python -m tests.compare_configs` | Chạy tiếp từ câu bị ngắt của lượt eval trước (ví dụ khi gặp rate limit) |
| `make eval ARGS="--limit 2"` | `python -m tests.eval --limit 2` | Chạy thử nghiệm nhanh với 2 câu đầu tiên |

Mọi lệnh phải chạy ở **thư mục gốc của repo**.

### Dùng trong `make demo`
- Gõ câu hỏi rồi Enter.
- `clear`: xóa bộ nhớ hội thoại.
- `exit` (hoặc `quit`, `q`): kết thúc phiên; hệ thống xuất CSV và đồ thị của phiên vào `logs/`.

Ví dụ câu hỏi:
```
Vì sao khi chunking tài liệu cần để các chunk gối đầu (overlap) lên nhau?
Bài tập lab của em bị nộp trễ 30 giờ. Theo quy định môn AIPR em bị trừ bao nhiêu phần trăm?
Em là Nguyễn Văn An, MSSV 24110001, bị ốm nằm viện, có giấy xác nhận y tế. Soạn giúp em đơn xin gia hạn bài Lab 1 thêm 3 ngày.
Hướng dẫn em fine-tune một mô hình ngôn ngữ bằng LoRA.   (ngoài chương trình → hệ thống từ chối)
```

---

## 4. Kết quả được tạo ra

| Đường dẫn | Nội dung |
|---|---|
| `logs/run_<thời gian>.jsonl` | Log mỗi lần chạy: câu hỏi, từng bước (thought, tool, tham số, kết quả), câu trả lời, **USAGE** (`calls`, `prompt_tokens`, `completion_tokens`, `max_prompt`) |
| `tests/logs/eval_run_<n>_<thời gian>.json` | Kết quả chi tiết từng câu của `make eval` và bảng chỉ số tổng hợp |
| `tests/logs/eval_run_<n>_<thời gian>.csv` | Cùng dữ liệu dạng bảng |
| `tests/logs/eval_run_<n>_<thời gian>_chart.png` | Đồ thị chỉ số theo từng câu |
| `tests/logs/comparison_hybrid_vs_dense.json` | So sánh 2 cấu hình truy vấn |
| `data/drafts/draft_submission_<MSSV>.txt` | Bản nháp đơn xin gia hạn (chỉ là bản nháp, không nộp) |
| `data/vector_store_cache.json` | Cache embedding (không commit) |

Các chỉ số trong `make eval`: faithfulness, answer relevance, context precision (LLM chấm), tỉ lệ từ chối đúng với câu ngoài chương trình, tỉ lệ verify sau khi ghi, tỉ lệ trích nguồn đúng, token và thời gian từng câu.

---

## 5. Kiến trúc

```
Sinh viên ──► Agent loop (src/agent/loop.py) ──► LLM (Groq, src/agent/call_llm.py)
                   │
                   │  MCP client  ── stdio ──  MCP server (src/mcp/mcp_server.py)
                                                   │
                                   ┌───────────────┼──────────────────┐
                                   ▼               ▼                  ▼
                           RAG (src/rag/)   Tool tính theo quy chế   Ghi / đọc file
                           bài giảng W1–W6                           data/drafts/
```

### MCP tools

| Tool | Loại | Vai trò |
|---|---|---|
| `search_lesson_content` | Đọc | Tìm đoạn bài giảng liên quan (dense/hybrid, lọc theo tuần); trả cờ `found` để từ chối khi ngoài chương trình |
| `get_curriculum_week` | Đọc | Tóm tắt mục tiêu và cấu trúc buổi học của một tuần |
| `get_gotchas_and_best_practices` | Đọc | Lỗi sinh viên hay mắc của một tuần |
| `calculate_course_grade` | Tính | Điểm tổng kết theo trọng số trong quy chế |
| `check_late_submission_penalty` | Tính | Mức trừ điểm nộp trễ theo quy chế |
| `calculator` | Tính | Tính biểu thức số học an toàn (không dùng `eval`) |
| `draft_template_submission` | **Ghi** | Soạn đơn xin gia hạn ra file `.txt` |
| `read_draft` | Đọc | Đọc lại bản nháp để **verify** sau khi ghi |

### Cấu trúc thư mục

```
data/
  lecture-notes/        bài giảng tuần 1–6 (bản cho sinh viên)
  Policy/               quy chế môn học
  prompts/              system prompt, critic prompt
  templates/            mẫu đơn xin gia hạn
src/
  agent/                vòng lặp agent, gọi LLM, Producer–Critic
  mcp/                  MCP server
  rag/                  nạp tài liệu, chunking, embedding, vector store, hybrid retriever, chỉ số đánh giá
tests/                  testset, eval, so sánh cấu hình
scripts/                client MCP thứ hai
logs/                   log các lần chạy
```

---

## 6. Dữ liệu

Kho tri thức gồm 7 tài liệu, mỗi tài liệu có khối metadata `doc_id`, `title`, `version`, `date`, `owner` ở đầu file:

| `doc_id` | Tài liệu |
|---|---|
| `LP_W01` … `LP_W06` | Bài giảng tuần 1–6 (tiếng Anh), bản cho sinh viên: đã bỏ đáp án câu hỏi kiểm tra, checklist của giảng viên và hướng dẫn chấm thi |
| `POLICY_AIPR` | Quy chế môn học (tiếng Việt) |

**Khi sửa dữ liệu** trong `data/lecture-notes/` hoặc `data/Policy/`, hãy xóa `data/vector_store_cache.json` để hệ thống embed lại từ dữ liệu mới.

---

## 7. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Chạy lệnh ở thư mục gốc của repo |
| Lỗi 429 (rate limit) từ Groq khi chạy `make eval` | Hệ thống tự chờ và thử lại; nếu vẫn dừng, chạy `make eval ARGS="--resume"` để tiếp tục |
| Kết quả tìm kiếm không phản ánh dữ liệu mới | Xóa `data/vector_store_cache.json` |
| Windows không có lệnh `make` | Chạy trực tiếp: `pip install -r requirements.txt`, `python -m src.agent.loop`, `python -m tests.eval`, `python -m tests.compare_configs` |

---

## 8. Khai báo sử dụng AI

Xem `ai-usage.md`.
