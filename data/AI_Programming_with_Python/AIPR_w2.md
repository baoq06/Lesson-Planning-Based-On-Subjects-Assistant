# Tuần 2: LLM API Access & Quản lý Prompt (Communication Layer)

## 1. Thông tin chung & Chuẩn đầu ra (Learning Objectives - A2.1 đến A2.4)
- **A2.1**: Gọi Chat Completions qua OpenAI-compatible API client (Groq SDK / OpenAI SDK).
- **A2.2**: Xử lý phản hồi dạng Streaming, ghép nối từng token delta thành văn bản hoàn chỉnh.
- **A2.3**: Xử lý lỗi API ngoại lệ (RateLimitError 429, timeouts, network) bằng cơ chế Retry với Exponential Backoff và Jitter.
- **A2.4**: Thiết kế Prompt Templates có tham số hóa, phân tách System Prompt và User Prompt.

## 2. Lý thuyết nền tảng & Kiến trúc hệ thống
### 2.1. Cấu trúc yêu cầu & Các siêu tham số cốt lõi (Three Knobs)
Mỗi lời gọi API LLM được điều chỉnh qua 3 tham số kỹ thuật then chốt:
- `temperature`: Điều khiển độ phân tán xác suất của phân phối từ tiếp theo.
  - $0.0$: Xác định (deterministic), ưu tiên lấy từ có xác suất cao nhất -> Bắt buộc dùng cho RAG trích xuất, toán học và gọi công cụ (Tool calling).
  - $>0.7$: Đa dạng, sáng tạo, dùng cho viết email, thơ hoặc thảo luận mở.
- `max_tokens`: Giới hạn trần số lượng token mà LLM được phép sinh ra trong một câu trả lời.
- `top_p` (Nucleus Sampling): Giới hạn việc lấy mẫu trong tập các từ có tổng xác suất tích lũy đạt ngưỡng $p$.

### 2.2. Chiến lược xử lý sự cố mạng & Giới hạn tốc độ (Rate Limit Resilience)
- **Exponential Backoff kết hợp Random Jitter**:
  Khi gặp lỗi HTTP 429 (Too Many Requests), việc retry ngay lập tức sẽ làm sập mạng hoặc bị khóa API. Công thức chờ lũy thừa kết hợp nhiễu ngẫu nhiên:
  $$\text{delay} = 2^{\text{attempt}} + \text{random.uniform}(0.1, 0.5)$$
- **Phân loại lỗi HTTP theo chuẩn Production**:
  - *Retryable (Có thể thử lại)*: Mã `429` (Rate Limit), `500` (Internal Server Error), `502` (Bad Gateway), `503` (Service Unavailable), `APIConnectionError`.
  - *Non-retryable (Dừng ngay lập tức)*: Mã `401` (Sai API Key), `400` (Prompt sai cú pháp / JSON Schema không hợp lệ), `404` (Sai tên model).

## 3. Các bài tập Lab & Mã nguồn mẫu từng bước (Step-by-step Exercises)
### Exercise 2.1 — Chat Completion cơ bản (test_api.py)
```python
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[
        {"role": "system", "content": "Bạn là trợ lý học tập môn AIPR."},
        {"role": "user", "content": "Nêu 3 ưu điểm của Dataclass trong Python."}
    ],
    temperature=0.0
)
print(response.choices[0].message.content)
```

### Exercise 2.2 — Streaming Response (streaming.py)
```python
def stream_llm_response(prompt: str):
    stream = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    full_content = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        full_content.append(delta)
    print()
    return "".join(full_content)
```

### Exercise 2.3 — ChatSession có Backoff & Token Logging (lab2_chat_session.py)
```python
import time, random, json
from groq import RateLimitError

def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=messages,
                temperature=0.0
            )
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0.1, 0.5)
            time.sleep(wait)
```

## 4. Các bẫy thường gặp & Nguyên tắc thiết kế (Gotchas & Best Practices)
- **Bẫy Hardcode API Key**: Tuyệt đối không lưu chuỗi khóa trong mã nguồn; bắt buộc đọc qua `os.getenv("GROQ_API_KEY")` và nạp bằng `.env`.
- **Bẫy Tràn bộ nhớ hội thoại**: Context window của LLM có hạn; lịch sử chat không được nhồi nhét vô tận mà phải áp dụng kỹ thuật Sliding Window hoặc tóm tắt hội thoại.

## 5. Nhiệm vụ tự học & Đánh giá năng lực (Independent Labs)
- Xây dựng module `LLMClient` có bộ đếm token sử dụng (`prompt_tokens`, `completion_tokens`) và ghi audit log ra file JSONL phục vụ tính toán chi phí.
