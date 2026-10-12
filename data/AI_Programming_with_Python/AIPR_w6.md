# Tuần 6: Agent Loop, ReAct Pattern & Agent Memory

## 1. Thông tin chung & Chuẩn đầu ra (Learning Objectives - A6.1 đến A6.4)
- **A6.1**: Định nghĩa công cụ (Tool Definition) theo JSON Schema chuẩn Function Calling để LLM gọi tự động.
- **A6.2**: Xây dựng vòng lặp Agent Loop (Call -> Execute -> Observe -> Repeat) có chặn giới hạn số bước (`iteration_cap=6`).
- **A6.3**: Hiện thực hóa mẫu **ReAct** (Thought -> Action -> Observation) để suy luận giải quyết bài toán phức tạp nhiều bước.
- **A6.4**: Quản lý bộ nhớ hội thoại (Sliding Window Memory) và bảo đảm thực thi công cụ an toàn (Safe Tool Execution).

## 2. Lý thuyết nền tảng & Kiến trúc hệ thống
### 2.1. Bản chất của ReAct Pattern (Reasoning + Acting)
ReAct kết hợp sức mạnh của việc suy luận logic nội tại với khả năng tương tác với môi trường bên ngoài:
- **Thought (Tư duy)**: Agent tự đặt câu hỏi và lên kế hoạch: *"Mình đang cần thông tin gì? Bước tiếp theo nên làm gì?"*
- **Action (Hành động)**: Agent kích hoạt một công cụ cụ thể với các tham số tương ứng (ví dụ: `search_lesson_plan(week=4)`).
- **Observation (Quan sát)**: Hệ thống thực thi công cụ và trả kết quả thực tế về cho Agent làm dữ kiện cho bước suy nghĩ tiếp theo.
- Chu trình lặp lại cho đến khi Agent nhận thấy đã đủ bằng chứng để xuất ra **Final Answer**.

### 2.2. Kiến trúc Agent Loop & Phòng vệ an toàn (Safe Tool Execution)
- **Agent Loop Termination**:
  Nếu không có điểm dừng, Agent có thể rơi vào vòng lặp vô tận (Infinite Tool Call Loop). Vòng lặp `for step in range(MAX_STEPS)` với `MAX_STEPS = 6` là bắt buộc để khống chế chi phí và tài nguyên.
- **Nguyên tắc Safe Tool Execution**:
  Tuyệt đối không để ngoại lệ của Tool làm crash Agent. Toàn bộ mã gọi Tool phải được bọc trong `try...except`. Nếu xảy ra lỗi (ví dụ: tham số sai, mất kết nối), trả về chuỗi:
  `"Error executing tool '<name>': <error_message>"`
  LLM sẽ đọc chuỗi lỗi này ở vòng lặp sau như một Observation và tự động sửa đổi tham số để gọi lại.
- **Cấm kỵ bảo mật**: Tuyệt đối không dùng `eval()` hoặc thực thi mã code không đáng tin cậy do LLM sinh ra.

## 3. Các bài tập Lab & Mã nguồn mẫu từng bước (Step-by-step Exercises)
### Exercise 6.1 — Định nghĩa Tool bằng JSON Schema (ex1_1_tool_call_v1.py)
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_lesson_plan",
            "description": "Tra cứu giáo trình bài học của môn AIPR theo từ khóa.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Câu hỏi cần tra cứu"},
                    "week": {"type": "integer", "description": "Tuần học cụ thể từ 1 đến 8"}
                },
                "required": ["query"]
            }
        }
    }
]
```

### Exercise 6.2 — Safe Tool Execution bọc ngoại lệ (src/agent/react_agent.py)
```python
async def safe_execute_tool(tool_fn, **kwargs) -> str:
    try:
        result = await tool_fn(**kwargs)
        return str(result)
    except Exception as e:
        # Bắt toàn bộ ngoại lệ và trả chuỗi mô tả lỗi để LLM tự khắc phục
        return f"Error executing tool: {type(e).__name__} - {str(e)}"
```

### Exercise 6.3 — Vòng lặp ReAct hoàn chỉnh (ex2_1_react.py)
```python
def run_react_agent(user_question: str, max_steps: int = 6):
    messages = [
        {"role": "system", "content": "Bạn là ReAct Agent môn AIPR. Hãy suy luận từng bước."},
        {"role": "user", "content": user_question}
    ]
    for step in range(max_steps):
        resp = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=messages,
            tools=tools,
            temperature=0.0
        )
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content  # Final Answer

        for tool_call in msg.tool_calls:
            # Thực thi và nhét Observation vào messages
            obs = execute_tool(tool_call.function.name, tool_call.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": obs
            })
    return "Đã vượt quá số bước suy luận tối đa."
```

## 4. Các bẫy thường gặp & Nguyên tắc thiết kế (Gotchas & Best Practices)
- **Bẫy Tool Arguments không phải JSON hợp lệ**: Thỉnh thoảng LLM trả về chuỗi JSON bị lỗi cú pháp; phải dùng `json.loads` trong khối `try...except` để bắt lỗi parse.
- **Bẫy Context Blowup**: Qua mỗi bước ReAct, tin nhắn `messages` dài ra gấp bội; cần lọc bớt hoặc cắt tỉa các observation quá dài.

## 5. Nhiệm vụ tự học & Đánh giá năng lực (Independent Labs)
- Kiểm thử khả năng tự sửa lỗi của Agent bằng cách cố tình truyền tham số sai (Fault Injection) và quan sát Agent tự điều chỉnh tham số ở bước sau.
