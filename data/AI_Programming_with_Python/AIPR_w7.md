# Tuần 7: Model Context Protocol (MCP) & Tích hợp Công cụ Ngoài

## 1. Thông tin chung & Chuẩn đầu ra (Learning Objectives - A7.1 đến A7.6)
- **A7.1**: Giải thích kiến trúc Host / Client / Server, 3 Primitives (Resources, Prompts, Tools), và 2 Transports (stdio, SSE).
- **A7.2**: Thêm và cấu hình các MCP Server cộng đồng từ file config mà không cần viết lại mã nguồn ứng dụng.
- **A7.3**: Kiểm tra và kiểm toán (Audit) Tool inventory, JSON schema và ranh giới bảo mật (Trust boundary).
- **A7.4**: Viết Python MCP Client kết nối qua `stdio`, khám phá công cụ bằng `list_tools()` và gọi bằng `call_tool()`.
- **A7.5**: Tự viết một MCP Server chuẩn, bọc VectorStore thành công cụ MCP và kiểm thử bằng JSON-RPC probe.
- **A7.6**: Xây dựng ReAct Agent động với tập công cụ hoàn toàn lấy từ `list_tools()` mà không hardcode trong mã nguồn.

## 2. Lý thuyết nền tảng & Kiến trúc hệ thống
### 2.1. Tại sao Model Context Protocol (MCP) là tất yếu?
- **Vấn đề $N \times M$**: Trước khi có MCP, nếu có $N$ mô hình AI/Host và $M$ công cụ/nguồn dữ liệu, lập trình viên phải viết $N \times M$ bộ adapter riêng biệt.
- **Giải pháp MCP**: Chuẩn hóa một giao thức thống nhất (dựa trên JSON-RPC 2.0). Mọi nguồn dữ liệu trở thành MCP Server; mọi ứng dụng AI trở thành MCP Client. Số lượng kết nối giảm về $N + M$.

### 2.2. Các thành phần & Khái niệm cốt lõi của MCP
- **Ba vai trò (Three Roles)**:
  - **Host**: Ứng dụng điều phối cấp cao (Claude Desktop, IDE, hoặc Python ReAct Agent).
  - **Client**: Module thiết lập kết nối $1:1$ với một Server cụ thể.
  - **Server**: Ứng dụng độc lập phơi bày dữ liệu và tính năng thông qua giao thức chuẩn.
- **Ba Primitive cốt lõi**:
  - `Tools`: Các hàm hành động mà LLM có thể gọi thực thi (tạo Side Effect hoặc tính toán).
  - `Resources`: Dữ liệu ngữ cảnh chỉ đọc (tệp tin, nhật ký, schema cơ sở dữ liệu).
  - `Prompts`: Các mẫu lời nhắc dựng sẵn do server cung cấp.
- **Hai phương thức truyền tải (Two Transports)**:
  - `stdio`: Giao tiếp qua luồng nhập/xuất chuẩn của tiến trình cục bộ (Local process stdin/stdout).
  - `sse` (Server-Sent Events): Giao tiếp qua HTTP dành cho các dịch vụ từ xa.

### 2.3. Ranh giới tin cậy & Quy tắc Stdio Transport bắt buộc
- **Ranh giới tin cậy (Trust Boundary)**: Server nằm ngoài tầm kiểm soát trực tiếp của LLM. Client phải audit schema và tham số trước khi thực thi.
- **Quy tắc vàng Stdio**: Tuyệt đối **KHÔNG ĐƯỢC DÙNG `print()`** bừa bãi trong MCP Server chạy qua `stdio` vì sẽ làm hỏng cấu trúc gói tin JSON-RPC trên luồng stdout. Toàn bộ log phải chuyển sang `sys.stderr`.

## 3. Các bài tập Lab & Mã nguồn mẫu từng bước (Step-by-step Exercises)
### Exercise 7.1 — Xây dựng MCP Server cho môn học (src/mcp_service/server.py)
```python
from mcp.server import MCPServer
import sys

mcp = MCPServer("aipr-study-assistant")

@mcp.tool()
def register_study_task(task_id: str, week: int, title: str, due_date: str) -> str:
    """Đăng ký một nhiệm vụ học tập có kiểm chứng (Write Side Effect)."""
    # Ghi log chuẩn qua stderr, bảo vệ stdout cho JSON-RPC
    print(f"[MCP-LOG] Đăng ký task {task_id} cho tuần {week}", file=sys.stderr)
    return f"Đã ghi nhận nhiệm vụ '{title}' vào hệ thống lưu trữ bền vững."

@mcp.tool()
def get_study_tasks(week: int = 0) -> str:
    """Truy xuất danh sách nhiệm vụ đã đăng ký."""
    return "Danh sách nhiệm vụ học tập hiện có: [HW_W04, HW_W07]"
```

### Exercise 7.2 — Python MCP Client kết nối qua Stdio (ex2_1_mcp_client.py)
```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_client():
    server_params = StdioServerParameters(command="python", args=["src/mcp_service/server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Dynamic Tool Discovery
            tools = await session.list_tools()
            print(f"Khám phá được {len(tools.tools)} công cụ từ server.")
            # Thực thi công cụ
            res = await session.call_tool("register_study_task", {"task_id": "T01", "week": 4, "title": "Lab BM25", "due_date": "2026-10-25"})
            print("Kết quả:", res.content[0].text)
```

## 4. Các bẫy thường gặp & Nguyên tắc thiết kế (Gotchas & Best Practices)
- **Bẫy Vỡ luồng JSON-RPC do Print**: Bất kỳ thư viện nào vô tình in ra stdout (ví dụ: cảnh báo nạp mô hình) đều làm crash kết nối MCP stdio.
- **Bẫy Không có Side Effect**: Đề tài dùng MCP mà chỉ viết tool đọc (Read-only) sẽ bị đánh trượt tiêu chí đồ án; bắt buộc phải có tool Write lưu xuống đĩa hoặc database.

## 5. Nhiệm vụ tự học & Đánh giá năng lực (Independent Labs)
- Xây dựng bài kiểm tra JSON-RPC probe độc lập để xác minh tính ổn định của MCP Server trước khi cho Agent kết nối vào.
