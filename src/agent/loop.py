import os
import sys
import json
import asyncio
from typing import Any, Optional
from dotenv import load_dotenv
from rich import print as rprint
from groq import Groq
from src.mcp.mcp_server import mcp
from src.schemas import ToolCall, AgentStep, AgentResponse

MODEL = "qwen/qwen3.8-27b"
MAX_STEPS = 7

SYSTEM_PROMPT = """Bạn là trợ lý AI thông minh chuyên hỗ trợ môn học AIPR.
- Khi cần tính toán hoặc tra cứu thông tin môn học, hãy gọi các công cụ (tools) thích hợp.
- Khi đã có kết quả từ công cụ hoặc khi trả lời người dùng, hãy trả lời trực tiếp, rõ ràng và mạch lạc, không cần thêm các nhãn 'Thought:' hay 'Action:'.
"""


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

def get_mcp_tools_schema(mcp_tools: list[Any], require_thought: bool = True) -> list[dict[str, Any]]:
    formatted_tools = []

    for tool in mcp_tools:
        schema = dict(getattr(tool, "input_schema", getattr(tool, "inputSchema", {})))
        properties = dict(schema.get("properties", {}))
        required = list(schema.get("required", []))

        if require_thought:
            new_properties = {
                "thought": {
                    "type": "string",
                    "description": (
                        "Suy nghĩ ngắn gọn (1-2 câu): Tại sao gọi công cụ này? "
                        "Cần dữ liệu gì và dự định làm gì tiếp theo với kết quả?"
                    ),
                }
            }

            new_properties.update(properties)
            properties = new_properties

            if "thought" not in required:
                required.insert(0, "thought")

        formatted_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
        
    return formatted_tools

def parse_tool_args(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except Exception:
            return {}
    return {}

async def safe_execute_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    try:
        result = await mcp.call_tool(tool_name, arguments)
        
        if result and hasattr(result, "content") and result.content:
            texts = [item.text for item in result.content if hasattr(item, "text")]
            return "\n".join(texts) if texts else "Không có kết quả trả về."
        return "Công cụ đã chạy nhưng không có dữ liệu trả về."
        
    except Exception as e:
        return f"Error executing tool '{tool_name}': {type(e).__name__} - {str(e)}"

async def run_agent_loop(question: str, max_steps: int = MAX_STEPS) -> AgentResponse:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    mcp_tools = await mcp.list_tools()
    tools = get_mcp_tools_schema(mcp_tools)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    steps: list[AgentStep] = []
    rprint(f"\n[bold cyan]Bắt đầu xử lý câu hỏi:[/bold cyan] {question}")

    for step_num in range(1, max_steps + 1):
        rprint(f"\n[bold yellow]── Bước {step_num}/{max_steps} ──[/bold yellow]")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            temperature=0.1
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            final_answer = msg.content or "Không có câu trả lời."
            rprint(f"[bold green]Final Answer:[/bold green]\n{final_answer}")
            return AgentResponse(answer=final_answer, steps=steps)
        
        thought = msg.content or f"Cần kích hoạt {len(msg.tool_calls)} công cụ."
        rprint(f"[bold magenta]Thought:[/bold magenta] {thought}")

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = parse_tool_args(tool_call.function.arguments)

            rprint(f"[bold blue]Action:[/bold blue] {fn_name}({fn_args})")
            observation = await safe_execute_mcp_tool(fn_name, fn_args)

            rprint(f"[bold green]Observation:[/bold green] {observation[:150]}...")
            step_record = AgentStep(
                step_num=step_num,
                thought=thought,
                tool_call=ToolCall(id=tool_call.id, name=fn_name, arguments=fn_args),
                observation=observation
            )
            steps.append(step_record)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": observation
            })

    stop_warning = f"Dừng: Đã đạt giới hạn tối đa {max_steps} bước suy luận."
    rprint(f"[bold red]{stop_warning}[/bold red]")
    return AgentResponse(answer=stop_warning, steps=steps)

def ask_agent(question: str, max_steps: int = MAX_STEPS):
    return asyncio.run(run_agent_loop(question=question, max_steps=max_steps))

if __name__ == "__main__":
    rprint("[bold green]====================================================[/bold green]")
    rprint("[bold green]   TRỢ LÝ BÀI GIẢNG & QUY CHẾ AIPR (INTERACTIVE CLI)[/bold green]")
    rprint("[bold green]====================================================[/bold green]")
    rprint("[dim]Gõ câu hỏi của bạn và nhấn Enter. Gõ 'exit' hoặc 'q' để thoát.[/dim]\n")
    while True:
        try:
            # Nhận câu hỏi trực tiếp từ terminal
            user_question = input("\n👤 Bạn: ").strip()
            # Thoát nếu người dùng gõ exit hoặc q
            if not user_question:
                continue
            if user_question.lower() in ["exit", "quit", "q"]:
                rprint("[yellow]Tạm biệt bạn![/yellow]")
                break
            # Gọi Agent xử lý
            ask_agent(user_question)
        except KeyboardInterrupt:
            rprint("\n[yellow]Đã ngắt chương trình. Hẹn gặp lại![/yellow]")
            break