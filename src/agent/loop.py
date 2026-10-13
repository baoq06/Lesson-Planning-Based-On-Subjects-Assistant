import os
import sys
import json
import asyncio
from typing import Any, Optional
from dotenv import load_dotenv
from rich import print as rprint
from groq import Groq
from pathlib import Path
from src.mcp.mcp_server import mcp
from src.schemas import ToolCall, AgentStep, AgentResponse

MODEL = "qwen/qwen3.8-27b"
MAX_STEPS = 7

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "data" / "system_prompt.txt"

def load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    fallback = REPO_ROOT / "data" / "prompts" / "system_prompt.txt"
    if fallback.exists():
        return fallback.read_text(encoding="utf-8").strip()
    return "Bạn là trợ lý AI thông minh chuyên hỗ trợ môn học AIPR."

SYSTEM_PROMPT = load_system_prompt()


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

MAX_HISTORY_TURNS = 4  

def trim_history(history: list[dict[str, Any]], max_turns: int = MAX_HISTORY_TURNS) -> list[dict[str, Any]]:
    if not history:
        return []

    max_messages = max_turns * 2
    return history[-max_messages:]

async def safe_execute_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    try:
        result = await mcp.call_tool(tool_name, arguments)
        
        if result and hasattr(result, "content") and result.content:
            texts = [item.text for item in result.content if hasattr(item, "text")]
            return "\n".join(texts) if texts else "Không có kết quả trả về."
        return "Công cụ đã chạy nhưng không có dữ liệu trả về."
        
    except Exception as e:
        return f"Error executing tool '{tool_name}': {type(e).__name__} - {str(e)}"

async def run_agent_loop(
    question: str, 
    max_steps: int = MAX_STEPS,
    history: Optional[list[dict[str, Any]]] = None
) -> AgentResponse:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    mcp_tools = await mcp.list_tools()
    tools = get_mcp_tools_schema(mcp_tools)

    trimmed_memory = trim_history(history or [])

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *trimmed_memory,
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

            args_for_mcp = dict(fn_args)
            extracted_thought = args_for_mcp.pop("thought", None)
            if extracted_thought:
                thought = extracted_thought
                rprint(f"[bold magenta]Extracted Thought:[/bold magenta] {thought}")

            rprint(f"[bold blue]Action:[/bold blue] {fn_name}({args_for_mcp})")
            observation = await safe_execute_mcp_tool(fn_name, args_for_mcp)

            rprint(f"[bold green]Observation:[/bold green] {observation[:150]}...")
            step_record = AgentStep(
                step_num=step_num,
                thought=str(thought),
                tool_call=ToolCall(id=tool_call.id, name=fn_name, arguments=args_for_mcp),
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

def ask_agent(
    question: str, 
    max_steps: int = MAX_STEPS,
    history: Optional[list[dict[str, Any]]] = None
) -> AgentResponse:
    return asyncio.run(run_agent_loop(question=question, max_steps=max_steps, history=history))

if __name__ == "__main__":
    rprint("[bold green]====================================================[/bold green]")
    rprint("[bold green]   TRỢ LÝ BÀI GIẢNG & QUY CHẾ AIPR (INTERACTIVE CLI)[/bold green]")
    rprint("[bold green]====================================================[/bold green]")
    rprint("[dim]Gõ câu hỏi và nhấn Enter. Gõ 'exit' để thoát. Gõ 'clear' để xóa nhớ.[/dim]\n")

    conversation_history: list[dict[str, Any]] = []

    while True:
        try:
            user_question = input("\nBạn: ").strip()
            if not user_question:
                continue
            if user_question.lower() in ["exit", "quit", "q"]:
                rprint("[yellow]Tạm biệt bạn![/yellow]")
                break
            if user_question.lower() == "clear":
                conversation_history.clear()
                rprint("[yellow]Đã làm mới bộ nhớ hội thoại.[/yellow]")
                continue
            
            response = ask_agent(user_question, history=conversation_history)

            if response.answer and not response.answer.startswith("Dừng:"):
                conversation_history.append({"role": "user", "content": user_question})
                conversation_history.append({"role": "assistant", "content": response.answer})

        except KeyboardInterrupt:
            rprint("\n[yellow]Đã ngắt chương trình. Hẹn gặp lại![/yellow]")
            break