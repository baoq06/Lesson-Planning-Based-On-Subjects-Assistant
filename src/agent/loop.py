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

SYSTEM_PROMPT = """You solve problems step by step using the ReAct pattern.
At each step reply in EXACTLY this format, then stop and wait for my reply:

Thought: <what you are thinking>
Action: <one action from the list below>

When you know the answer, reply with:

Final Answer: <the answer>

Available actions:
- population(<country>) : look up the population of a country. Example: population(vietnam)
- area(<country>) : look up the land area of a country in square kilometers. Example: area(vietnam)
- calculator(<expression>) : compute arithmetic. Example: calculator(330 * 100)

Rules:
- One action per step.
- Always use the available actions to look up facts; do not answer from memory.

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
    rprint(f"\n[bold cyan]🚀 Bắt đầu xử lý câu hỏi:[/bold cyan] {question}")

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
        # 2. KIỂM TRA ĐIỂM DỪNG: Không còn yêu cầu tool -> Đạt Final Answer
        if not msg.tool_calls:
            final_answer = msg.content or "Không có câu trả lời."
            rprint(f"[bold green] Final Answer:[/bold green]\n{final_answer}")
            return AgentResponse(answer=final_answer, steps=steps)
        thought = msg.content or f"Cần kích hoạt {len(msg.tool_calls)} công cụ."
        rprint(f"[bold magenta] Thought:[/bold magenta] {thought}")
        # 3. EXECUTE & OBSERVE: Duyệt qua từng tool call do LLM yêu cầu
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = parse_tool_args(tool_call.function.arguments)
            rprint(f"[bold blue] Action:[/bold blue] {fn_name}({fn_args})")
            # Thực thi an toàn qua MCP
            observation = await safe_execute_mcp_tool(fn_name, fn_args)
            rprint(f"[bold green] Observation:[/bold green] {observation[:150]}...")
            # Lưu vết vào cấu trúc AgentStep của schemas.py
            step_record = AgentStep(
                step_num=step_num,
                thought=thought,
                tool_call=ToolCall(id=tool_call.id, name=fn_name, arguments=fn_args),
                observation=observation
            )
            steps.append(step_record)
            # 4. REPEAT: Nhét observation vào messages để LLM đọc ở vòng lặp sau
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": observation
            })
    # Nếu chạy hết max_steps mà chưa xong:
    stop_warning = f"⚠️ Dừng: Đã đạt giới hạn tối đa {max_steps} bước suy luận."
    rprint(f"[bold red]{stop_warning}[/bold red]")
    return AgentResponse(answer=stop_warning, steps=steps)

def ask_agent(question: str, max_steps: int = MAX_STEPS) -> AgentResponse:

    return asyncio.run(run_agent_loop(question=question, max_steps=max_steps))
if __name__ == "__main__":
    test_q = "Điểm chuyên cần 9, bài lab 8.5, test1 7.5, test2 8, đồ án cuối kỳ 9, vắng 5%. Hãy tính điểm môn AIPR giúp tôi?"
    
    result = ask_agent(test_q)
    print("\nKết quả hoàn chỉnh:", result.answer)