import re
import json
import os
import sys
from typing import Any, Optional
from rich import print

# from src.data_layer.models import CitationResponse
# from src.agent.llm_client import call_llm_with_retry, load_prompt

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

# def parse_citations(text: str) -> list[str]:
#     """Trích xuất mã bài học hoặc chunk ID từ câu trả lời (ví dụ: [LP_W01], [Tuần 2])."""
#     citations = re.findall(r"\[(LP_W\d+(?:_c\d+)?|Tuần\s*\d+|Week\s*\d+)\]", text, re.IGNORECASE)
#     return list(dict.fromkeys(citations))  # Loại bỏ trùng lặp giữ nguyên thứ tự


def is_refusal_answer(text: str) -> bool:
    """Kiểm tra xem câu trả lời có phải là từ chối ngoài phạm vi môn học hay không."""
    keywords = [
        "không có trong chương trình",
        "ngoài phạm vi",
        "không nằm trong lesson plan",
        "không thuộc môn học",
        "tôi chỉ hỗ trợ môn học",
        "no_answer",
    ]
    lower = text.lower()
    return any(k in lower for k in keywords)

def parse_reply(reply: str) -> tuple[str, Any]:
    final = re.search(r"Final Answer:\s*(.+)", reply, flags=re.IGNORECASE | re.DOTALL)
    if final:
        return "final", final.group(1).strip()
    action = re.search(r"Action:\s*(\w+)\s*\((.*)\)", reply, flags=re.IGNORECASE)
    if action:
        return "action", (action.group(1), action.group(2))
    return "unknown", None

# async def safe_execute_mcp_tool(client: Any, fn_name: str, fn_args: dict[str, Any]) -> str:
#     """
#     Thực thi Tool an toàn (Safe Tool Execution theo chuẩn Rubric):
#     Bắt toàn bộ Exception và trả về chuỗi 'Error: ...' để LLM biết lỗi và tự sửa sai.
#     """
#     try:
#         result = await client.call_tool(fn_name, fn_args)
#         if result and hasattr(result, "content") and result.content:
#             return result.content[0].text
#         return "Không có dữ liệu trả về từ công cụ."
#     except Exception as e:
#         # Tuyệt đối không raise exception làm crash agent
#         return f"Error executing tool '{fn_name}': {str(e)}"


# def convert_mcp_to_groq_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
#     """Chuyển đổi danh sách Tool của MCP sang JSON Schema của Groq."""
#     groq_tools = []
#     for tool in mcp_tools:
#         groq_tools.append({
#             "type": "function",
#             "function": {
#                 "name": tool.name,
#                 "description": tool.description or "",
#                 "parameters": tool.input_schema,
#             },
#         })
#     return groq_tools

def run_react_agent(
    client: Groq, question: str, max_steps=MAX_STEPS) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n[bold yellow]──────── Bước {step}/{max_steps} ────────[/bold yellow]")
    
        reply = client.chat.completions.create(
                    model=MODEL, messages=messages
                ).choices[0].message.content

        print(f"\n[bold cyan]🤖 LLM Reply:[/bold cyan]\n{reply}")

        kind, payload = parse_reply(reply)

        if kind == "final":
            return payload
        
        if kind != "action":
            return "(the model did not follow the ReAct format)"

        name, argument = payload
        observation = run_action(name, argument)
        print(f"\n[bold green]👁️ Observation:[/bold green] {observation[:150]}...")

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "(stopped: too many steps)"            
                



    