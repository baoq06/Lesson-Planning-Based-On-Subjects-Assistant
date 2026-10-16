import os
import sys
import json
import time
import asyncio
from datetime import datetime
from typing import Any, Optional
from dotenv import load_dotenv
from rich import print as rprint
from groq import Groq
from pathlib import Path
from src.schemas import ToolCall, AgentStep, AgentResponse
from src.agent.call_llm import call_llm

from mcp import Client, StdioServerParameters


MODEL = "qwen/qwen3.8-27b"
MAX_STEPS = 7

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "data" / "prompts" / "system_prompt.txt"

LOGS_DIR = REPO_ROOT / "logs"
SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# Tạo thư mục riêng cho từng session bên trong logs/
CURRENT_SESSION_DIR = LOGS_DIR / f"session_{SESSION_TIMESTAMP}"
CURRENT_SESSION_DIR.mkdir(parents=True, exist_ok=True)
# File JSONL sẽ được lưu bên trong thư mục của session này
RUN_LOG_PATH = CURRENT_SESSION_DIR / f"run_{SESSION_TIMESTAMP}.jsonl"

SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()


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

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        if "$defs" in schema:
            parameters["$defs"] = schema["$defs"]

        formatted_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        })
        
    return formatted_tools

def parse_tool_args(raw_args: Any) -> Optional[dict[str, Any]]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        stripped = raw_args.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
            return None
        except Exception:
            return None
    if raw_args is None:
        return {}
    return None

MAX_HISTORY_TURNS = 4  

def trim_history(history: list[dict[str, Any]], max_turns: int = MAX_HISTORY_TURNS) -> list[dict[str, Any]]:
    if not history:
        return []

    max_messages = max_turns * 2
    return history[-max_messages:]

async def safe_execute_mcp_tool(client: Client, tool_name: str, arguments: dict[str, Any]) -> str:
    try:
        result = await client.call_tool(tool_name, arguments)
        
        if result and hasattr(result, "content") and result.content:
            texts = [item.text for item in result.content if hasattr(item, "text")]
            return "\n".join(texts) if texts else "Error: No output returned by tool."
        return "Error: Tool executed but returned no data."
        
    except Exception as e:
        return f"Error: Failed to execute tool '{tool_name}': {type(e).__name__} - {e}"

def log_question_run(
    question: str,
    steps: list[AgentStep],
    answer: str,
    usage: dict[str, Any],
    rag_triad: Optional[dict[str, float]] = None,
    latency_sec: Optional[float] = None,
    cited_docs: Optional[list[str]] = None,
) -> None:
    record = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "steps": [
            {
                "step_num": s.step_num,
                "thought": s.thought,
                "tool": s.tool_call.name if s.tool_call else None,
                "args": s.tool_call.arguments if s.tool_call else {},
                "observation": s.observation,
            }
            for s in steps
        ],
        "answer": answer,
        "usage": usage,
        "rag_triad": rag_triad or {},
        "latency_sec": latency_sec,
        "cited_doc_ids": cited_docs or [],
    }
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

async def run_agent_loop(
    question: str, 
    max_steps: int = MAX_STEPS, 
    history: Optional[list[dict[str, Any]]] = None,
    auto_log: bool = True
) -> AgentResponse:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=0)

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp.mcp_server"],
        env={**os.environ, "SESSION_LOG_DIR": str(CURRENT_SESSION_DIR)}
    )

    async with Client(server) as client:
        mcp_tools = (await client.list_tools()).tools
        tools = get_mcp_tools_schema(mcp_tools)
        trimmed_memory = trim_history(history or [])
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *trimmed_memory,
            {"role": "user", "content": question}
        ]

        steps: list[AgentStep] = []
        usage_stats = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "max_prompt": 0,
        }
        rprint(f"\n[bold cyan]Bắt đầu xử lý câu hỏi:[/bold cyan] {question}")

        for step_num in range(1, max_steps + 1):
            rprint(f"\n[bold yellow]── Bước {step_num}/{max_steps} ──[/bold yellow]")

            response = call_llm(
                client=groq_client,
                messages=messages,
                tools=tools,
                model=MODEL,
                temperature=0.1
            )

            usage_stats["calls"] += 1
            if response.usage:
                p_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                c_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                usage_stats["prompt_tokens"] += p_tokens
                usage_stats["completion_tokens"] += c_tokens
                usage_stats["total_tokens"] += (p_tokens + c_tokens)
                usage_stats["max_prompt"] = max(usage_stats["max_prompt"], p_tokens)
            
            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                final_answer = msg.content or "Không có câu trả lời."
                rprint(f"[bold green]Final Answer:[/bold green]\n{final_answer}")
                if auto_log:
                    log_question_run(question, steps, final_answer, usage_stats)
                return AgentResponse(answer=final_answer, steps=steps, usage=usage_stats)
            
            thought = msg.content or f"Cần kích hoạt {len(msg.tool_calls)} công cụ."
            rprint(f"[bold magenta]Thought:[/bold magenta] {thought}")

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = parse_tool_args(tool_call.function.arguments)

                if fn_args is None:
                    observation = "Error: invalid JSON arguments"
                    rprint(f"[bold red]Observation:[/bold red] {observation}")
                    step_record = AgentStep(
                        step_num=step_num,
                        thought=str(thought),
                        tool_call=ToolCall(id=tool_call.id, name=fn_name, arguments={}),
                        observation=observation
                    )
                    steps.append(step_record)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": observation
                    })
                    continue

                args_for_mcp = dict(fn_args)
                extracted_thought = args_for_mcp.pop("thought", None)
                if extracted_thought:
                    thought = extracted_thought
                    rprint(f"[bold magenta]Extracted Thought:[/bold magenta] {thought}")

                rprint(f"[bold blue]Action:[/bold blue] {fn_name}({args_for_mcp})")
                observation = await safe_execute_mcp_tool(client, fn_name, args_for_mcp)

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
        if auto_log:
            log_question_run(question, steps, stop_warning, usage_stats)
        return AgentResponse(answer=stop_warning, steps=steps, usage=usage_stats)

def ask_agent(
    question: str, 
    max_steps: int = MAX_STEPS,
    history: Optional[list[dict[str, Any]]] = None,
    auto_log: bool = True
) -> AgentResponse:
    return asyncio.run(run_agent_loop(question=question, max_steps=max_steps, history=history, auto_log=auto_log))

if __name__ == "__main__":
    from src.rag.eval_metrics import evaluate_rag_triad, export_evaluation_csv, generate_evaluation_chart
    from rich.panel import Panel

    rprint("[bold green]====================================================[/bold green]")
    rprint("[bold green]   TRỢ LÝ BÀI GIẢNG & QUY CHẾ AIPR (INTERACTIVE CLI)[/bold green]")
    rprint("[bold green]====================================================[/bold green]")
    rprint("[dim]Gõ câu hỏi và nhấn Enter. Gõ 'exit' để thoát. Gõ 'clear' để xóa nhớ.[/dim]\n")

    conversation_history: list[dict[str, Any]] = []
    session_records: list[dict[str, Any]] = []
    groq_eval_client: Optional[Groq] = None

    while True:
        try:
            user_question = input("\nBạn: ").strip()
            if not user_question:
                continue
            if user_question.lower() in ["exit", "quit", "q"]:
                rprint("[yellow]Đang kết thúc phiên và xuất báo cáo đánh giá...[/yellow]")
                break
            if user_question.lower() == "clear":
                conversation_history.clear()
                rprint("[yellow]Đã làm mới bộ nhớ hội thoại.[/yellow]")
                continue

            t0 = time.time()
            response = ask_agent(user_question, history=conversation_history, auto_log=False)
            elapsed = time.time() - t0

            if response.answer and not response.answer.startswith("Dừng:"):
                conversation_history.append({"role": "user", "content": user_question})
                conversation_history.append({"role": "assistant", "content": response.answer})

            # Tính toán 3 chỉ số RAG Triad liên tục cho câu hỏi
            context_text = "\n".join([str(s.observation) for s in response.steps if s.observation])
            cited_docs = [
                d for d in ["POLICY_AIPR", "LP_W01", "LP_W02", "LP_W03", "LP_W04", "LP_W05", "LP_W06"]
                if d in (response.answer or "")
            ]

            if not groq_eval_client:
                groq_eval_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            triad = evaluate_rag_triad(
                groq_client=groq_eval_client,
                question=user_question,
                context=context_text,
                answer=response.answer or "",
            )

            # Xuất trực tiếp 3 chỉ số RAG Triad vào file log run_*.jsonl (không in ra terminal)
            log_question_run(
                question=user_question,
                steps=response.steps,
                answer=response.answer or "",
                usage=response.usage if isinstance(response.usage, dict) else {},
                rag_triad=triad,
                latency_sec=round(elapsed, 2),
                cited_docs=cited_docs,
            )

            session_records.append({
                "id": f"Demo_Q{len(session_records)+1:02d}",
                "type": "demo",
                "question": user_question,
                "answer": response.answer,
                "faithfulness": triad["faithfulness"],
                "answer_relevance": triad["answer_relevance"],
                "context_precision": triad["context_precision"],
                "latency_sec": round(elapsed, 2),
                "usage": response.usage if isinstance(response.usage, dict) else {},
                "cited_doc_ids": cited_docs,
                "tools_called": [s.tool_call.name for s in response.steps if s.tool_call],
            })

        except (KeyboardInterrupt, EOFError):
            rprint("\n[yellow]Đã nhận lệnh thoát phiên. Đang xuất báo cáo...[/yellow]")
            break

    # Khi kết thúc 1 phiên demo: Xuất CSV và Đồ thị PNG tổng hợp toàn bộ các câu trong phiên
    if session_records:
        demo_csv = CURRENT_SESSION_DIR / f"demo_run_{SESSION_TIMESTAMP}.csv"
        demo_chart = CURRENT_SESSION_DIR / f"demo_run_{SESSION_TIMESTAMP}_chart.png"
        rprint("\n[bold cyan]── TỔNG KẾT PHIÊN LÀM VIỆC (DEMO SESSION) ──[/bold cyan]")
        export_evaluation_csv(demo_csv, session_records)
        generate_evaluation_chart(demo_chart, f"AIPR Interactive Demo Session Benchmark ({SESSION_TIMESTAMP})", session_records)
        rprint(f"[bold green]✓ Đã hoàn thành lưu phiên demo ({len(session_records)} câu hỏi):[/bold green]")
        rprint(f"  • Thư mục Session: [cyan]{CURRENT_SESSION_DIR}[/cyan]")
        rprint(f"  • File JSONL:      [cyan]{RUN_LOG_PATH}[/cyan]")
        rprint(f"  • File CSV:        [cyan]{demo_csv}[/cyan]")
        rprint(f"  • Đồ thị PNG:      [cyan]{demo_chart}[/cyan]\n")
    else:
        rprint("[yellow]Chưa có câu hỏi nào được thực hiện trong phiên này.[/yellow]")
