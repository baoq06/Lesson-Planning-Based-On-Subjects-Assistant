import os
import sys
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

# Đảm bảo đường dẫn import từ repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.loop import ask_agent
from src.agent.call_llm import call_llm
from groq import Groq

load_dotenv()

TESTSET_PATH = REPO_ROOT / "tests" / "testset.jsonl"
LOGS_DIR = REPO_ROOT / "tests" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

REFUSAL_EXACT = "Xin lỗi, thông tin này không có trong tài liệu bài giảng và quy chế môn học AIPR."

console = Console()

def evaluate_faithfulness_llm(groq_client: Groq, question: str, context: str, answer: str) -> float:
    """Đánh giá Faithfulness (Tính trung thực): câu trả lời có hoàn toàn dựa trên dữ liệu tra cứu không?"""
    if not context.strip():
        if REFUSAL_EXACT.lower() in answer.lower():
            return 1.0
        return 0.5

    prompt = f"""Bạn là Giám khảo đánh giá tính trung thực (Faithfulness) của hệ thống RAG/Agent.
Nhiệm vụ: Hãy kiểm tra xem câu trả lời (Answer) có hoàn toàn bám sát thông tin thực tế được cung cấp trong ngữ cảnh tra cứu (Context) hay không.

[Câu hỏi]: {question}

[Ngữ cảnh từ công cụ/tài liệu]:
{context[:2500]}

[Câu trả lời của Agent]:
{answer}

Quy tắc chấm:
- Điểm 1.0: Mọi thông tin và số liệu trong Answer đều được chứng minh trực tiếp từ Context, không tự bịa đặt.
- Điểm 0.5: Trả lời phần lớn đúng nhưng có suy diễn nhẹ hoặc thiếu sót nhỏ.
- Điểm 0.0: Bịa đặt, mâu thuẫn trực tiếp với Context hoặc tự ý trả lời kiến thức ngoài mà không có trong Context.

Chỉ trả về DUY NHẤT một số thập phân (1.0, 0.5, hoặc 0.0), không viết thêm bất kỳ chữ nào khác.
Điểm số:"""

    try:
        res = call_llm(
            client=groq_client,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=20.0
        )
        score_text = res.choices[0].message.content.strip()
        match = re.search(r"(1\.0|0\.5|0\.0|1|0)", score_text)
        if match:
            return float(match.group(1))
        return None
    except Exception as e:
        console.print(f"[dim red]Lỗi khi gọi LLM-as-judge faithfulness: {e}[/dim red]")
        return None

def evaluate_answer_relevance_llm(groq_client: Groq, question: str, answer: str) -> Optional[float]:
    """Đánh giá Answer Relevance (Độ phù hợp): câu trả lời có đi thẳng vào trọng tâm câu hỏi không?"""
    if not answer.strip():
        return 0.0

    # Nếu là câu từ chối chuẩn khi câu hỏi nằm ngoài phạm vi thì tính là phù hợp tuyệt đối
    if REFUSAL_EXACT.lower() in answer.lower():
        return 1.0

    prompt = f"""Bạn là Giám khảo đánh giá độ liên quan và phù hợp của câu trả lời (Answer Relevance) cho hệ thống AI.
Nhiệm vụ: Hãy kiểm tra xem câu trả lời (Answer) có giải quyết đúng, trúng và trực tiếp câu hỏi (Question) của người dùng hay không (không xét tính đúng sai của dữ kiện, chỉ xét mức độ liên quan và tập trung vào câu hỏi).

[Câu hỏi]: {question}

[Câu trả lời của Agent]:
{answer}

Quy tắc chấm:
- Điểm 1.0: Câu trả lời đi thẳng vào trọng tâm câu hỏi, giải quyết trực tiếp thắc mắc, không lan man.
- Điểm 0.5: Trả lời đúng một phần trọng tâm hoặc có chứa nhiều thông tin thừa thãi không cần thiết.
- Điểm 0.0: Hoàn toàn lạc đề, né tránh câu hỏi hoặc không liên quan gì đến câu hỏi.

Chỉ trả về DUY NHẤT một số thập phân (1.0, 0.5, hoặc 0.0), không viết thêm bất kỳ chữ nào khác.
Điểm số:"""

    try:
        res = call_llm(
            client=groq_client,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=20.0
        )
        score_text = res.choices[0].message.content.strip()
        match = re.search(r"(1\.0|0\.5|0\.0|1|0)", score_text)
        if match:
            return float(match.group(1))
        return None
    except Exception as e:
        console.print(f"[dim red]Lỗi khi gọi LLM-as-judge relevance: {e}[/dim red]")
        return None

def evaluate_context_precision_llm(groq_client: Groq, question: str, context: str, expected_answer: str = "") -> Optional[float]:
    """Đánh giá Context Precision (Độ chính xác ngữ cảnh): ngữ cảnh tra cứu từ tools có thực sự hữu ích và trúng câu hỏi không?"""
    if not context.strip():
        return 1.0

    prompt = f"""Bạn là Giám khảo đánh giá độ chính xác của ngữ cảnh truy xuất (Context Precision) trong hệ thống RAG/Agent.
Nhiệm vụ: Hãy kiểm tra xem ngữ cảnh thu thập được từ các công cụ (Context) có thực sự chứa các dữ kiện hữu ích và liên quan trực tiếp đến câu hỏi (Question) hay không.

[Câu hỏi]: {question}
{f'[Ý trả lời mong đợi]: {expected_answer}' if expected_answer else ''}

[Ngữ cảnh từ công cụ/tài liệu]:
{context[:2500]}

Quy tắc chấm:
- Điểm 1.0: Ngữ cảnh chứa thông tin chính xác, thiết yếu để giải quyết câu hỏi, ít hoặc không có thông tin rác.
- Điểm 0.5: Ngữ cảnh có chứa thông tin cần thiết nhưng bị lẫn nhiều dữ liệu nhiễu hoặc tài liệu chưa sát.
- Điểm 0.0: Toàn bộ ngữ cảnh hoàn toàn không liên quan đến câu hỏi.

Chỉ trả về DUY NHẤT một số thập phân (1.0, 0.5, hoặc 0.0), không viết thêm bất kỳ chữ nào khác.
Điểm số:"""

    try:
        res = call_llm(
            client=groq_client,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=20.0
        )
        score_text = res.choices[0].message.content.strip()
        match = re.search(r"(1\.0|0\.5|0\.0|1|0)", score_text)
        if match:
            return float(match.group(1))
        return None
    except Exception as e:
        console.print(f"[dim red]Lỗi khi gọi LLM-as-judge context precision: {e}[/dim red]")
        return None

def run_evaluation(testset_file: Path = TESTSET_PATH, limit: Optional[int] = None) -> dict[str, Any]:
    """Chạy toàn bộ pipeline kiểm thử và xuất báo cáo đo lường định lượng."""
    if not testset_file.exists():
        console.print(f"[bold red]Lỗi: Không tìm thấy file {testset_file}[/bold red]")
        sys.exit(1)

    with open(testset_file, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    if limit:
        cases = cases[:limit]

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Đếm số lần chạy trước đó để xác định run_index
    existing_runs = list(LOGS_DIR.glob("eval_run_*.json"))
    run_index = len(existing_runs) + 1

    console.print(Panel(
        f"[bold green]KHỞI ĐỘNG ĐÁNH GIÁ ĐỊNH LƯỢNG — RUN #{run_index}[/bold green]\n"
        f"File test: [cyan]{testset_file}[/cyan] | Số lượng câu hỏi: [yellow]{len(cases)}[/yellow]\n"
        f"Thời gian: [dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        border_style="green"
    ))

    results = []
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_latency = 0.0

    no_answer_total = 0
    no_answer_correct = 0

    action_total = 0
    action_verified = 0

    citation_matches = 0
    citation_expected_total = 0

    faithfulness_scores = []
    relevance_scores = []
    precision_scores = []

    for idx, case in enumerate(cases, 1):
        q_id = case.get("id", f"TC{idx:02d}")
        q_type = case.get("type", "factual")
        question = case["question"]
        expected_docs = case.get("expected_doc_ids", [])
        expected_tools = case.get("expected_tools", [])
        expected_ans = case.get("expected_answer", "")

        console.print(f"\n[bold cyan]── [{idx}/{len(cases)}] {q_id} ({q_type}) ──[/bold cyan]")
        console.print(f"[white]{question}[/white]")

        start_time = time.time()
        agent_res = ask_agent(question)
        elapsed = time.time() - start_time
        total_latency += elapsed

        answer = agent_res.answer or ""
        steps = agent_res.steps or []
        usage = agent_res.usage or {}

        # 1. Đo lường Token
        p_tok = usage.get("prompt_tokens", 0)
        c_tok = usage.get("completion_tokens", 0)
        tot_tok = usage.get("total_tokens", p_tok + c_tok)
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        total_tokens += tot_tok

        # Thu thập toàn bộ context/observations từ các bước tool
        context_parts = []
        tools_called = []
        for s in steps:
            if s.tool_call:
                tools_called.append(s.tool_call.name)
            if s.observation:
                context_parts.append(str(s.observation))
        full_context = "\n".join(context_parts)

        # 2. Đánh giá nhóm no_answer: Đo tỉ lệ từ chối đúng
        refusal_ok = None
        if q_type == "no_answer":
            no_answer_total += 1
            if REFUSAL_EXACT.lower() in answer.lower():
                refusal_ok = True
                no_answer_correct += 1
            else:
                refusal_ok = False

        # 3. Đánh giá nhóm action: Write-then-Read Verification
        action_ok = None
        if q_type == "action" or expected_tools:
            action_total += 1
            if "draft_template_submission" in tools_called and "read_draft" in tools_called:
                write_idx = tools_called.index("draft_template_submission")
                read_idx = tools_called.index("read_draft")
                action_ok = (read_idx > write_idx)
            else:
                action_ok = False
            
            if action_ok:
                action_verified += 1

        # 4. Đánh giá Citation (Trích nguồn doc_id)
        cited_in_answer = []
        for doc_id in ["POLICY_AIPR", "LP_W01", "LP_W02", "LP_W03", "LP_W04", "LP_W05", "LP_W06"]:
            if doc_id in answer:
                cited_in_answer.append(doc_id)

        citation_ok = True
        if expected_docs:
            citation_expected_total += len(expected_docs)
            for doc in expected_docs:
                if doc in answer:
                    citation_matches += 1
                else:
                    citation_ok = False

        # 5. Đánh giá bộ ba RAG Triad: Faithfulness, Answer Relevance, Context Precision
        faith_score = evaluate_faithfulness_llm(groq_client, question, full_context, answer)
        relevance_score = evaluate_answer_relevance_llm(groq_client, question, answer)
        precision_score = evaluate_context_precision_llm(groq_client, question, full_context, expected_ans)

        faithfulness_scores.append(faith_score)
        relevance_scores.append(relevance_score)
        precision_scores.append(precision_score)

        record = {
            "id": q_id,
            "type": q_type,
            "question": question,
            "expected_doc_ids": expected_docs,
            "expected_tools": expected_tools,
            "answer": answer,
            "steps_count": len(steps),
            "tools_called": tools_called,
            "cited_doc_ids": cited_in_answer,
            "refusal_correct": refusal_ok,
            "action_verified": action_ok,
            "citation_ok": citation_ok,
            "faithfulness": faith_score,
            "answer_relevance": relevance_score,
            "context_precision": precision_score,
            "latency_sec": round(elapsed, 2),
            "usage": usage,
        }
        results.append(record)

        if idx < len(cases):
            time.sleep(2.0)

    # Tổng kết số liệu định lượng
    refusal_rate = (no_answer_correct / no_answer_total * 100) if no_answer_total > 0 else 100.0
    action_rate = (action_verified / action_total * 100) if action_total > 0 else 100.0
    citation_rate = (citation_matches / citation_expected_total * 100) if citation_expected_total > 0 else 100.0
    valid_faith = [s for s in faithfulness_scores if s is not None]
    avg_faithfulness = (sum(valid_faith) / len(valid_faith)) if valid_faith else 0.0
    valid_rel = [s for s in relevance_scores if s is not None]
    avg_relevance = (sum(valid_rel) / len(valid_rel)) if valid_rel else 0.0
    valid_prec = [s for s in precision_scores if s is not None]
    avg_precision = (sum(valid_prec) / len(valid_prec)) if valid_prec else 0.0
    avg_latency = (total_latency / len(cases)) if cases else 0.0

    summary_data = {
        "run_id": f"run_{run_index}",
        "timestamp": run_timestamp,
        "total_testcases": len(cases),
        "metrics": {
            "average_faithfulness": round(avg_faithfulness, 4),
            "average_answer_relevance": round(avg_relevance, 4),
            "average_context_precision": round(avg_precision, 4),
            "no_answer_refusal_rate_pct": round(refusal_rate, 2),
            "no_answer_correct": no_answer_correct,
            "no_answer_total": no_answer_total,
            "action_verification_rate_pct": round(action_rate, 2),
            "action_verified": action_verified,
            "action_total": action_total,
            "citation_hit_rate_pct": round(citation_rate, 2),
            "citation_matches": citation_matches,
            "citation_expected": citation_expected_total,
            "average_latency_sec": round(avg_latency, 2),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
        },
        "details": results
    }

    # Lưu log ra file JSON
    out_file = LOGS_DIR / f"eval_run_{run_index}_{run_timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # Hiển thị bảng kết quả chi tiết
    console.print("\n")
    table = Table(title="BẢNG ĐÁNH GIÁ CHI TIẾT CÁC TEST CASES (RAG TRIAD)", header_style="bold magenta")
    table.add_column("ID", style="cyan", width=5)
    table.add_column("Type", style="yellow", width=9)
    table.add_column("Trích Doc ID", style="blue")
    table.add_column("Từ chối", justify="center", width=7)
    table.add_column("Faithful", justify="right", width=8)
    table.add_column("Relevance", justify="right", width=9)
    table.add_column("Precision", justify="right", width=9)
    table.add_column("Tokens", justify="right", width=7)
    table.add_column("Latency", justify="right", width=8)

    for r in results:
        # Refusal icon
        if r["refusal_correct"] is True:
            ref_str = "[green]✓ Đúng[/green]"
        elif r["refusal_correct"] is False:
            ref_str = "[red]✗ Sai[/red]"
        else:
            ref_str = "[dim]—[/dim]"

        cited_str = ", ".join(r["cited_doc_ids"]) if r["cited_doc_ids"] else "[dim]none[/dim]"
        tok_val = r["usage"].get("total_tokens", 0)

        faith_str = f"{r['faithfulness']:.2f}" if r["faithfulness"] is not None else "[dim]N/A[/dim]"
        rel_str = f"{r['answer_relevance']:.2f}" if r["answer_relevance"] is not None else "[dim]N/A[/dim]"
        prec_str = f"{r['context_precision']:.2f}" if r["context_precision"] is not None else "[dim]N/A[/dim]"

        table.add_row(
            r["id"],
            r["type"],
            cited_str,
            ref_str,
            faith_str,
            rel_str,
            prec_str,
            str(tok_val),
            f"{r['latency_sec']}s",
        )

    console.print(table)

    # Hiển thị bảng tổng kết chỉ số định lượng
    sum_table = Table(title=f"TỔNG KẾT CHỈ SỐ ĐỊNH LƯỢNG — EVALUATION RUN #{run_index}", header_style="bold cyan")
    sum_table.add_column("Chỉ số đo lường", style="bold white")
    sum_table.add_column("Kết quả đạt được", justify="right", style="bold green")
    sum_table.add_column("Mục tiêu Rubric", justify="right", style="dim")

    sum_table.add_row("1. Tính trung thực (Faithfulness Score)", f"{avg_faithfulness:.2f} / 1.0", "≥ 0.85")
    sum_table.add_row("2. Độ phù hợp câu trả lời (Answer Relevance)", f"{avg_relevance:.2f} / 1.0", "≥ 0.85")
    sum_table.add_row("3. Độ chính xác ngữ cảnh (Context Precision)", f"{avg_precision:.2f} / 1.0", "≥ 0.80")
    sum_table.add_row("4. Tỉ lệ từ chối đúng (Refusal Rate)", f"{refusal_rate:.1f}% ({no_answer_correct}/{no_answer_total})", "100%")
    sum_table.add_row("5. Tỉ lệ Verify Hành động (Action Rate)", f"{action_rate:.1f}% ({action_verified}/{action_total})", "100%")
    sum_table.add_row("6. Tỉ lệ trích nguồn đúng (Citation Rate)", f"{citation_rate:.1f}% ({citation_matches}/{citation_expected_total})", "≥ 80%")
    sum_table.add_row("7. Thời gian xử lý trung bình", f"{avg_latency:.2f} giây / câu", "< 15s")
    sum_table.add_row("8. Tổng Token tiêu thụ", f"{total_tokens:,} tokens", "Chi phí hợp lý")
    sum_table.add_row("Log đánh giá đã lưu tại", str(out_file), "Cần ≥3 runs")

    console.print(sum_table)
    console.print(f"\n[bold green]✓ Hoàn thành đánh giá Run #{run_index}. File lưu:[/bold green] [cyan]{out_file}[/cyan]\n")

    return summary_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AIPR Course Assistant - Evaluation Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số câu test để chạy thử nhanh")
    parser.add_argument("--testset", type=str, default=str(TESTSET_PATH), help="Đường dẫn file testset.jsonl")
    args = parser.parse_args()

    run_evaluation(testset_file=Path(args.testset), limit=args.limit)
