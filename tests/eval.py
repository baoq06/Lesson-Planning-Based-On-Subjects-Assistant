import os
import sys
import json
import time
import re
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

def evaluate_faithfulness_llm(groq_client: Groq, question: str, context: str, answer: str) -> Optional[float]:
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


def save_eval_artifacts(
    out_json: Path,
    out_csv: Path,
    run_id: str,
    run_timestamp: str,
    total_target: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Cập nhật tức thì kết quả đánh giá vào file JSON và CSV sau mỗi test case."""
    no_answer_total = sum(1 for r in results if r.get("type") == "no_answer")
    no_answer_correct = sum(1 for r in results if r.get("type") == "no_answer" and r.get("refusal_correct") is True)

    action_total = sum(1 for r in results if r.get("type") == "action" or r.get("expected_tools"))
    action_verified = sum(1 for r in results if (r.get("type") == "action" or r.get("expected_tools")) and r.get("action_verified") is True)

    citation_expected_total = sum(len(r.get("expected_doc_ids", [])) for r in results if r.get("expected_doc_ids"))
    citation_matches = sum(
        sum(1 for doc in r.get("expected_doc_ids", []) if doc in r.get("cited_doc_ids", []))
        for r in results if r.get("expected_doc_ids")
    )

    faith_scores = [r["faithfulness"] for r in results if r.get("faithfulness") is not None]
    avg_faith = (sum(faith_scores) / len(faith_scores)) if faith_scores else 0.0

    rel_scores = [r["answer_relevance"] for r in results if r.get("answer_relevance") is not None]
    avg_rel = (sum(rel_scores) / len(rel_scores)) if rel_scores else 0.0

    prec_scores = [r["context_precision"] for r in results if r.get("context_precision") is not None]
    avg_prec = (sum(prec_scores) / len(prec_scores)) if prec_scores else 0.0

    refusal_rate = (no_answer_correct / no_answer_total * 100) if no_answer_total > 0 else 100.0
    action_rate = (action_verified / action_total * 100) if action_total > 0 else 100.0
    citation_rate = (citation_matches / citation_expected_total * 100) if citation_expected_total > 0 else 100.0

    total_latency = sum(r.get("latency_sec", 0.0) for r in results)
    avg_latency = (total_latency / len(results)) if results else 0.0

    p_tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results)
    c_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in results)
    tot_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in results)

    summary_data = {
        "run_id": run_id,
        "timestamp": run_timestamp,
        "total_testcases": total_target,
        "completed_testcases": len(results),
        "is_completed": len(results) >= total_target,
        "metrics": {
            "average_faithfulness": round(avg_faith, 4),
            "average_answer_relevance": round(avg_rel, 4),
            "average_context_precision": round(avg_prec, 4),
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
            "total_prompt_tokens": p_tokens,
            "total_completion_tokens": c_tokens,
            "total_tokens": tot_tokens,
        },
        "details": results
    }

    # 1. Ghi file JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # 2. Ghi file CSV
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID", "Type", "Question", "Tools_Called", "Refusal_Correct",
            "Faithfulness", "Answer_Relevance", "Context_Precision",
            "Latency_Sec", "Total_Tokens", "Cited_Doc_IDs"
        ])
        for r in results:
            writer.writerow([
                r["id"],
                r["type"],
                r["question"].replace("\n", " "),
                "; ".join(r.get("tools_called", [])),
                r["refusal_correct"] if r.get("refusal_correct") is not None else "N/A",
                r["faithfulness"] if r.get("faithfulness") is not None else "N/A",
                r["answer_relevance"] if r.get("answer_relevance") is not None else "N/A",
                r["context_precision"] if r.get("context_precision") is not None else "N/A",
                r.get("latency_sec", 0.0),
                r.get("usage", {}).get("total_tokens", 0),
                "; ".join(r.get("cited_doc_ids", [])),
            ])

    return summary_data


def generate_eval_charts(out_chart: Path, summary_data: dict[str, Any]) -> None:
    """Vẽ đồ thị trực quan hóa kết quả đánh giá (RAG Triad + Latency)."""
    details = summary_data.get("details", [])
    if not details:
        return

    try:
        case_ids = [r["id"] for r in details]
        faiths = [r["faithfulness"] if r.get("faithfulness") is not None else 0.0 for r in details]
        rels = [r["answer_relevance"] if r.get("answer_relevance") is not None else 0.0 for r in details]
        precs = [r["context_precision"] if r.get("context_precision") is not None else 0.0 for r in details]
        latencies = [r.get("latency_sec", 0.0) for r in details]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"AIPR Evaluation Benchmark Report ({summary_data.get('run_id', 'run')})", fontsize=14, fontweight="bold")

        x = list(range(len(case_ids)))
        width = 0.25

        # Subplot 1: RAG Triad scores
        ax1.bar([p - width for p in x], faiths, width=width, label="Faithfulness", color="#2ecc71")
        ax1.bar(x, rels, width=width, label="Answer Relevance", color="#3498db")
        ax1.bar([p + width for p in x], precs, width=width, label="Context Precision", color="#9b59b6")
        ax1.axhline(0.85, color="#e74c3c", linestyle="--", linewidth=1.2, label="Rubric Target (≥0.85 / ≥0.80)")
        ax1.set_ylabel("Score (0.0 - 1.0)", fontweight="bold")
        ax1.set_ylim(0, 1.15)
        ax1.set_title("RAG Triad Metrics per Test Case", fontsize=11)
        ax1.legend(loc="upper right")
        ax1.grid(axis="y", linestyle=":", alpha=0.6)

        # Subplot 2: Latency
        ax2.plot(case_ids, latencies, marker="o", color="#e67e22", linewidth=2, label="Latency (seconds)")
        ax2.axhline(15.0, color="#c0392b", linestyle=":", linewidth=1, label="Target (<15s)")
        ax2.set_ylabel("Latency (sec)", fontweight="bold")
        ax2.set_xlabel("Test Case ID", fontweight="bold")
        ax2.set_title("Response Latency per Test Case", fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels(case_ids, rotation=45)
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(out_chart, dpi=200)
        plt.close()
        console.print(f"[bold green]✓ Đã vẽ biểu đồ đánh giá:[/bold green] [cyan]{out_chart.name}[/cyan]")
    except Exception as e:
        console.print(f"[dim red]Không thể vẽ đồ thị: {e}[/dim red]")


def run_evaluation(
    testset_file: Path = TESTSET_PATH,
    limit: Optional[int] = None,
    resume: bool = False,
    force_new: bool = False,
) -> dict[str, Any]:
    """Chạy toàn bộ pipeline kiểm thử và xuất báo cáo đo lường định lượng."""
    if not testset_file.exists():
        console.print(f"[bold red]Lỗi: Không tìm thấy file {testset_file}[/bold red]")
        sys.exit(1)

    with open(testset_file, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    if limit:
        cases = cases[:limit]

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Tự động phát hiện run chưa hoàn thành gần nhất để resume (rất tiện khi chạy make eval)
    resumed_file = None
    if not force_new:
        existing_json_files = sorted(LOGS_DIR.glob("eval_run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for jf in existing_json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if not data.get("is_completed", False):
                    resumed_file = jf
                    break
            except Exception:
                continue

    if resumed_file:
        data = json.loads(resumed_file.read_text(encoding="utf-8"))
        run_id = data.get("run_id", "run_1")
        run_timestamp = data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        out_file = resumed_file
        out_csv = out_file.with_suffix(".csv")
        results = data.get("details", [])
        completed_ids = {r["id"] for r in results}
        idx_match = re.search(r"eval_run_(\d+)", out_file.name)
        run_index = int(idx_match.group(1)) if idx_match else 1

        console.print(Panel(
            f"[bold yellow]TIẾP TỤC ĐÁNH GIÁ (RESUME) — {run_id.upper()}[/bold yellow]\n"
            f"File log đang cập nhật: [cyan]{out_file.name}[/cyan]\n"
            f"Đã hoàn thành trước đó: [green]{len(completed_ids)}/{len(cases)} câu[/green]\n"
            f"Sẽ chạy tiếp các câu còn lại...",
            border_style="yellow"
        ))
    else:
        existing_runs = list(LOGS_DIR.glob("eval_run_*.json"))
        run_index = len(existing_runs) + 1
        run_id = f"run_{run_index}"
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = LOGS_DIR / f"eval_run_{run_index}_{run_timestamp}.json"
        out_csv = LOGS_DIR / f"eval_run_{run_index}_{run_timestamp}.csv"
        results = []
        completed_ids = set()

        console.print(Panel(
            f"[bold green]KHỞI ĐỘNG ĐÁNH GIÁ ĐỊNH LƯỢNG — {run_id.upper()}[/bold green]\n"
            f"File test: [cyan]{testset_file.name}[/cyan] | Số lượng câu hỏi: [yellow]{len(cases)}[/yellow]\n"
            f"File lưu kết quả: [cyan]{out_file.name}[/cyan] & [cyan]{out_csv.name}[/cyan]\n"
            f"Thời gian: [dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="green"
        ))

    summary_data = {}

    for idx, case in enumerate(cases, 1):
        q_id = case.get("id", f"TC{idx:02d}")
        q_type = case.get("type", "factual")
        question = case["question"]
        expected_docs = case.get("expected_doc_ids", [])
        expected_tools = case.get("expected_tools", [])
        expected_ans = case.get("expected_answer", "")

        # Nếu câu này đã chạy xong ở lần trước thì bỏ qua
        if q_id in completed_ids:
            console.print(f"[dim yellow]── [{idx}/{len(cases)}] {q_id} ({q_type}): Đã chạy xong trước đó, bỏ qua.[/dim yellow]")
            continue

        console.print(f"\n[bold cyan]── [{idx}/{len(cases)}] {q_id} ({q_type}) ──[/bold cyan]")
        console.print(f"[white]{question}[/white]")

        start_time = time.time()
        agent_res = ask_agent(question)
        elapsed = time.time() - start_time

        answer = agent_res.answer or ""
        steps = agent_res.steps or []
        usage = agent_res.usage or {}

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
            if REFUSAL_EXACT.lower() in answer.lower():
                refusal_ok = True
            else:
                refusal_ok = False

        # 3. Đánh giá nhóm action: Write-then-Read Verification
        action_ok = None
        if q_type == "action" or expected_tools:
            if "draft_template_submission" in tools_called and "read_draft" in tools_called:
                write_idx = tools_called.index("draft_template_submission")
                read_idx = tools_called.index("read_draft")
                action_ok = (read_idx > write_idx)
            else:
                action_ok = False

        # 4. Đánh giá Citation (Trích nguồn doc_id)
        cited_in_answer = []
        for doc_id in ["POLICY_AIPR", "LP_W01", "LP_W02", "LP_W03", "LP_W04", "LP_W05", "LP_W06"]:
            if doc_id in answer:
                cited_in_answer.append(doc_id)

        citation_ok = True
        if expected_docs:
            for doc in expected_docs:
                if doc not in cited_in_answer:
                    citation_ok = False
                    break

        # 5. Đánh giá LLM-as-judge (RAG Triad)
        faith_score = None
        relevance_score = None
        precision_score = None

        if q_type != "no_answer":
            faith_score = evaluate_faithfulness_llm(groq_client, question, full_context, answer)
            relevance_score = evaluate_answer_relevance_llm(groq_client, question, answer)
            precision_score = evaluate_context_precision_llm(groq_client, question, full_context, expected_ans)
        else:
            faith_score = 1.0 if refusal_ok else 0.0
            relevance_score = 1.0 if refusal_ok else 0.0
            precision_score = 1.0

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
        completed_ids.add(q_id)

        # CẬP NHẬT TỨC THÌ VÀO CẢ FILE JSON VÀ CSV SAU MỖI CÂU
        summary_data = save_eval_artifacts(out_file, out_csv, run_id, run_timestamp, len(cases), results)
        console.print(f"[dim green]✓ Đã cập nhật kết quả {q_id} vào {out_file.name} và {out_csv.name}[/dim green]")

        if idx < len(cases):
            time.sleep(2.0)

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
        if r.get("refusal_correct") is True:
            ref_str = "[green]✓ Đúng[/green]"
        elif r.get("refusal_correct") is False:
            ref_str = "[red]✗ Sai[/red]"
        else:
            ref_str = "[dim]—[/dim]"

        cited_str = ", ".join(r.get("cited_doc_ids", [])) if r.get("cited_doc_ids") else "[dim]none[/dim]"
        tok_val = r.get("usage", {}).get("total_tokens", 0)

        faith_str = f"{r['faithfulness']:.2f}" if r.get("faithfulness") is not None else "[dim]N/A[/dim]"
        rel_str = f"{r['answer_relevance']:.2f}" if r.get("answer_relevance") is not None else "[dim]N/A[/dim]"
        prec_str = f"{r['context_precision']:.2f}" if r.get("context_precision") is not None else "[dim]N/A[/dim]"

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
    metrics = summary_data.get("metrics", {})
    sum_table = Table(title=f"TỔNG KẾT CHỈ SỐ ĐỊNH LƯỢNG — {run_id.upper()}", header_style="bold cyan")
    sum_table.add_column("Chỉ số đo lường", style="bold white")
    sum_table.add_column("Kết quả đạt được", justify="right", style="bold green")
    sum_table.add_column("Mục tiêu Rubric", justify="right", style="dim")

    sum_table.add_row("1. Tính trung thực (Faithfulness Score)", f"{metrics.get('average_faithfulness', 0):.2f} / 1.0", "≥ 0.85")
    sum_table.add_row("2. Độ phù hợp câu trả lời (Answer Relevance)", f"{metrics.get('average_answer_relevance', 0):.2f} / 1.0", "≥ 0.85")
    sum_table.add_row("3. Độ chính xác ngữ cảnh (Context Precision)", f"{metrics.get('average_context_precision', 0):.2f} / 1.0", "≥ 0.80")
    sum_table.add_row("4. Tỉ lệ từ chối đúng (Refusal Rate)", f"{metrics.get('no_answer_refusal_rate_pct', 0):.1f}% ({metrics.get('no_answer_correct', 0)}/{metrics.get('no_answer_total', 0)})", "100%")
    sum_table.add_row("5. Tỉ lệ Verify Hành động (Action Rate)", f"{metrics.get('action_verification_rate_pct', 0):.1f}% ({metrics.get('action_verified', 0)}/{metrics.get('action_total', 0)})", "100%")
    sum_table.add_row("6. Tỉ lệ trích nguồn đúng (Citation Rate)", f"{metrics.get('citation_hit_rate_pct', 0):.1f}% ({metrics.get('citation_matches', 0)}/{metrics.get('citation_expected', 0)})", "≥ 80%")
    sum_table.add_row("7. Thời gian xử lý trung bình", f"{metrics.get('average_latency_sec', 0):.2f} giây / câu", "< 15s")
    sum_table.add_row("8. Tổng Token tiêu thụ", f"{metrics.get('total_tokens', 0):,} tokens", "Chi phí hợp lý")
    sum_table.add_row("Log đánh giá JSON & CSV", f"{out_file.name} | {out_csv.name}", "Cần ≥3 runs")

    console.print(sum_table)

    # Tự động vẽ đồ thị
    out_chart = out_file.with_name(out_file.stem + "_chart.png")
    generate_eval_charts(out_chart, summary_data)

    console.print(f"\n[bold green]✓ Hoàn thành đánh giá {run_id}.[/bold green]")
    console.print(f"- JSON: [cyan]{out_file}[/cyan]")
    console.print(f"- CSV:  [cyan]{out_csv}[/cyan]")
    console.print(f"- PNG:  [cyan]{out_chart}[/cyan]\n")

    return summary_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AIPR Course Assistant - Evaluation Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số câu test để chạy thử nhanh")
    parser.add_argument("--testset", type=str, default=str(TESTSET_PATH), help="Đường dẫn file testset.jsonl")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục chạy tiếp từ câu chưa xong của lần trước")
    parser.add_argument("--force-new", action="store_true", help="Bắt buộc tạo run mới kể cả khi run cũ chưa hoàn thành")
    args = parser.parse_args()

    run_evaluation(testset_file=Path(args.testset), limit=args.limit, resume=args.resume, force_new=args.force_new)
