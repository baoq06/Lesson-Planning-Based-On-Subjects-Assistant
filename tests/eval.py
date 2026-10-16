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

from src.rag.eval_metrics import evaluate_rag_triad, export_evaluation_csv, generate_evaluation_chart

console = Console()

def save_eval_checkpoint_json(
    out_json: Path,
    run_id: str,
    run_timestamp: str,
    total_target: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Cập nhật tức thì checkpoint JSON sau mỗi test case để phục vụ resume nếu bị gián đoạn."""
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

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    return summary_data



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

        # 5. Đánh giá RAG Triad với thang điểm liên tục (LLM-as-a-judge)
        triad = evaluate_rag_triad(
            groq_client=groq_client,
            question=question,
            context=full_context,
            answer=answer,
            expected_answer=expected_ans,
            is_no_answer=(q_type == "no_answer")
        )
        faith_score = triad["faithfulness"]
        relevance_score = triad["answer_relevance"]
        precision_score = triad["context_precision"]

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

        # LƯU CHECKPOINT TỨC THÌ VÀO FILE JSON ĐỂ RESUME NẾU BỊ NGẮT
        summary_data = save_eval_checkpoint_json(out_file, run_id, run_timestamp, len(cases), results)
        console.print(f"[dim green]✓ Đã checkpoint kết quả {q_id} vào {out_file.name} (F: {faith_score:.2f}, R: {relevance_score:.2f}, P: {precision_score:.2f})[/dim green]")

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

    # KHI XONG HẾT TESTSET: Xuất file CSV và vẽ biểu đồ PNG tổng hợp của toàn bộ testset
    out_chart = out_file.with_name(out_file.stem + "_chart.png")
    export_evaluation_csv(out_csv, results)
    generate_evaluation_chart(out_chart, f"AIPR Testset Evaluation Benchmark ({run_id.upper()})", results)

    console.print(f"\n[bold green]✓ Hoàn thành đánh giá toàn bộ testset ({len(results)} câu) — {run_id}.[/bold green]")
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
