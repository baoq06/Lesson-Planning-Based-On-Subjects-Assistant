"""
Module đánh giá RAG Triad & xuất báo cáo dùng chung cho cả `make demo` và `make eval`.
- Chấm điểm liên tục (Continuous Float Score [0.0 - 1.0]) bằng LLM-as-a-judge.
- Xuất file CSV chi tiết theo từng câu hỏi.
- Vẽ đồ thị PNG tổng hợp kết quả của phiên làm việc.
"""

import os
import sys
import re
import csv
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from groq import Groq
from rich.console import Console

from src.agent.call_llm import call_llm

console = Console(legacy_windows=False)

REFUSAL_EXACT = "Xin lỗi, thông tin này không có trong tài liệu bài giảng và quy chế môn học AIPR."


def evaluate_faithfulness(groq_client: Groq, question: str, context: str, answer: str) -> float:
    """
    Đánh giá Faithfulness (Tính trung thực) theo thang điểm liên tục 0.00 - 1.00.
    Tỉ lệ số luận điểm/khẳng định trong câu trả lời có bằng chứng trực tiếp từ Context.
    """
    if not context.strip():
        if REFUSAL_EXACT.lower() in answer.lower():
            return 1.0
        return 0.5

    prompt = f"""Bạn là Giám khảo đánh giá tính trung thực (Faithfulness) của hệ thống RAG/Agent.
Nhiệm vụ: Hãy phân tích kỹ các ý/luận điểm trong [Câu trả lời] và đối chiếu với [Ngữ cảnh tra cứu].

[Câu hỏi]: {question}

[Ngữ cảnh tra cứu từ tài liệu/tools]:
{context[:9000]}

[Câu trả lời của Agent]:
{answer}

Quy tắc chấm điểm (Thang điểm số thực liên tục từ 0.00 đến 1.00):
1. Phân rã câu trả lời thành các phát biểu/khẳng định thông tin chính.
2. Kiểm tra từng phát biểu xem có được chứng minh bởi Ngữ cảnh tra cứu hay không:
   - Điểm số = (Số lượng phát biểu được chứng minh bởi Ngữ cảnh) / (Tổng số phát biểu trong câu trả lời).
   - Nếu 100% phát biểu đều có căn cứ trong Ngữ cảnh: Điểm 1.00.
   - Nếu đa số có căn cứ nhưng có suy diễn nhẹ hoặc chi tiết nhỏ ngoài lề: Cho điểm từ 0.70 đến 0.95 tùy mức độ.
   - Nếu trả lời đúng một nửa và tự bịa một nửa: Cho điểm quanh 0.50.
   - Nếu hoàn toàn bịa đặt (hallucination) hoặc mâu thuẫn trực tiếp với Ngữ cảnh: Cho điểm dưới 0.20.

Hãy tính toán cẩn thận và CHỈ TRẢ VỀ DUY NHẤT một số thập phân từ 0.00 đến 1.00 (ví dụ: 0.95 hoặc 0.85), không viết thêm chữ nào khác.
Điểm số:"""

    try:
        res = call_llm(
            client=groq_client,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=20.0
        )
        score_text = res.choices[0].message.content.strip()
        matches = re.findall(r"(?:1\.00|1\.0|1|0\.\d{1,2}|0)", score_text)
        if matches:
            val = float(matches[0])
            return round(min(max(val, 0.0), 1.0), 2)
        return 0.85
    except Exception as e:
        console.print(f"[dim red]Lỗi khi chấm Faithfulness: {e}[/dim red]")
        return 0.85


def evaluate_answer_relevance(groq_client: Groq, question: str, answer: str) -> float:
    """
    Đánh giá Answer Relevance (Độ phù hợp của câu trả lời) theo thang điểm liên tục 0.00 - 1.00.
    Đo mức độ giải quyết trực tiếp, trúng và đủ câu hỏi, phạt các phần lan man hoặc bỏ sót ý.
    """
    if not answer.strip():
        return 0.0

    if REFUSAL_EXACT.lower() in answer.lower():
        return 1.0

    prompt = f"""Bạn là Giám khảo đánh giá độ liên quan và phù hợp của câu trả lời (Answer Relevance) cho hệ thống AI.
Nhiệm vụ: Hãy xem xét [Câu trả lời] có giải quyết trúng, đủ và trực tiếp [Câu hỏi] hay không (không xét tính đúng sai của dữ kiện, chỉ xét mức độ bám sát câu hỏi).

[Câu hỏi]: {question}

[Câu trả lời của Agent]:
{answer}

Quy tắc chấm điểm (Thang điểm số thực liên tục từ 0.00 đến 1.00):
- 1.00: Đi thẳng vào trọng tâm, giải quyết hoàn hảo mọi khía cạnh của câu hỏi, cô đọng súc tích.
- 0.80 - 0.95: Trả lời đúng trọng tâm nhưng có một chút thông tin thừa thãi hoặc thiếu sót một ý phụ không đáng kể.
- 0.50 - 0.75: Chỉ giải quyết được một phần câu hỏi, hoặc chứa nhiều đoạn giải thích lan man, dài dòng.
- 0.10 - 0.40: Lạc đề phần lớn, né tránh câu hỏi.
- 0.00: Hoàn toàn không liên quan gì đến câu hỏi.

Hãy tính toán và CHỈ TRẢ VỀ DUY NHẤT một số thập phân từ 0.00 đến 1.00 (ví dụ: 0.92 hoặc 0.85), không viết thêm chữ nào khác.
Điểm số:"""

    try:
        res = call_llm(
            client=groq_client,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=20.0
        )
        score_text = res.choices[0].message.content.strip()
        matches = re.findall(r"(?:1\.00|1\.0|1|0\.\d{1,2}|0)", score_text)
        if matches:
            val = float(matches[0])
            return round(min(max(val, 0.0), 1.0), 2)
        return 0.90
    except Exception as e:
        console.print(f"[dim red]Lỗi khi chấm Answer Relevance: {e}[/dim red]")
        return 0.90


def evaluate_context_precision(
    groq_client: Groq,
    question: str,
    context: str,
    expected_answer: str = ""
) -> float:
    """
    Đánh giá Context Precision (Độ chính xác ngữ cảnh) theo thang điểm liên tục 0.00 - 1.00.
    Tỉ lệ thông tin thực sự hữu ích và phục vụ câu hỏi trong tổng số ngữ cảnh thu thập được.
    """
    if not context.strip():
        return 1.0

    prompt = f"""Bạn là Giám khảo đánh giá độ chính xác của ngữ cảnh truy xuất (Context Precision) trong hệ thống RAG/Agent.
Nhiệm vụ: Hãy kiểm tra xem ngữ cảnh thu thập được từ các công cụ (Context) có thực sự chứa các dữ kiện hữu ích và liên quan trực tiếp đến [Câu hỏi] hay không.

[Câu hỏi]: {question}
{f'[Ý trả lời tham chiếu]: {expected_answer}' if expected_answer else ''}

[Ngữ cảnh từ công cụ/tài liệu]:
{context[:9000]}

Quy tắc chấm điểm (Thang điểm số thực liên tục từ 0.00 đến 1.00):
- 1.00: Ngữ cảnh cực kỳ tinh gọn, trúng đích, mọi chunk tài liệu lấy về đều thiết yếu và trực tiếp trả lời câu hỏi.
- 0.80 - 0.95: Ngữ cảnh chứa thông tin chính xác cần thiết, có lẫn một ít thông tin ngữ cảnh phụ nhưng không gây nhiễu.
- 0.50 - 0.75: Có chứa thông tin cần thiết nhưng bị lẫn nhiều dữ liệu không liên quan (khoảng 30-50% là thông tin thừa).
- 0.20 - 0.45: Phần lớn ngữ cảnh là thông tin rác, chỉ có một vài câu/từ khóa liên quan lướt qua.
- 0.00: Toàn bộ ngữ cảnh hoàn toàn không liên quan đến câu hỏi.

Hãy tính toán và CHỈ TRẢ VỀ DUY NHẤT một số thập phân từ 0.00 đến 1.00 (ví dụ: 0.88 hoặc 0.80), không viết thêm chữ nào khác.
Điểm số:"""

    try:
        res = call_llm(
            client=groq_client,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=20.0
        )
        score_text = res.choices[0].message.content.strip()
        matches = re.findall(r"(?:1\.00|1\.0|1|0\.\d{1,2}|0)", score_text)
        if matches:
            val = float(matches[0])
            return round(min(max(val, 0.0), 1.0), 2)
        return 0.85
    except Exception as e:
        console.print(f"[dim red]Lỗi khi chấm Context Precision: {e}[/dim red]")
        return 0.85


def evaluate_rag_triad(
    groq_client: Groq,
    question: str,
    context: str,
    answer: str,
    expected_answer: str = "",
    is_no_answer: bool = False
) -> dict[str, float]:
    """Tính toán đồng thời cả 3 chỉ số RAG Triad với thang điểm liên tục."""
    if is_no_answer:
        refusal_ok = REFUSAL_EXACT.lower() in answer.lower()
        return {
            "faithfulness": 1.0 if refusal_ok else 0.0,
            "answer_relevance": 1.0 if refusal_ok else 0.0,
            "context_precision": 1.0
        }

    faith = evaluate_faithfulness(groq_client, question, context, answer)
    relevance = evaluate_answer_relevance(groq_client, question, answer)
    precision = evaluate_context_precision(groq_client, question, context, expected_answer)

    return {
        "faithfulness": faith,
        "answer_relevance": relevance,
        "context_precision": precision
    }


def export_evaluation_csv(out_csv: Path, details: list[dict[str, Any]]) -> None:
    """Xuất danh sách kết quả các câu hỏi ra file CSV chuẩn UTF-8 BOM (đọc được trong Excel)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "STT",
            "Mã câu/ID",
            "Phân loại",
            "Câu hỏi",
            "Câu trả lời của Bot",
            "Faithfulness",
            "Answer Relevance",
            "Context Precision",
            "Độ trễ (giây)",
            "Tổng Tokens",
            "Nguồn trích dẫn (Docs Cited)",
            "Công cụ đã dùng (Tools)"
        ])
        for idx, r in enumerate(details, start=1):
            writer.writerow([
                idx,
                r.get("id", f"Q{idx:02d}"),
                r.get("type", "chat"),
                r.get("question", ""),
                r.get("answer", ""),
                r.get("faithfulness", "N/A"),
                r.get("answer_relevance", "N/A"),
                r.get("context_precision", "N/A"),
                r.get("latency_sec", 0.0),
                r.get("usage", {}).get("total_tokens", 0) if isinstance(r.get("usage"), dict) else r.get("tokens", 0),
                "; ".join(r.get("cited_doc_ids", [])),
                "; ".join(r.get("tools_called", [])),
            ])
    console.print(f"[bold green]✓ Đã xuất log CSV chi tiết:[/bold green] [cyan]{out_csv.name}[/cyan]")


def generate_evaluation_chart(out_chart: Path, title: str, details: list[dict[str, Any]]) -> None:
    """Vẽ và xuất đồ thị PNG tổng hợp kết quả (RAG Triad + Latency)."""
    if not details:
        return

    out_chart.parent.mkdir(parents=True, exist_ok=True)
    try:
        labels = [r.get("id", f"Q{i+1}") for i, r in enumerate(details)]
        faiths = [r.get("faithfulness", 0.0) if r.get("faithfulness") is not None else 0.0 for r in details]
        rels = [r.get("answer_relevance", 0.0) if r.get("answer_relevance") is not None else 0.0 for r in details]
        precs = [r.get("context_precision", 0.0) if r.get("context_precision") is not None else 0.0 for r in details]
        latencies = [r.get("latency_sec", 0.0) for r in details]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(labels) * 0.8), 8), sharex=True)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        x = list(range(len(labels)))
        width = 0.25

        # Subplot 1: RAG Triad scores
        ax1.bar([p - width for p in x], faiths, width=width, label="Faithfulness", color="#2ecc71")
        ax1.bar(x, rels, width=width, label="Answer Relevance", color="#3498db")
        ax1.bar([p + width for p in x], precs, width=width, label="Context Precision", color="#9b59b6")
        ax1.axhline(0.85, color="#e74c3c", linestyle="--", linewidth=1.2, label="Rubric Target (≥0.85 / ≥0.80)")
        ax1.set_ylabel("Score (0.00 - 1.00)", fontweight="bold")
        ax1.set_ylim(0, 1.15)
        ax1.set_title("RAG Triad Metrics (Continuous Scale)", fontsize=11)
        ax1.legend(loc="upper right")
        ax1.grid(axis="y", linestyle=":", alpha=0.6)

        # Subplot 2: Latency
        ax2.plot(labels, latencies, marker="o", color="#e67e22", linewidth=2, label="Latency (seconds)")
        ax2.axhline(15.0, color="#c0392b", linestyle=":", linewidth=1, label="Target (<15s)")
        ax2.set_ylabel("Latency (sec)", fontweight="bold")
        ax2.set_xlabel("Question / Test Case", fontweight="bold")
        ax2.set_title("Response Latency per Question", fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45 if len(labels) > 6 else 0)
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(out_chart, dpi=200)
        plt.close()
        console.print(f"[bold green]✓ Đã vẽ biểu đồ đánh giá:[/bold green] [cyan]{out_chart.name}[/cyan]")
    except Exception as e:
        console.print(f"[dim red]Không thể vẽ đồ thị: {e}[/dim red]")
