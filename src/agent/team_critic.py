import re
import os
import sys
import asyncio
import time
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from typing import Optional

from src.agent.loop import (
    run_agent_loop,
    log_question_run,        
    CURRENT_SESSION_DIR,     
    RUN_LOG_PATH,            
    SESSION_TIMESTAMP        
)
from src.rag.eval_metrics import (
    evaluate_rag_triad,        
    export_evaluation_csv,     
    generate_evaluation_chart  
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
from src.agent.loop import run_agent_loop
from src.agent.call_llm import call_llm
from src.schemas import AgentResponse, ProducerHandoff, CriticReview

load_dotenv()

def parse_critic_review(text: str) -> CriticReview:
    # Chuyển đổi text thô từ Critic LLM thành Handoff Contract CriticReview.
    raw = text or ""
    issues = [line.strip() for line in flagged_claims(raw)]
    m = re.search(r"VERDICT\s*:\s*(PASS|REVISE)", raw, re.I)
    
    if not m:
        # Nếu Critic không xuất dòng VERDICT chuẩn -> Fail-closed thành REVISE
        return CriticReview(
            verdict="REVISE",
            critique=raw,
            issues=issues,
            suggested_fix="Critic returned no clean VERDICT line (failing closed to REVISE)."
        )
    
    verdict = m.group(1).upper()
    suggested_fix = "Revise draft to remove ungrounded claims." if issues else None
    return CriticReview(
        verdict=verdict,
        critique=raw,
        issues=issues,
        suggested_fix=suggested_fix
    )

def flagged_claims(verdict_text: str) -> list[str]:
    return [
        ln.strip() for ln in (verdict_text or "").splitlines()
        if re.search(r"SUPPORTED\s*:\s*no\b", ln, re.I)
    ]

def audit_answer(answer: str) -> dict:
    citations = set(re.findall(r"\[(POLICY_AIPR|LP_W\d{2})\]", answer or ""))
    refusal = bool(re.search(
        r"(not found in the aipr course materials)",
        answer or "",
        re.I
    ))
    return {
        "citations_count": len(citations),
        "citations": sorted(citations),
        "is_refusal": refusal
    }

WATCH = os.getenv("WATCH", "1") != "0"
USE_CRITIC = os.getenv("CRITIC", "1") != "0"
REVISION_BUDGET = int(os.getenv("REVISION_BUDGET", "1"))

CRITIC_PROMPT_PATH = REPO_ROOT / "data" / "prompts" / "critic_prompt.txt"
CRITIC_SYSTEM = CRITIC_PROMPT_PATH.read_text(encoding="utf-8").strip()

class Team:
    def __init__(self, groq_client: Groq, model: Optional[str] = None):
        self.client = groq_client
        self.model = model or os.getenv("LLM_MODEL", "qwen/qwen3.8-27b").strip()

    def _show(self, frm: str, to: str, label: str, block: str) -> None:
        if not WATCH:
            return
        print(f"\n{'-' * 72}\n{frm} -> {to}   [{label}: {len(block)} chars]\n{'-' * 72}")
        print(block)

    async def run(self, question: str, history: Optional[list[dict]] =  None) -> tuple[str, dict]:
        producer_res: AgentResponse = await run_agent_loop(
            question=question,
            history=history,
            auto_log=False
            )
        draft = producer_res.answer
        last_res = producer_res

        evidence_items = [
            f"[{s.tool_call.name if s.tool_call else 'N/A'}]: {s.observation}"
            for s in producer_res.steps if s.observation
        ]
        evidence = "\n".join(evidence_items)

        # Đóng gói Handoff contract từ Producer
        handoff = ProducerHandoff(
          question=question,
          draft=draft,
          evidence=evidence,
          revision_budget=REVISION_BUDGET
      )

        self._show("producer", "critic", "DRAFT + EVIDENCE", f"DRAFT:\n{draft}\n\nEVIDENCE:\n{evidence}")

        stats = {"revisions": 0, "unsupported": 0, "parse_errors": 0, "verdicts": []}

        if USE_CRITIC:
            budget = REVISION_BUDGET
            while True:
                resp = call_llm(
                    client=self.client,
                    messages=[
                        {"role": "system", "content": CRITIC_SYSTEM},
                        {"role": "user", "content": handoff.to_critic_prompt()}
                    ],
                    model=self.model,
                    temperature=0.0
                )
                verdict_text = resp.choices[0].message.content.strip()
                self._show("critic", "producer", "VERDICT", verdict_text)

                review: CriticReview = parse_critic_review(verdict_text)

                stats["verdicts"].append((review.verdict, len(review.issues)))
                stats["unsupported"] = len(review.issues)
                if review.suggested_fix and "failing closed" in review.suggested_fix:
                    stats["parse_errors"] += 1
                    print(f"!! {review.suggested_fix}")

                for line in review.issues:
                  print(f"   flagged: {line}")
              
                # Kiểm tra điều kiện dừng dựa trên contract
                if review.verdict == "PASS":
                    break
                if budget <= 0:
                    print("!! Revision budget exhausted - shipping draft as stands")
                    break

                # chuyển giao yêu cầu sửa đổi
                budget -= 1
                stats["revisions"] += 1
                revised_task = (
                    f"{question}\n\n"
                    f"An independent critic reviewed your previous draft and flagged unsupported claims:\n"
                    f"{review.critique}\n\n"
                    "Revise the answer so that EVERY claim is supported by course evidence "
                    "or explicitly marked as not covered. Do not fabricate facts."
                )
                revised_res = await run_agent_loop(
                    question=revised_task,
                    history=history,
                    auto_log=False
                    )
                draft = revised_res.answer
                last_res = revised_res

                # Cập nhật lại draft trong contract cho vòng lặp tiếp theo nếu có
                handoff.draft = draft
                self._show("producer (revision)", "critic", "REVISED DRAFT", draft)
        else:
            print("!! CRITIC=0 - Draft ships unchecked (control mode)")
        return draft, stats, last_res

def main():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=0)
    team = Team(groq_client=client)
    print("=" * 64)
    print("   HỆ THỐNG MULTI-AGENT AIPR (PRODUCER + CRITIC CLI)")
    print("=" * 64)
    print("Gõ câu hỏi và nhấn Enter. Gõ 'exit' để thoát. Gõ 'clear' để xóa nhớ.\n")
    conversation_history: list[dict] = []
    session_records: list[dict] = []
    groq_eval_client: Optional[Groq] = None
    while True:
        try:
            user_question = input("\nBạn: ").strip()
            if not user_question:
                continue
            if user_question.lower() in ["exit", "quit", "q"]:
                print("\nĐang kết thúc phiên và xuất báo cáo đánh giá...")
                break
            if user_question.lower() == "clear":
                conversation_history.clear()
                print("\n[Hệ thống] Đã làm mới bộ nhớ hội thoại.")
                continue
            print(f"\n[Multi-Agent] Đang xử lý câu hỏi: \"{user_question}\"...")
            t0 = time.time()
            
            final_answer, stats, last_res = asyncio.run(
                team.run(question=user_question, history=conversation_history)
            )
            elapsed = time.time() - t0

            print("\n" + "=" * 60)
            print("TRỢ LÝ (Đã qua Critic kiểm duyệt):")
            print("=" * 60)
            print(final_answer)
            print("\n[Thống kê Critic]:", stats)

            if final_answer:
                conversation_history.append({"role": "user", "content": user_question})
                conversation_history.append({"role": "assistant", "content": final_answer})
            # Tính toán 3 chỉ số RAG Triad liên tục cho câu hỏi
            context_text = "\n".join([str(s.observation) for s in last_res.steps if s.observation])
            cited_docs = [
                d for d in ["POLICY_AIPR", "LP_W01", "LP_W02", "LP_W03", "LP_W04", "LP_W05", "LP_W06"]
                if d in (final_answer or "")
            ]
            if not groq_eval_client:
                groq_eval_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            triad = evaluate_rag_triad(
                groq_client=groq_eval_client,
                question=user_question,
                context=context_text,
                answer=final_answer or "",
            )
            # Xuất trực tiếp 3 chỉ số RAG Triad vào file log run_*.jsonl (không in ra terminal)
            log_question_run(
                question=user_question,
                steps=last_res.steps,
                answer=final_answer or "",
                usage=last_res.usage if isinstance(last_res.usage, dict) else {},
                rag_triad=triad,
                latency_sec=round(elapsed, 2),
                cited_docs=cited_docs,
            )
            # Gom record phục vụ xuất CSV & Chart cuối phiên
            session_records.append({
                "id": f"Demo_Q{len(session_records)+1:02d}",
                "type": "demo",
                "question": user_question,
                "answer": final_answer,
                "faithfulness": triad["faithfulness"],
                "answer_relevance": triad["answer_relevance"],
                "context_precision": triad["context_precision"],
                "latency_sec": round(elapsed, 2),
                "usage": last_res.usage if isinstance(last_res.usage, dict) else {},
                "cited_doc_ids": cited_docs,
                "tools_called": [s.tool_call.name for s in last_res.steps if s.tool_call],
            })
        except (KeyboardInterrupt, EOFError):
            print("\n\nĐã nhận lệnh thoát phiên. Đang xuất báo cáo...")
            break

    # Khi kết thúc 1 phiên demo: Xuất CSV và Đồ thị PNG tổng hợp toàn bộ các câu trong phiên
    if session_records:
        demo_csv = CURRENT_SESSION_DIR / f"demo_run_{SESSION_TIMESTAMP}.csv"
        demo_chart = CURRENT_SESSION_DIR / f"demo_run_{SESSION_TIMESTAMP}_chart.png"
        print("\n── TỔNG KẾT PHIÊN LÀM VIỆC (DEMO SESSION) ──")
        export_evaluation_csv(demo_csv, session_records)
        generate_evaluation_chart(demo_chart, f"AIPR Interactive Demo Session Benchmark ({SESSION_TIMESTAMP})", session_records)
        print(f"✓ Đã hoàn thành lưu phiên demo ({len(session_records)} câu hỏi):")
        print(f"  • Thư mục Session: {CURRENT_SESSION_DIR}")
        print(f"  • File JSONL:      {RUN_LOG_PATH}")
        print(f"  • File CSV:        {demo_csv}")
        print(f"  • Đồ thị PNG:      {demo_chart}\n")
    else:
        print("Chưa có câu hỏi nào được thực hiện trong phiên này.")
if __name__ == "__main__":
    main()