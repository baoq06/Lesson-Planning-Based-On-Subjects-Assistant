import re
import os
import sys
import asyncio
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from typing import Optional

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
from src.schemas import AgentResponse

load_dotenv()

def parse_verdict(text: str) -> tuple[str, int, bool]:
    raw = text or ""
    unsupported = len(re.findall(r"SUPPORTED\s*:\s*no\b", raw, re.I))
    m = re.search(r"VERDICT\s*:\s*(PASS|REVISE)", raw, re.I)
    if not m:
        return "REVISE", unsupported, True
    verdict = m.group(1).upper()
    if verdict == "PASS" and unsupported:
        verdict = "REVISE"
    return verdict, unsupported, False

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

    async def run(self, question: str) -> tuple[str, dict]:
        producer_res: AgentResponse = await run_agent_loop(question=question)
        draft = producer_res.answer

        evidence_items = [
            f"[{s.tool_call.name if s.tool_call else 'N/A'}]: {s.observation}"
            for s in producer_res.steps if s.observation
        ]
        evidence = "\n".join(evidence_items)
        self._show("producer", "critic", "DRAFT + EVIDENCE", f"DRAFT:\n{draft}\n\nEVIDENCE:\n{evidence}")

        stats = {"revisions": 0, "unsupported": 0, "parse_errors": 0, "verdicts": []}

        if USE_CRITIC:
            budget = REVISION_BUDGET
            while True:
                critic_prompt = (
                    f"REQUEST:\n{question}\n\n"
                    f"RAW EVIDENCE:\n{evidence if evidence.strip() else '(No tool evidence gathered)'}\n\n"
                    f"DRAFT:\n{draft}"
                )

                resp = call_llm(
                    client=self.client,
                    messages=[
                        {"role": "system", "content": CRITIC_SYSTEM},
                        {"role": "user", "content": critic_prompt}
                    ],
                    model=self.model,
                    temperature=0.0
                )
                verdict_text = resp.choices[0].message.content.strip()
                self._show("critic", "producer", "VERDICT", verdict_text)

                verdict, unsupported, parse_error = parse_verdict(verdict_text)
                stats["verdicts"].append((verdict, unsupported))
                stats["unsupported"] = unsupported
                stats["parse_errors"] += int(parse_error)

                if parse_error:
                    print("!! Critic returned no clean VERDICT line - failing closed (treated as REVISE)")
                
                for line in flagged_claims(verdict_text):
                    print(f"   flagged: {line}")
                if verdict == "PASS":
                    break
                if budget <= 0:
                    print("!! Revision budget exhausted - shipping draft as stands")
                    break

                budget -= 1
                stats["revisions"] += 1
                revised_task = (
                    f"{question}\n\n"
                    f"An independent critic reviewed your previous draft and flagged unsupported claims:\n"
                    f"{verdict_text}\n\n"
                    "Revise the answer so that EVERY claim is supported by course evidence "
                    "or explicitly marked as not covered. Do not fabricate facts."
                )
                revised_res = await run_agent_loop(question=revised_task)
                draft = revised_res.answer
                self._show("producer (revision)", "critic", "REVISED DRAFT", draft)
        else:
            print("!! CRITIC=0 - Draft ships unchecked (control mode)")
        return draft, stats

async def main():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=0)
    team = Team(groq_client=client)

    if len(sys.argv) > 1:
        test_question = " ".join(sys.argv[1:])
    else:
        print("=" * 60)
        print("  HỆ THỐNG MULTI-AGENT (PRODUCER + CRITIC)")
        print("=" * 60)
        try:
            user_input = input("\nNhập câu hỏi (hoặc nhấn Enter để chạy câu test bẫy quy chế mặc định): ").strip()
            test_question = user_input if user_input else "Em nộp bài trễ 30 tiếng và xin gia hạn 10 ngày do ốm, MSSV của em là 12345."
        except (KeyboardInterrupt, EOFError):
            test_question = "Em nộp bài trễ 30 tiếng và xin gia hạn 10 ngày do ốm, MSSV của em là 12345."

    print(f"\n>>> ĐANG CHẠY MULTI-AGENT CHO CÂU HỎI:\n\"{test_question}\"\n")
    final_answer, stats = await team.run(test_question)
    print("\n" + "=" * 60)
    print("  KẾT QUẢ CUỐI CÙNG SAU PHÊ DUYỆT (FINAL SHIPPED ANSWER):")
    print("=" * 60)
    print(final_answer)
    print("\n[THỐNG KÊ AUDIT & REVISION]:", stats, audit_answer(final_answer))
if __name__ == "__main__":
    asyncio.run(main())