# Week 8: Multi-Agent Architecture & Critic Verifier

## 1. General Information & Learning Objectives (Week 8)
- Design Multi-Agent coordination models: Pipeline, Handoff pattern, Orchestrator-Worker, Router, Critic-Verifier.
- Build an independent `Agent` class with a dedicated System Prompt, its own tool set and Token Usage cost tracking.
- Implement the **Producer-Critic** architecture with a clear Handoff Contract (DRAFT + EVIDENCE).
- Apply a **Bounded Revision** mechanism (limit the number of revision rounds, for example `REVISION_BUDGET=1`) to verify quality and prevent infinite loops.

## 2. Foundational Theory & System Architecture
### 2.1. Classification of Multi-Agent Patterns
- **Pipeline Pattern (ex1_2)**: A fixed sequential flow (Researcher -> Specialist -> Writer). Weakness: the final writer has the authority to decide everything and nobody checks it afterwards.
- **Delegate / Coordinator Pattern (ex1_3)**: A coordinating model decides at runtime which Worker to call.
- **Router Pattern (ex1_5)**: Classifies the question into groups and routes it to exactly 1 Agent (saves tokens).
- **Critic / Verifier Pattern (ex1_6)**: The ultimate architecture against Hallucination in production:
  - A **Worker Agent (Producer)** creates a Draft and gathers Evidence.
  - A **Critic Agent (Verifier)** works independently to review.

### 2.2. The independence principle & the Handoff Contract of the Critic Agent
- **Absolute independence**: The Critic Agent **HAS NO SEARCH TOOL** and **MUST NEVER REWRITE THE ANSWER ITSELF**.
- **Handoff Contract**: The Critic's input consists only of:
  1. The user's request (Request).
  2. The draft answer (Draft Answer).
  3. The retrieved document evidence (Evidence).
- **Critic Verdict**:
  - `VERDICT: PASS`: Every claim traces back to the Evidence; an out-of-scope question was refused safely.
  - `VERDICT: REVISE | REASON: <error>`: Fabricated information that is not in the documents, or a missing source citation, was found.
- **Bounded Revision Budget**: At most $1$ revision round is allowed (`REVISION_BUDGET=1`). If it still fails after 1 round, stop to avoid exhausting the token budget and looping forever.

## 3. Lab Exercises & Step-by-step Sample Code
### Exercise 8.1 — Independent Agent class (ex1_1_agent_class.py)
```python
class Agent:
    def __init__(self, name: str, system_prompt: str, tools: list = None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []

    def run(self, task: str) -> str:
        # An independent agent loop with token usage tracking
        return "Result processed by " + self.name
```

### Exercise 8.2 — Handoff Contract & Critic Verdict (ex1_6_critic.py)
```python
from dataclasses import dataclass, field

@dataclass
class CriticVerdict:
    verdict: str                  # "PASS" or "REVISE"
    feedback: str = ""            # Details of the errors to fix if REVISE
    unsupported_claims: list[str] = field(default_factory=list)

CRITIC_PROMPT = """You are an Independent Verification Agent.
You do NOT write the answer yourself.
Task: Check the DRAFT against the EVIDENCE.
The last line must be exactly:
VERDICT: PASS
or
VERDICT: REVISE | REASON: <briefly state the error>
"""
```

### Exercise 8.3 — Producer-Critic orchestration with Bounded Revision
```python
def run_critic_pipeline(user_question: str, revision_budget: int = 1):
    # Step 1: The Worker creates the first draft
    draft_response = worker_agent.run(user_question)

    # Step 2: Bounded quality verification
    for r in range(revision_budget):
        verdict = critic_agent.evaluate(
            request=user_question,
            draft=draft_response.answer,
            evidence=draft_response.citations
        )
        if verdict.verdict == "PASS":
            return draft_response  # Meets the standard

        # If REVISE: the Worker fixes the draft based on the Critic's feedback
        revision_prompt = f"{user_question}\n[NOTE FROM CRITIC]: {verdict.feedback}"
        draft_response = worker_agent.run(revision_prompt)

    return draft_response
```

## 4. Common Pitfalls & Design Principles (Gotchas & Best Practices)
- **Critic rewriting on its own pitfall**: If the Critic is allowed to fix the text itself, it tends to generate new hallucinations; the Critic may only give feedback and the Worker fixes the draft.
- **No revision limit (Infinite Debate) pitfall**: If two Agents argue back and forth without a `REVISION_BUDGET`, the system exhausts the API budget and hangs the user's session.

## 5. Independent Tasks & Competency Assessment (Independent Labs)
- Try the **Fault Injection (deliberately injecting fabricated claims)** technique into the Worker's draft and verify whether the Critic Verifier catches it and requests a revision.
