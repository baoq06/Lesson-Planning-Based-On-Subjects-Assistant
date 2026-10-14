# Week 5 — Lecture Notes 

## RAG Evaluation & Groundedness

> **Course:** AI Programming with Python
> **Week:** 5 of 8 — 2 sessions/week × 4 hours (2h theory + 2h lab per session; theory total 4h/week)
> **Prerequisites:** Week 4 (RAG pipeline, hybrid search, re-ranking, metadata filtering), Week 3 (VectorStore, embeddings), Week 2 (LLM API)
> **Learning Objectives:**
>
> - A5.1 Compute faithfulness (answer claims are supported by context) and answer relevance (answer addresses the question).
> - A5.2 Use the RAGAS framework to evaluate a RAG pipeline quantitatively.
> - A5.3 Implement a citation verification function that checks each claim against source chunks.
> - A5.4 Identify and diagnose common RAG failure modes (missing context, irrelevant retrieval, hallucination).

---

## Session 1 — RAG Metrics: Faithfulness, Relevance, Precision

> **Session format:** 4 hours total = **Part A — Theory (2h)** (this guide) + **Part B — Lab (2h)** (`lab_sheet.md`).

### Session Structure

| # | Segment | Duration | Gagné Event | Activity |
|---|---|---|---|---|
| — | **PART A — THEORY (2h)** | 0:00–2:00 | — | Segments 1–3, 5–6, 8–9, 12–13 |
| 1 | Hook & Problem | 0:00–0:10 | 1. Gain attention | "Confident, fluent, and wrong" — how to catch it |
| 2 | Objectives & Recall | 0:10–0:20 | 2–3. Objectives + Recall | Recap pipeline; precision/recall prior knowledge |
| 3 | Concept 1: Faithfulness | 0:20–0:45 | 4. Present content | Claim decomposition + LLM-as-judge |
| 5 | KC 1 | 0:45–0:50 | 7. Feedback | MCQ: high faithfulness means... |
| 6 | Concept 2: Relevance & Precision | 0:50–1:15 | 4. Present content | LLM-as-judge for relevance; precision formula |
| — | Break | 1:15–1:20 | — | — |
| 9 | Concept 3: Evaluation Pipeline | 1:20–1:40 | 4. Present content | Test set → run → aggregate |
| 12 | KC 2 | 1:40–1:45 | 8. Assess | MCQ: faithfulness 0.9 + relevance 0.3 |
| 13 | Wrap & Reflect | 1:45–2:00 | 9. Retention/Transfer | Preview RAGAS |
| — | **PART B — LAB (2h)** | 2:00–4:00 | — | Segments 4, 7, 10–11 — see `lab_sheet.md` |
| 4 | Code Along 1 | 2:00–2:30 | 5–6. Guidance + Practice | `check_faithfulness()` |
| 7 | Code Along 2 | 2:30–2:55 | 5–6. Guidance + Practice | `answer_relevance()` + `context_precision()` |
| 10 | Independent Lab: RAG Evaluator | 2:55–3:35 | 6. Elicit performance | 5-question test set + diagnosis |
| 11 | Peer Review | 3:35–3:50 | 7. Feedback | Are the judge prompts fair? |
| — | Buffer / questions | 3:50–4:00 | — | — |

---

### 1. Hook & Problem (0:00–0:10)

**Instructor script:**

> "Here's an answer from a RAG system about the remote work policy:"
>
> > "Employees may work remotely up to five days per week, with no approval required. Remote workers receive a $200 monthly stipend for home internet."
>
> "The answer is **confident**. It's **fluent**. And it is **completely wrong** — the actual policy is three days per week, prior approval required, no stipend.
>
> "Why did this happen? The retrieval may have found the wrong chunks, or the LLM may have ignored the context and answered from its training data. Either way, the user got a confident, fluent, wrong answer.
>
> "**How do we catch this automatically?** We can't read every answer manually. We need metrics — numbers that tell us whether the answer is grounded in the retrieved context.
>
> "This week you'll build the measurement layer: faithfulness, relevance, precision — and then automate it with RAGAS. By the end, you'll be able to say: 'My RAG system scores 0.92 faithfulness on this test set' — and mean something real."

**Cognitive load note:** The hook shows the *problem* (confident wrong answers) before the *solution* (metrics). Students see why evaluation matters before learning how to measure — Merrill's problem-centered principle.

---

### 2. Objectives & Recall (0:10–0:20)

**State objectives:**

> "By the end of this session, you will be able to:
> 1. Decompose an answer into claims and check each against the context (faithfulness).
> 2. Measure whether an answer addresses the question (relevance).
> 3. Measure whether retrieved chunks are actually relevant (context precision).
> 4. Build an evaluation pipeline over a test set."

**Recall sprint (pair, 2 min):**

> "In 30 seconds, explain to your partner:
> 1. What does your `rag_pipeline()` return? *(answer, sources)*
> 2. What is precision? What is recall? *(from Week 3 threshold discussion / databases)*
> 3. What keeps an LLM grounded in RAG? *(the grounding prompt)*"

**Why this helps:** Evaluation reuses the pipeline, and precision/recall are re-used from Week 3's threshold discussion. Activating them now reduces load.

---

### 3. Concept 1 — Faithfulness (Groundedness) (0:20–0:45)

#### The Core Idea

**Show on projector:**

```
ANSWER (from RAG):
"Employees may work remotely up to five days per week."

     │  STEP 1: DECOMPOSE into claims
     ▼
Claim 1: "Employees may work remotely"
Claim 2: "up to five days per week"

     │  STEP 2: Check EACH claim against the context (LLM-as-judge)
     ▼
Context: "Employees may work remotely up to three days per week..."

Claim 1 → SUPPORTED ✓
Claim 2 → NOT SUPPORTED ✗  (context says THREE, answer says FIVE)

     │  STEP 3: AGGREGATE
     ▼
faithfulness = supported_claims / total_claims = 1/2 = 0.5
```

**Key insight:** "Faithfulness answers one question: **is every claim in the answer supported by the retrieved context?** It does NOT measure whether the answer is correct in the real world — only whether it's grounded in what we gave the LLM. A hallucinated answer scores LOW faithfulness; a correct-but-unrelated answer could score HIGH faithfulness but low relevance."

#### Worked Example: The LLM-as-Judge Pattern

```python
JUDGE_PROMPT = """You are a strict verifier.
Check whether the given claim is SUPPORTED by the context.

Context:
{context}

Claim:
{claim}

Is this claim supported by the context? Answer only "YES" or "NO"."""

def judge_claim(claim: str, context:str)->bool:
    """Return True if the LLM judge says the context supports the claim."""
    verdict = ask(JUDGE_PROMPT.format(context=context, claim=claim))
    return verdict.upper().startswith("YES")
```

**Teaching tip — why temperature=0.0 for the judge:** The judge LLM is not being creative — it's being a measuring instrument. Deterministic output means consistent verdicts. If the judge used temperature=1.0, the same claim could be judged YES one time and NO the next — the metric would be meaningless.

> **🌡️ Instrument analogy (remember this):** *A judge LLM is like a thermometer — you don't ask it to be creative, you ask it to measure consistently. That's why temperature=0.0.* This analogy applies to EVERY judge prompt you write this week and in Lab 5.

---

### 4. Code Along 1 — check_faithfulness (2:00–2:30) [PART B — LAB]

**File:** `ex1_1_faithfulness.py`

```python
"""Exercise 1.1 — Faithfulness: is the RAG answer grounded in its context?

Pipeline (3 steps):
    1. Decompose the answer into individual factual claims
    2. Ask an LLM judge whether each claim is SUPPORTED by the context
    3. Score = supported claims / total claims   (1.0 = fully faithful)
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

def ask(prompt):
    """Send one prompt to the LLM and return its reply as text."""
    reply = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # 0 = deterministic: we want a verdict, not creativity
    )
    return reply.choices[0].message.content.strip()

# ===== Step 1: Decompose the answer into claims =====

DECOMPOSE_PROMPT = """Break this answer into individual factual claims.
Return one claim per line. Do not add anything else.

Answer:
{answer}"""

def decompose_claims(answer):
    """Split an answer into a list of individual factual claims."""
    text = ask(DECOMPOSE_PROMPT.format(answer=answer))
    return [line.strip("-• ") for line in text.splitlines() if line.strip()]

# ===== Step 2: Judge each claim against the context =====

JUDGE_PROMPT = """You are a strict verifier.
Check whether the given claim is SUPPORTED by the context.

Context:
{context}

Claim:
{claim}

Is this claim supported by the context? Answer only "YES" or "NO"."""

def judge_claim(claim: str, context:str)->bool:
    """Return True if the LLM judge says the context supports the claim."""
    verdict = ask(JUDGE_PROMPT.format(context=context, claim=claim))
    return verdict.upper().startswith("YES")

# ===== Step 3: Put it together and score =====

def check_faithfulness(answer:str, context_chunks:list)->list[dict]:
    """Check each claim in `answer` against the retrieved context chunks.

    Args:
        answer: The RAG answer text.
        context_chunks: List of DocumentChunk objects (each with `.content`
            and `.metadata`) retrieved by search().

    Returns a list of (claim, supported) pairs.
    """
    # Join chunk contents into a single context text for the judge
    context = "\n\n".join(c.content for c in context_chunks)
    claims = decompose_claims(answer)
    return [(claim, judge_claim(claim, context)) for claim in claims]

def faithfulness_score(results:list[dict])->float:
    """Score = fraction of supported claims, in [0, 1]."""
    if not results:
        return 0.0
    
    supported = sum(ok for _, ok in results)
    return supported / len(results)

# ===== Test =====
if __name__ == "__main__":
    # Fake chunks for testing (each chunk has .content and .metadata)
    @dataclass
    class FakeChunk:
        content: str
        metadata: dict[str, str] = field(default_factory=dict)

    context = [
        FakeChunk("Employees may work remotely up to three days per week.", {"source": "policy.txt"}),
    ]

    answers = [
        ("good", "Employees may work remotely up to three days per week."),
        ("bad",  "Employees may work remotely up to five days per week, with a stipend."),
    ]

    for label, answer in answers:
        print(f"\nAnswer ({label}): {answer}")
        results = check_faithfulness(answer, context)
        for claim, ok in results:
            verdict = "supported" if ok else "NOT supported"
            print(f"  - {claim}  ->  {verdict}")
        print(f"Faithfulness score: {faithfulness_score(results):.2f}")
```

**Common mistakes to watch for:**

| Mistake | Correction |
|---|---|
| Judge uses temperature > 0 → inconsistent verdicts | `temperature=0.0` — the judge is a measuring instrument |
| Claims not decomposed (whole answer judged as one claim) | Always `decompose_claims()` first |
| Joining context loses chunk boundaries | Keep chunk source metadata for traceability |
| Verdict parsing: "YES, because..." fails `startswith("YES")` | This actually works — but show students the parsing pattern |

---

### 5. Knowledge Check 1 (0:45–0:50)

**MCQ:**

> A high faithfulness score means:
>
> (a) The answer is correct
> (b) The answer is supported by the context
> (c) The answer is relevant to the question

**Correct answer: (b) The answer is supported by the context**

**Explanation:** Faithfulness measures grounding — whether claims in the answer are supported by the retrieved context. It does NOT measure real-world correctness (the context itself could be outdated or wrong) nor relevance (an answer can be fully grounded in context but not answer the question asked).

---

### 6. Concept 2 — Answer Relevance & Context Precision (0:50–1:15)

#### The Three-Metric Framework

**Show on projector:**

```
          ┌──────────────────────────────────────────────────┐
          │        THE THREE METRICS OF RAG QUALITY          │
          ├──────────────────────────────────────────────────┤
          │  FAITHFULNESS   = is the answer grounded?        │
          │                  (claims supported by context)   │
          │                                                  │
          │  ANSWER RELEVANCE = does the answer address      │
          │                  the question?                   │
          │                                                  │
          │  CONTEXT PRECISION = are the retrieved chunks    │
          │                  actually relevant?              │
          └──────────────────────────────────────────────────┘
          Grounded + Relevant + Precise = trustworthy RAG
```

> **Persistent metric cheat-sheet — keep visible for BOTH sessions:**
>
> ```
> METRIC          ASKS                    MEASURES            TYPE
> ─────────────────────────────────────────────────────────────────
> faithfulness Is the answer          claims supported    LLM-judge
>                 grounded?              / total claims
>                 (answer ↔ context)
> relevance    Does the answer        answer addresses    LLM-judge
>                 address the            the question?
>                 question?              YES/NO
>                 (answer ↔ question)
> precision    Are retrieved          relevant chunks     formula
>                 chunks relevant?       / total chunks
>                 (chunks ↔ question)
> recall       Were all relevant      relevant found      formula +
>                 chunks found?          / all relevant      ground truth
>                 (corpus ↔ retrieval)   in corpus
> ```
>
> **⚠️ Don't mix these up** — they ask different questions and measure different things. When in doubt, check the cheat-sheet.

**Key insight:** "Faithfulness alone is not enough. An answer can be perfectly grounded in the context (high faithfulness) but completely miss the question (low relevance). Or the context can contain the right chunks buried among irrelevant ones (low precision). You need all three."

**Contrastive think-pair-share (2 min):**

> "System A has faithfulness=0.9, relevance=0.3. System B has faithfulness=0.3, relevance=0.9. What is likely different about the two systems? Discuss with your partner."
>
> *(Answer: A retrieves wrong chunks but answers them faithfully → retrieval problem; B retrieves right chunks but the prompt/grounding fails → generation problem.)*

#### Worked Example: Context Precision Formula

```
context_precision = relevant_chunks / total_retrieved_chunks

For a query with 3 retrieved chunks where only 2 are actually relevant:
  context_precision = 2/3 = 0.67
```

---

### 7. Code Along 2 — answer_relevance + context_precision (2:30–2:55) [PART B — LAB]

**File:** `ex1_2_relevance_precision.py`

```python
# ===== Answer Relevance (LLM-as-judge) =====
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

def ask(prompt: str) -> str:
    """Send one prompt to the LLM (temperature 0 = verdicts, not creativity)."""
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    return r.choices[0].message.content.strip()


def judge(prompt: str) -> bool:
    """Ask a strict YES/NO question and return the verdict."""
    return ask(prompt).upper().startswith("YES")

# ===== Step 1: Answer Relevance (LLM-as-judge) =====

RELEVANCE_PROMPT = """You are evaluating a RAG system. A user asked a question and the system answered.
Determine whether the answer actually addresses the question.

Question:
{question}

Answer:
{answer}

Is the answer relevant to the question? Answer only "YES" or "NO"."""

def answer_relevance(question: str, answer: str) -> bool:
    """Does the answer address the question?"""
    return judge(RELEVANCE_PROMPT.format(question=question, answer=answer))

# ===== Step 2: Context Precision (LLM-as-judge per chunk) =====

CHUNK_PROMPT = """Does this chunk contain information relevant to the question?

Question: {question}

Chunk: {chunk}

Answer only "YES" or "NO"."""

def context_precision(question: str, chunks: list) -> float:
    """Fraction of retrieved chunks that are actually relevant (0..1)."""
    if not chunks:
        return 0.0
    hits = 0
    for chunk in chunks:
        prompt = CHUNK_PROMPT.format(question=question, chunk=chunk.content[:500])
        if judge(prompt):
            hits += 1
    return hits / len(chunks)
```

**Teaching tip:** Point out the contrast — faithfulness and relevance are *LLM-as-judge* (qualitative judgment), while context precision is *formula-based* (count of relevant over total). The LLM judges the numerator; the formula does the math.

---

### 8. Break (1:15–1:20)

---

### 9. Concept 3 — Evaluation Pipeline (1:20–1:40)

#### Worked Example: Test Set + Aggregation

```python
@dataclass
class EvalExample:
    """One question in the evaluation test set."""
    question: str
    expected_contexts: list[str]  # doc_ids or source names that SHOULD be retrieved
    reference_answer: str = ""    # gold answer (optional)

def evaluate(rag_pipeline_fn, test_set: list[EvalExample]) -> list[dict]:
    """
    Run the RAG pipeline over a test set and compute metrics per question.

    Returns:
        List of {"question", "faithfulness", "relevance", "context_precision", "answer"} dicts
    """
    results = []
    for example in test_set:
        answer, sources = rag_pipeline_fn(example.question)

        context_chunks = [c for c, _ in sources]
        claim_results = check_faithfulness(answer, context_chunks)

        results.append({
            "question": example.question,
            "answer": answer,
            "faithfulness": faithfulness_score(claim_results),
            "relevance": answer_relevance(example.question, answer),
            "context_precision": context_precision(example.question, context_chunks),
        })
    return results

def aggregate(results: list[dict]) -> dict:
    """Aggregate per-question metrics into summary statistics."""
    import statistics
    return {
        "n_questions": len(results),
        "faithfulness_mean": statistics.mean(r["faithfulness"] for r in results),
        "relevance_mean": statistics.mean(r["relevance"] for r in results),
        "context_precision_mean": statistics.mean(r["context_precision"] for r in results),
    }
```

**Key insight:** "Evaluation is a pipeline too — test set → run pipeline → per-question metrics → aggregation. The whole point of the metrics is to give you a *number you can act on*: 'context precision is low' → 'fix retrieval'; 'faithfulness is low' → 'strengthen the prompt'."

---

### 10. Independent Lab — RAG Evaluator (2:55–3:35) [PART B — LAB]

**File:** `ex1_3_rag_evaluator.py`

Students create a 5-question test set, run the pipeline, compute the 3 metrics, and diagnose the worst question. Full instructions in `lab_sheet.md`.

**Teaching tips:**

- Emphasize the test set design: include the 3 question types (direct / synthesis / no-answer) from Week 4 — a test set that only has easy questions gives a falsely high score
- Watch for students judging chunks in a loop (many API calls) — remind them to batch or limit
- The "diagnose the worst question" step is the learning moment: metric tells you WHAT, you must figure out WHY

---

### 11. Peer Review (3:35–3:50) [PART B — LAB]

**Pair up. Review checklist:**

> 1. Are the judge prompts fair? (no leading language, clear YES/NO instruction)
> 2. Is the test set balanced? (mix of direct/synthesis/no-answer questions)
> 3. Are the metrics computed consistently? (same judge prompt across all questions)
> 4. Is the diagnosis of the worst question plausible?

---

### 12. Knowledge Check 2 (1:40–1:45)

**MCQ:**

> The RAG pipeline gets a faithfulness score of 0.9 but answer relevance of 0.3. What's the likely issue?
>
> (a) The LLM is hallucinating
> (b) The retrieval is returning irrelevant chunks
> (c) The embedding model is poor

**Correct answer: (b) The retrieval is returning irrelevant chunks**

**Explanation:** Faithfulness is high (answers ARE grounded in context) but relevance is low (answers don't address the question). If the retrieved chunks are topically related but don't contain the actual answer, the LLM produces a grounded-but-irrelevant answer. High faithfulness + low relevance → retrieval problem, not hallucination.

---

### 13. Wrap & Reflect (1:45–2:00)

**Instructor summary:**

> "Today you built the measurement layer:
> - **Faithfulness** — every claim checked against context (LLM-as-judge)
> - **Answer relevance** — does the answer address the question?
> - **Context precision** — are the retrieved chunks actually relevant?
> - **Evaluation pipeline** — test set → metrics → aggregation
>
> Session 2: manual evaluation of 5 questions is fine. But what about 500? We'll automate with RAGAS — and learn the five failure modes that explain WHY metrics are low."

**Reflection prompt (3 min):**

> "Which metric do you think is most important for your own RAG use case? Why? What surprised you about the LLM-as-judge pattern?"

---

## Session 2 — RAGAS Framework & Failure Mode Analysis

> **Session format:** 4 hours total = **Part A — Theory (2h)** (this guide) + **Part B — Lab (2h)** (`lab_sheet.md`).

### Session Structure

| # | Segment | Duration | Gagné Event | Activity |
|---|---|---|---|---|
| — | **PART A — THEORY (2h)** | 0:00–2:00 | — | Segments 1–3, 5–6, 8–9, 12–13 |
| 1 | Hook | 0:00–0:10 | 1. Gain attention | "5 questions by hand vs 500 questions automated" |
| 2 | Objectives & Recall | 0:10–0:20 | 2–3. Objectives + Recall | Recap 3 metrics; "what did you learn in the lab?" |
| 3 | Concept 1: RAGAS | 0:20–0:45 | 4. Present content | Install, run, interpret RAGAS scores |
| 5 | KC 1 | 0:45–0:50 | 7. Feedback | MCQ: context_recall definition |
| 6 | Concept 2: Failure Modes | 0:50–1:15 | 4. Present content | 5 failure modes + fixes |
| — | Break | 1:15–1:20 | — | — |
| 9 | Concept 3: Improving RAG from Metrics | 1:20–1:40 | 4. Present content | Metric → fix mapping |
| 12 | KC 2 | 1:40–1:45 | 8. Assess | MCQ: fidelity 0.95 + precision 0.30 |
| 13 | Wrap & Reflect | 1:45–2:00 | 9. Retention/Transfer | Preview Week 6 |
| — | **PART B — LAB (2h)** | 2:00–4:00 | — | Segments 4, 7, 10–11 — see `lab_sheet.md` |
| 4 | Code Along 1 | 2:00–2:30 | 5–6. Guidance + Practice | RAGAS on the Session 1 test set |
| 7 | Code Along 2 | 2:30–2:55 | 5–6. Guidance + Practice | Instrument pipeline for failure detection |
| 10 | Independent Lab: RAG Iteration | 2:55–3:35 | 6. Elicit performance | One targeted change → re-measure |
| 11 | Peer Review | 3:35–3:50 | 7. Feedback | Was the improvement measurable? |
| — | Buffer / questions | 3:50–4:00 | — | — |

---

### 1. Hook (0:00–0:10)

**Instructor script:**

> "Yesterday you evaluated 5 questions by hand. Each question needed several LLM calls — decompose, judge each claim, judge relevance, judge each chunk. For 5 questions that's maybe 30 LLM calls.
>
> "Now imagine your company has 500 questions, or you want to compare two RAG configurations (different chunk sizes, different retrieval strategies) — you need to evaluate both, 500 questions each.
>
> "That's 6,000 LLM calls by hand. Not feasible.
>
> "Enter **RAGAS** — a framework that standardises these metrics, runs them in batches, and gives you comparable numbers across configurations. Today you'll learn to use it, and you'll learn the five failure modes that explain WHY your metrics are low — so the numbers actually tell you what to fix."

---

### 2. Objectives & Recall (0:10–0:20)

**State objectives:**

> "By the end of this session, you will be able to:
> 1. Run RAGAS to compute standardised RAG metrics automatically.
> 2. Compare manual evaluation (Session 1) with RAGAS scores.
> 3. Identify the five common RAG failure modes.
> 4. Use metrics to drive one targeted improvement, and verify it."

**Recall sprint (pair, 2 min):**

> "In 30 seconds, explain to your partner:
> 1. What does faithfulness measure? *(claims supported by context)*
> 2. What is the difference between relevance and precision? *(answer vs chunks)*
> 3. What did you find in your lab's worst-performing question?"

---

### 3. Concept 1 — RAGAS Framework (0:20–0:45)

#### What RAGAS Adds

**Show on projector:**

```
MANUAL (Session 1):                        RAGAS (automated):
  you write judge prompts                   standardised metrics:
  you call the LLM yourself                   answer_relevancy
  you aggregate by hand                       faithfulness
  hard to compare configs                     context_recall
                                              context_precision
                                              batch evaluation
                                              comparable across configs
```

**Key insight:** "RAGAS doesn't do anything you couldn't do by hand in Session 1 — it standardises it. The value is (a) you don't re-invent judge prompts, (b) metrics are consistent, and (c) you can compare configuration A vs B objectively."

#### RAGAS Metrics

| Metric | What it measures | Your Session 1 equivalent |
|---|---|---|
| `faithfulness` | Claims in answer supported by context | `check_faithfulness()` |
| `answer_relevancy` | Answer addresses the question | `answer_relevance()` |
| `context_precision` | Retrieved chunks are relevant | `context_precision()` |
| `context_recall` | All relevant chunks from corpus were retrieved | (new — needs a reference set) |

> **RAGAS Dataset schema (copy the builder, don't re-derive it):**
>
> ```
> question: str                 ← the question
> answer: str                   ← the RAG answer
> contexts: list[list[str]]     ← chunk contents, one list per question
> ground_truth: str             ← reference answer (required for recall)
> ```
>
> The learning is in READING the RAGAS scores — constructing the Dataset is plumbing. Use the `build_ragas_dataset()` function as-is.

---

### 4. Code Along 1 — Run RAGAS (2:00–2:30) [PART B — LAB]

**File:** `ex2_1_ragas.py`

```python
# Install first: pip install ragas datasets

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# ===== Build the test set in RAGAS format =====

def build_ragas_dataset(test_set, answers, sources_list) -> Dataset:
    """
    Convert evaluation results into a RAGAS Dataset.

    Args:
        test_set: list of EvalExample (question, expected_contexts)
        answers: list of answer strings (one per question)
        sources_list: list of lists of chunk content (one per question)

    Returns:
        datasets.Dataset in RAGAS schema
    """
    data = {
        "question": [ex.question for ex in test_set],
        "answer": answers,
        "contexts": sources_list,   # list of lists of chunk content strings
        "ground_truth": [ex.reference_answer for ex in test_set],
    }
    return Dataset.from_dict(data)

# ===== Run evaluation =====

def run_ragas(test_set, rag_pipeline_fn) -> dict:
    """Run the RAG pipeline and evaluate with RAGAS."""
    answers, sources_list = [], []
    for ex in test_set:
        answer, sources = rag_pipeline_fn(ex.question)
        answers.append(answer)
        sources_list.append([c.content for c, _ in sources])

    dataset = build_ragas_dataset(test_set, answers, sources_list)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    return result

# ===== Compare with manual evaluation =====
# Run this after Session 1's manual metrics — the scores should be in the same ballpark.
```

**Teaching tip:** If RAGAS is slow or has dependency issues, prepare a fallback: run `evaluate()` on a 2-question subset first, and note that RAGAS calls the LLM under the hood (same cost structure as manual evaluation).

---

### 5. Knowledge Check 1 (0:45–0:50)

**MCQ:**

> What does RAGAS's `context_recall` measure?
>
> (a) How many relevant chunks were retrieved
> (b) How many relevant items from the full corpus were retrieved
> (c) How fast the retrieval is

**Correct answer: (b) How many relevant items from the full corpus were retrieved**

**Explanation:** Context recall = (relevant chunks retrieved) / (ALL relevant chunks in the corpus). Unlike context precision (which measures the retrieved set), recall measures whether retrieval *missed* any relevant chunks. It requires a reference set (ground truth) of what SHOULD have been retrieved.

---

### 6. Concept 2 — RAG Failure Modes (0:50–1:15)

> **Teaching note:** Split into two micro-cycles. **Micro-cycle A** (15 min) covers the three *content* failure modes students most likely saw in their own labs. **Micro-cycle B** (10 min) covers the two *presentation* failure modes (subtler, advanced).

#### Micro-Cycle A — The Three Content Failure Modes (15 min)

**Show on projector:**

```
FAILURE MODE          SYMPTOM                  FIX
─────────────────────────────────────────────────────────────
1. Missing context    No relevant chunks       Increase top_k, better
                      retrieved → LLM can't    chunking, hybrid search
                      answer → low recall

2. Hallucination      LLM ignores context,     Strengthen grounding
                      answers from training    prompt, temperature=0.0,
                      data → low faithfulness  verify claims

3. Irrelevant         Chunks topically         Metadata filtering,
   retrieval          related but don't        re-ranking, better
                      answer the question      embeddings
                      → low precision
```

**Key insight:** "These three failure modes each show up in a DIFFERENT metric: missing context → low recall; hallucination → low faithfulness; irrelevant retrieval → low precision. The metric tells you which failure you're in."

**Think-pair-share (2 min):** "Which of these three have YOU seen in your own labs? What did it look like — what was the symptom?" *(Students connect the taxonomy to their own experience before the advanced modes.)*

#### Micro-Cycle B — The Two Presentation Failure Modes (10 min)

```
FAILURE MODE          SYMPTOM                  FIX
─────────────────────────────────────────────────────────────
4. Lost-in-the-       Relevant chunk is in     Re-rank, reorder prompt
   middle             the middle of the        (most relevant first),
                      prompt, LLM pays more    reduce context size
                      attention to edges

5. Over-reliance      LLM copies context       Weaker "based ONLY on",
                      verbatim without         ask for synthesis,
                      synthesis → stiff        better prompt guidance
                      robotic answers
```

**Key insight:** "These two are subtler — they're about HOW the LLM uses the context, not whether it found it. Lost-in-the-middle is a prompt-ordering problem; over-reliance is a prompt-guidance problem. Both are fixed in the prompt, not the retrieval."

#### The Metric → Failure Mode → Fix Map

| Metric pattern | Failure mode | Fix |
|---|---|---|
| Low faithfulness | Hallucination (3) | Strengthen prompt, temp=0.0 |
| Low context precision | Irrelevant retrieval (5) | Hybrid search, re-rank, filter |
| Low context recall | Missing context (1) | Increase top_k, better chunking |
| High all + stiff answers | Over-reliance (4) | Weaken "based ONLY", ask for synthesis |
| Consistent, specific pattern | Lost-in-the-middle (2) | Reorder, re-rank |

---

### 7. Code Along 2 — Instrument for Failure Detection (2:30–2:55) [PART B — LAB]

**File:** `ex2_2_failure_detection.py`

```python
def detect_failure_modes(answer: str, question: str, chunks: list) -> list[str]:
    """
    Detect likely RAG failure modes from the answer + retrieval.

    Returns:
        List of failure-mode names (may be empty = healthy)
    """
    failures = []

    # 1. Missing context / low recall: few or no chunks retrieved
    if not chunks:
        failures.append("missing_context (no chunks retrieved)")
        return failures  # nothing else to check

    # 2. Hallucination: low faithfulness (claims not supported)
    claims = decompose_claims(answer)
    context_text = "\n\n".join(c.content for c in chunks)
    unsupported = sum(1 for c in claims if not judge_claim(c, context_text))
    if claims and unsupported / len(claims) > 0.5:
        failures.append(f"hallucination ({unsupported}/{len(claims)} claims unsupported)")

    # 3. Irrelevant retrieval: low context precision
    relevant = sum(1 for c in chunks if judge_chunk_relevant(question, c))
    if chunks and relevant / len(chunks) < 0.5:
        failures.append(f"irrelevant_retrieval ({relevant}/{len(chunks)} chunks relevant)")

    # 4. Over-reliance: answer copies context verbatim
    if answer.strip() in context_text:
        failures.append("over_reliance (answer copied verbatim from context)")

    return failures

# Logging wrapper
def log_evaluation(query, answer, chunks, failures, log_file="data/eval_log.jsonl"):
    """Log one evaluation run with failure-mode flags."""
    import json
    entry = {
        "query": query,
        "answer": answer,
        "num_chunks": len(chunks),
        "failures": failures,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

**Teaching tip:** The failure-detection function is a *diagnostic* — it flags suspected failure modes so students know where to look. It's not a replacement for the metrics; it's a fast triage before running full RAGAS.

---

### 8. Break (1:15–1:20)

---

### 9. Concept 3 — Improving RAG Based on Metrics (1:20–1:40)

#### The Improvement Loop

**Show on projector:**

```
  ┌──────────────────────────────────────────────────┐
  │           THE METRIC-DRIVEN LOOP                 │
  │                                                  │
  │  1. EVALUATE   → run RAGAS on test set           │
  │        │                                         │
  │  2. DIAGNOSE   → which metric is weakest?        │
  │        │        → which failure mode?            │
  │        ▼                                         │
  │  3. CHANGE     → ONE targeted change:            │
  │        │         chunk size / top_k / prompt /   │
  │        │         hybrid / re-rank / filter       │
  │        │                                         │
  │  4. RE-EVALUATE → run RAGAS again, compare       │
  │        │                                         │
  │        └──────────► back to 2                    │
  └──────────────────────────────────────────────────┘
```

**Worked Example: Before/After**

```
CONFIG A (baseline):          CONFIG B (after fix):
  chunk_size=150                chunk_size=300 (bigger chunks)
  top_k=3                       top_k=5
  vector search only            hybrid search

  faithfulness  0.88             faithfulness  0.90
  answer_relev  0.82             answer_relev  0.85
  context_prec  0.55  ← weak     context_prec  0.72  ← improved
  context_recall 0.40 ← weak     context_recall 0.65  ← improved

  Diagnosis: low precision + low recall = retrieval problem
  Fix: hybrid search + larger top_k + bigger chunks
  Result: precision +0.17, recall +0.25 — measurable improvement
```

**Key insight:** "Metrics tell you WHAT to fix — they don't fix it themselves. The loop is: measure → diagnose → change ONE thing → re-measure. If you change three things at once and the score improves, you don't know which change caused it."

---

### 10. Independent Lab — RAG Iteration (2:55–3:35) [PART B — LAB]

**File:** `ex2_3_rag_iteration.py`

Students: (1) run RAGAS on their pipeline, (2) identify weakest metric, (3) make ONE targeted change, (4) re-run and compare, (5) document. Full instructions in `lab_sheet.md`.

**Teaching tips:**
- The ONE-change rule is the critical discipline — students will want to change everything
- The documentation step matters: "what changed, why, what was the measurable effect" is exactly what the final project report needs
- Encourage students to pick a REAL weakness they saw in Session 1's lab, not a random change

---

### 11. Peer Review (3:35–3:50) [PART B — LAB]

**Pair up. Review checklist:**

> 1. Is the improvement measurable? (before/after numbers, not "feels better")
> 2. Was the change justified by the metric diagnosis? (not a random change)
> 3. Was only ONE thing changed? (if not, results are confounded)
> 4. Is the documentation clear? (what/why/result)

---

### 12. Knowledge Check 2 (1:40–1:45)

**MCQ:**

> After evaluation, faithfulness is 0.95 but context_precision is 0.30. What should you change?
>
> (a) Increase chunk size
> (b) Improve retrieval quality (hybrid search, re-ranking)
> (c) Change the LLM model

**Correct answer: (b) Improve retrieval quality**

**Explanation:** Faithfulness is already high (answers are grounded), so the prompt and generation are fine. Low context precision means the RETRIEVED chunks aren't relevant — the fix is on the retrieval side (hybrid search, re-ranking, metadata filtering), not chunk size (a) or model (c).

---

### 13. Wrap & Reflect (1:45–2:00)

**Instructor summary:**

> "This week you built the measurement layer for RAG:
> - **Session 1:** faithfulness, relevance, precision — the three-metric framework, LLM-as-judge pattern
> - **Session 2:** RAGAS automation, the five failure modes, the metric-driven improvement loop
>
> Next week: RAG is passive — it retrieves and generates. Agents are active — they decide what to do. And your VectorStore becomes their long-term memory. That's the next step in the journey."

**Reflection prompt (3 min):**

> "What was the most valuable metric for your system? Which failure mode surprised you most — have you seen it in your own labs?"

---

## Instructor Preparation Checklist

### Before Session 1
- [ ] Verify students have working RAG pipeline (Lab 4 `RAGSystem` from Week 4)
- [ ] Test `decompose_claims` + `judge_claim` with API key (LLM-as-judge calls add up — estimate cost)
- [ ] Prepare hook demo (confident-wrong answer)
- [ ] Prepare KC slides
- [ ] Warn students: evaluation makes MANY LLM calls (5 questions × 3 metrics × N claims) — budget for ~1-2 USD per student

### Before Session 2
- [ ] `pip install ragas datasets` — test on a 2-question subset first (dependency risk)
- [ ] Prepare failure-mode table visual
- [ ] Prepare before/after worked example
- [ ] Review Session 1 reflections

### Cost Note
- LLM-as-judge evaluation is token-hungry: a 5-question test set with 3 metrics can use 30–60 LLM calls. gpt-4o-mini keeps this under ~$0.10/student, but warn students not to run 10+ questions repeatedly.

---

## Common Student Mistakes & Interventions

| Mistake | When It Appears | Intervention |
|---|---|---|
| Judge uses temperature > 0 | Code Along 1, S1 | "The judge is a measuring instrument — temperature=0.0" |
| Not decomposing claims (judging whole answer) | Code Along 1, S1 | "Decompose first — one claim per check" |
| Judging chunks in a loop (many API calls) | Code Along 2, S1 | "Batch or limit the test set — cost control" |
| Confusing faithfulness with correctness | KC 1, S1 | "Faithfulness = grounded, not true" |
| Test set only easy questions | Independent Lab, S1 | "Include direct/synthesis/no-answer — like Week 4" |
| Changing 3 things at once in iteration | Independent Lab, S2 | "ONE change — or you can't attribute the improvement" |
| Not comparing before/after numbers | Independent Lab, S2 | "The numbers are the point — not 'feels better'" |
| Confusing precision with recall | KC 1/2, S2 | "Precision = retrieved set quality; recall = missed relevant chunks" |
| RAGAS install/import errors | Code Along 1, S2 | "Test on 2-question subset; check datasets version" |

---



