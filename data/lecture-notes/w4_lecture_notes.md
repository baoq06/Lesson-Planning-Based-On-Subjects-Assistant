# Week 4 — Lecture Notes (Instructor Guide)

## Building the RAG Pipeline

> **Course:** AI Programming with Python
> **Week:** 4 of 8 — 2 sessions/week × 4 hours (2h theory + 2h lab per session; theory total 4h/week)
> **Prerequisites:** Week 3 (Document/DocumentChunk @dataclass, DocumentStore, VectorStore, embeddings, cosine similarity), Week 2 (LLM API, streaming, error handling)
> **Learning Objectives:**
> - A4.1 Design a RAG prompt template that instructs the LLM to answer based only on provided context.
> - A4.2 Build a complete RAG pipeline: embed query → search → build prompt → call LLM → return answer.
> - A4.3 Implement a retriever that returns both the answer and the source chunks (citations).
> - A4.4 Handle edge cases: no relevant chunks found, empty query, overly long context.

---

## Session 1 — Retrieval and Prompt Design

> **Session format:** 4 hours total = **Part A — Theory (2h)** (this guide) + **Part B — Lab (2h)** (`lab_sheet.md`).

### Session Structure

| # | Segment | Duration | Gagné Event | Activity |
|---|---|---|---|---|
| — | **PART A — THEORY (2h)** | 0:00–2:00 | — | Segments 1–3, 5–6, 8–9, 12–13 |
| 1 | Hook & Problem | 0:00–0:10 | 1. Gain attention | RAG output with vs without citations |
| 2 | Objectives & Recall | 0:10–0:20 | 2–3. Objectives + Recall | Recap VectorStore.search; temperature |
| 3 | Concept 1: RAG Prompt Template | 0:20–0:45 | 4. Present content | Grounding prompt worked example |
| 5 | KC 1 | 0:45–0:50 | 7. Feedback | MCQ: why "based ONLY on context" |
| 6 | Concept 2: Complete RAG Pipeline | 0:50–1:15 | 4. Present content | Pipeline v1 (5 stages) |
| — | Break | 1:15–1:20 | — | — |
| 9 | Concept 3: Context Window Mgmt | 1:20–1:40 | 4. Present content | tiktoken, truncation, re-rank |
| 12 | KC 2 | 1:40–1:45 | 8. Assess | MCQ: LLM ignores RAG prompt |
| 13 | Wrap & Reflect | 1:45–2:00 | 9. Retention/Transfer | Preview Session 2 |
| — | **PART B — LAB (2h)** | 2:00–4:00 | — | Segments 4, 7, 10–11 — see `lab_sheet.md` |
| 4 | Code Along 1 (guided) | 2:00–2:30 | 5–6. Guidance + Practice | `build_rag_prompt()` (lab_sheet Ex 1.1) |
| 7 | Code Along 2 (guided) | 2:30–2:55 | 5–6. Guidance + Practice | `rag_pipeline()` + test (lab_sheet Ex 1.2) |
| 10 | Independent Lab | 2:55–3:35 | 6. Elicit performance | RAG v1 — 3 questions (lab_sheet Ex 1.3 — 40 min) |
| 11 | Peer Review | 3:35–3:50 | 7. Feedback | Pair code review (lab_sheet Ex 1.4 — 15 min) |
| — | Buffer / questions | 3:50–4:00 | — | — |

---

### Before Class: Setup Checklist

- [ ] Verify students have their `VectorStore` from Lab 3 working (or provide a reference solution)
- [ ] Create `data/company_policy.txt` — a multi-section policy document (see lab_sheet.md)
- [ ] `pip install tiktoken rank_bm25 sentence-transformers` on all machines (rank_bm25 and sentence-transformers used in Session 2)
- [ ] Test all Code Along scripts with your API key
- [ ] Prepare the 2D hook demo (RAG answer with vs without citations)

**Prerequisite gate:** If more than a third of students don't have a working `VectorStore`, run a 10-minute "bring-up" sprint at the start: everyone loads the reference `VectorStore` code (provided in `lab_sheet.md` Week 3 Lab 3 solution) and verifies `search()` works on 3 documents before proceeding.

---

### 1. Hook & Problem (0:00–0:10)

**Instructor script:**

> "Here are two answers to the same question — 'What is the remote work policy?' — from the same LLM. One is the raw answer. The other has citations."
>
> **Without citations:** "The remote work policy allows employees to work from home up to three days per week, subject to manager approval."
>
> **With citations:** "The remote work policy allows employees to work from home up to three days per week [Source: policy_manual, section 4.2], subject to manager approval [Source: policy_manual, section 4.3]."
>
> "Which one can you *trust*? The first answer sounds plausible — but so does a hallucination. The second answer lets you verify: open section 4.2, check the exact wording, confirm the policy.
>
> "This is the difference between RAG that works and RAG that *appears* to work. Today, you build the pipeline that produces the verifiable answer.
>
> "By the end of this session: you take a user question, retrieve the relevant chunks from your VectorStore (Week 3), build a prompt that forces the LLM to use ONLY those chunks, and get back an answer **with the sources attached**."

**Cognitive load note:** The hook shows the *destination* (verifiable RAG output) before the *mechanics*. Students know where they're going — this reduces extraneous load during the pipeline construction.

---

### 2. Objectives & Recall (0:10–0:20)

**State objectives:**

> "By the end of this session, you will be able to:
> 1. Write a RAG prompt template that keeps the LLM grounded in provided context.
> 2. Build the complete 5-stage RAG pipeline: embed → search → build prompt → call LLM → return answer + sources.
> 3. Handle the three edge cases: no relevant chunks, empty query, context too long."

**Recall sprint (pair activity, 2 min):**

> "In 30 seconds, explain to your partner:
> 1. What does `store.search(query, top_k=3)` return? *(list of (chunk, score) tuples)*
> 2. What is temperature? *(randomness of token selection — 0.0 = deterministic)*
> 3. How did you embed a query in Lab 3? *(embed_text(query))*

**Why this helps:** The RAG pipeline reuses all three. Activating them now reduces the load of holding them all at once.

---

### 3. Concept 1 — The RAG Prompt Template (0:20–0:45)

#### Worked Example: The Grounding Prompt

**Show on the projector:**

```
┌──────────────────────────────────────────────────────────────┐
│  SYSTEM PROMPT (the "grounding contract")                    │
│  You are a helpful assistant. Answer the user's question     │
│  based ONLY on the following context.                        │
│                                                              │
│  If the context does not contain enough information, say     │
│  "I cannot find enough information in the provided           │
│  documents."                                                 │
│                                                              │
│  Context:                                                    │
│  [chunk 1: section 4.2 — Remote work policy]                 │
│  [chunk 2: section 4.3 — Approval process]                   │
│                                                              │
│  Question: What is the remote work policy?                   │
│                                                              │
│  ──────────────────────────────────────────────────────────  │
│  ↑ ONLY this material is allowed to inform the answer        │
│  ↑ Anything outside it = hallucination                       │
└──────────────────────────────────────────────────────────────┘
```

**Key insight:** "The prompt is the ONLY thing keeping the LLM grounded. Every word matters. 'Based ONLY on the following context' is not polite phrasing — it's the instruction that prevents the LLM from answering from its training data."

#### The Three Instructions of a RAG Prompt

| Instruction | Purpose | Failure mode without it |
|---|---|---|
| "Answer based ONLY on the context" | Prevents using training data | Hallucination — LLM answers from memory |
| "If context is insufficient, say 'I cannot find...'" | Handles the no-match case | Fabricated answer — LLM invents to avoid saying "I don't know" |
| Context + Question structure | Separates data from task | Confusion — LLM treats context as part of the question |

---

### 4. Code Along 1 — build_rag_prompt (2:00–2:30) [PART B — LAB]

**File:** `ex1_1_build_rag_prompt.py`

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== Step 1: The prompt template =====

RAG_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the following context.

If the context does not contain enough information, say: "I cannot find enough information in the provided documents."

Context:
{context}

Question: {question}"""

# ===== Step 2: Format chunks into the context section =====

def format_chunk(chunk, index):
    """Format a single chunk with its source metadata for the prompt."""
    source = chunk.metadata.get("source", "unknown")
    title = chunk.metadata.get("title", "document")
    return f"[{index+1}] (Source: {title}, {source})\n{chunk.content}"

def build_rag_prompt(question: str, chunks: list, top_k: int = 3) -> list[dict]:
    """
    Build the message list for a RAG request.

    Args:
        question: The user's question
        chunks: List of DocumentChunk objects from VectorStore.search()
        top_k: Max chunks to include (defensive — search should already limit)

    Returns:
        messages: list of {"role", "content"} dicts ready for the API
    """
    # Format the context section (numbered, with sources)
    context_text = "\n\n".join(
        format_chunk(chunk, i) for i, chunk in enumerate(chunks[:top_k])
    )

    system_prompt = RAG_SYSTEM_PROMPT.format(
        context=context_text,
        question=question
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}  # Question also sent explicitly
    ]

# ===== Step 3: Test — show the rendered prompt =====

# Use a fake chunk to demonstrate the format (no API call yet)
class FakeChunk:
    def __init__(self, content, source, title):
        self.content = content
        self.metadata = {"source": source, "title": title}

chunks = [
    FakeChunk(
        "Employees may work remotely up to three days per week.",
        "company_policy.txt", "Remote Work Policy"
    ),
    FakeChunk(
        "Remote work requires prior approval from the line manager.",
        "company_policy.txt", "Remote Work Policy"
    ),
]

messages = build_rag_prompt("What is the remote work policy?", chunks)

print("=== SYSTEM PROMPT (rendered) ===")
print(messages[0]["content"])
print("\n=== USER MESSAGE ===")
print(messages[1]["content"])
```

**Checkpoint — students should see:**

```
=== SYSTEM PROMPT (rendered) ===
You are a helpful assistant. Answer the user's question based ONLY on the following context.
...
Context:
[1] (Source: Remote Work Policy, company_policy.txt)
Employees may work remotely up to three days per week.

[2] (Source: Remote Work Policy, company_policy.txt)
Remote work requires prior approval from the line manager.

Question: What is the remote work policy?

=== USER MESSAGE ===
What is the remote work policy?
```

**Teaching tip:** Point out that the question appears TWICE (once in the system prompt, once as the user message). Explain: the system prompt carries the full grounding contract + context; the user message is the raw question. This redundancy is intentional — the system prompt is the "workspace," the user message is the "task."

---

### 5. Knowledge Check 1 (0:45–0:50)

**MCQ:**

> Why does the RAG prompt say "Answer based ONLY on the following context"?
>
> (a) To save tokens
> (b) To prevent hallucination from the LLM's training data
> (c) To make the answer shorter

**Correct answer: (b) To prevent hallucination**

**Explanation:** The instruction tells the LLM to ignore its internal knowledge and use only the retrieved chunks. Without it, the LLM will answer from training data — which doesn't include the company's private policy — producing a plausible-sounding but wrong answer. (a) and (c) are incidental, not the purpose.

---

### 6. Concept 2 — The Complete RAG Pipeline (0:50–1:15)

#### Worked Example: The 5-Stage Pipeline

**Show this diagram:**

```
User question: "What is the remote work policy?"
     │
     ▼
┌─────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. EMBED    │──►│ 2. SEARCH    │──►│ 3. BUILD     │
│ query →     │   │ VectorStore  │   │ prompt with  │
│ vector      │   │ top_k chunks │   │ context      │
└─────────────┘   └──────────────┘   └──────────────┘
                                          │
     ┌────────────────────────────────────┘
     ▼
┌─────────────┐   ┌──────────────┐
│ 5. RETURN   │◄──│ 4. CALL LLM  │
│ answer +    │   │ with prompt  │
│ sources     │   │              │
└─────────────┘   └──────────────┘
```

**Key insight:** "The RAG pipeline is a sequence of transforms — each stage takes the output of the previous one. This is why we built `embed_text()` and `search()` in Week 3: the pipeline is just *wiring them together* with a prompt in the middle."

**Critical design decision — returning sources:**

> "The pipeline returns `(answer, sources)` — not just the answer. Why? Because 'sources' is what makes RAG trustworthy. The user (or another system) can verify the answer against the chunks. This is the difference between RAG and a plain LLM call."

**Before/after — how the answer shape changed since Week 2:**

```
WEEK 2 (plain LLM call):          WEEK 4 (RAG pipeline):
answer = response...content       answer, sources = rag_pipeline(query, store)
print(answer)                     print(answer)        # same as Week 2
                                  for chunk, score in sources:  # NEW
                                      print(f"[{score:.3f}] {chunk.content[:50]}")
```

**Key insight:** "The RAG pipeline is NOT a different thing from the Week 2 LLM call — it's the Week 2 call PLUS retrieval PLUS sources. Think of it as a superset: everything you knew still applies, we've just added the retrieval stage before it and the sources after it."

---

### 7. Code Along 2 — rag_pipeline (2:30–2:55) [PART B — LAB]

**File:** `ex1_2_rag_pipeline.py`

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== Reuse from Week 3 (import or paste) =====
# VectorStore, Document, DocumentChunk, embed_text, cosine_similarity
# (See lab_sheet.md Week 3 for the full implementation)

# ===== Reuse from Code Along 1 =====
RAG_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the following context.

If the context does not contain enough information, say: "I cannot find enough information in the provided documents."

Context:
{context}

Question: {question}"""

def build_rag_prompt(question: str, chunks: list) -> list[dict]:
    context_text = "\n\n".join(
        f"[{i+1}] (Source: {c.metadata.get('title', 'doc')}) {c.content}"
        for i, c in enumerate(chunks)
    )
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT.format(context=context_text, question=question)},
        {"role": "user", "content": question}
    ]

# ===== The RAG pipeline =====

def rag_pipeline(query: str, store, top_k: int = 3) -> tuple[str, list]:
    """
    Complete RAG pipeline: embed → search → build prompt → call LLM → answer + sources.

    Args:
        query: The user's question
        store: VectorStore (from Week 3)
        top_k: Number of chunks to retrieve

    Returns:
        (answer, sources) where sources is the list of (chunk, score) tuples
    """
    # Step 1+2: Embed query and search the store
    results = store.search(query, top_k=top_k)

    # Edge case: no relevant chunks found
    if not results:
        return ("I cannot find enough information in the provided documents.", [])

    # Extract chunks (drop scores for the prompt)
    chunks = [chunk for chunk, score in results]

    # Step 3: Build the prompt
    messages = build_rag_prompt(query, chunks)

    # Step 4: Call the LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0  # Deterministic — we want consistent grounded answers
    )
    answer = response.choices[0].message.content

    # Step 5: Return answer + sources (with scores for transparency)
    return answer, results

# ===== Test =====
if __name__ == "__main__":
    # Build a small test store
    store = VectorStore()
    # ... load documents from Week 3 sample data ...

    test_questions = [
        "What is the remote work policy?",
        "How many vacation days do employees get?",
    ]
    for q in test_questions:
        print(f"\n=== Question: {q} ===")
        answer, sources = rag_pipeline(q, store)
        print(f"Answer: {answer}")
        print(f"Sources ({len(sources)}):")
        for chunk, score in sources:
            print(f"  [{score:.3f}] {chunk.content[:60]}...")
```

**Common mistakes to watch for:**

| Mistake | Correction |
|---|---|
| Returning only the answer, dropping sources | Always return `(answer, sources)` |
| `temperature` too high → inconsistent answers | Use `temperature=0.0` for grounded Q&A |
| Forgetting the no-results edge case | `if not results: return "I cannot find..."` |
| Sending scores to the LLM in the context | Scores are for *you*; the prompt should only have content |
| Using a different embedding model for query vs chunks | Same model for both — else dimensions mismatch |

---

### 8. Break (1:15–1:20)

---

### 9. Concept 3 — Context Window Management (1:20–1:40)

#### The Problem

> "Your retrieval returns 10 chunks. The context window is 8,000 tokens. If 10 chunks = 12,000 tokens, the API call FAILS or gets truncated. You must manage how much context you send."

#### Worked Example: Counting Tokens with tiktoken

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in a text for a specific model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def estimate_chunks_tokens(chunks: list) -> int:
    """Estimate total tokens for a list of chunks."""
    return sum(count_tokens(c.content) for c in chunks)

# Example: check if our context fits
chunks = store.search("remote work policy", top_k=10)
chunks = [c for c, s in chunks]
total_tokens = estimate_chunks_tokens(chunks)
print(f"10 chunks = {total_tokens} tokens (limit: 8000)")
```

#### Worked Example: Truncation Strategy

```python
MAX_CONTEXT_TOKENS = 4000  # Leave room for the answer + prompt overhead

def fit_chunks_to_window(chunks: list, max_tokens: int = MAX_CONTEXT_TOKENS) -> list:
    """
    Keep adding chunks (in relevance order) until we hit the token budget.
    This is a greedy approach: most-relevant first.
    """
    selected = []
    used = 0
    for chunk in chunks:
        tokens = count_tokens(chunk.content)
        if used + tokens > max_tokens:
            break  # Stop — we've hit the budget
        selected.append(chunk)
        used += tokens
    print(f"Selected {len(selected)}/{len(chunks)} chunks ({used}/{max_tokens} tokens)")
    return selected
```

**Key insight — the ordering matters:**

> "Chunks come back from `search()` sorted by relevance (highest score first). The greedy truncation takes the most-relevant chunks and drops the least-relevant ones when the window fills. This is 'context budgeting' — the same skill as budgeting memory or time."

**Edge case table:**

| Edge case | Detection | Response |
|---|---|---|
| No chunks found | `search()` returns `[]` | Return "I cannot find..." immediately |
| Empty query | `not query.strip()` | Return a message asking for a question |
| Context too long | token count > budget | Greedy truncation (above) |

---

### 10. Independent Lab — RAG v1 (2:55–3:35) [PART B — LAB]

**File:** `ex1_3_rag_v1.py`

Students build the full pipeline and test 3 question types. Full instructions in `lab_sheet.md`.

**Teaching tips:**
- Watch for students who skip the no-results edge case
- Encourage them to print the *rendered prompt* at least once — understanding what the LLM sees is the key insight
- Question types to test: (1) directly answered, (2) partially answered, (3) not answered — observing the LLM's "I cannot find..." behaviour is the learning moment

---

### 11. Peer Review (3:35–3:50) [PART B — LAB]

**Pair up. Review checklist:**

> 1. Does the pipeline return sources, not just the answer?
> 2. Does the LLM correctly say "I cannot find..." when context is insufficient?
> 3. Is the prompt template grounded ("based ONLY on context")?
> 4. Are edge cases handled (no results, empty query, long context)?
> 5. Is `temperature=0.0` used for consistent answers?

---

### 12. Knowledge Check 2 (1:40–1:45)

**MCQ:**

> The LLM ignores the RAG prompt and answers from its training data. What's the most likely fix?
>
> (a) Increase temperature
> (b) Strengthen the prompt instruction
> (c) Use a different embedding model

**Correct answer: (b) Strengthen the prompt instruction**

**Explanation:** If the LLM is hallucinating from training data, the retrieval and embeddings are fine — the *prompt* is too weak. Strengthening "based ONLY on the following context" (or adding explicit "do not use your own knowledge") re-asserts the grounding contract. (a) would make it worse (more randomness); (c) is irrelevant to generation.

---

### 13. Wrap & Reflect (1:45–2:00)

**Instructor summary:**

> "Today you built the core of RAG:
> - A **grounding prompt** that keeps the LLM honest.
> - The **5-stage pipeline**: embed → search → build → call → return.
> - **Sources attached** to the answer — the trust mechanism.
> - **Context budgeting** so you never overflow the window.
>
> Session 2: your pipeline works, but retrieval isn't optimal yet. We'll add hybrid search (BM25 + vector), re-ranking, and metadata filtering."

**Reflection prompt (3 min):**

> "What surprised you about how the LLM behaves when context is missing or noisy? Which edge case was hardest to handle?"

---

## Session 2 — RAG Optimisation & Integration

> **Session format:** 4 hours total = **Part A — Theory (2h)** (this guide) + **Part B — Lab (2h)** (`lab_sheet.md`).

### Session Structure

| # | Segment | Duration | Gagné Event | Activity |
|---|---|---|---|---|
| — | **PART A — THEORY (2h)** | 0:00–2:00 | — | Segments 1–3, 5–6, 8–9, 12 |
| 1 | Hook | 0:00–0:10 | 1. Gain attention | "Our RAG works but misses good chunks" |
| 2 | Objectives & Recall | 0:10–0:20 | 2–3. Objectives + Recall | Recap pipeline; "What is precision?" |
| 3 | Concept 1: Hybrid Search | 0:20–0:45 | 4. Present content | BM25 + vector worked example |
| 5 | KC 1 | 0:45–0:50 | 7. Feedback | MCQ: when BM25 beats vector |
| 6 | Concept 2: Re-Ranking | 0:50–1:15 | 4. Present content | Two-stage retrieval worked example |
| — | Break | 1:15–1:20 | — | — |
| 9 | Concept 3: Metadata Filtering | 1:20–1:35 | 4. Present content | Pre-filter by source/date (compressed, 15 min — CLT) |
| 12 | KC 2 + Mid-Course Announcement | 1:35–1:50 | 8. Assess | MCQ: purpose of re-ranking + practical prep |
| — | Wrap & Reflect | 1:50–2:00 | 9. Retention/Transfer | Preview Mid-Course Practical |
| — | **PART B — LAB (2h)** | 2:00–4:00 | — | Segments 4, 7, 10–11 — see `lab_sheet.md` |
| 4 | Code Along 1 (guided) | 2:00–2:30 | 5–6. Guidance + Practice | Combine scores: 0.3*BM25 + 0.7*vector (lab_sheet Ex 2.1) |
| 7 | Code Along 2 (guided) | 2:30–2:55 | 5–6. Guidance + Practice | CrossEncoder re-rank (lab_sheet Ex 2.2) |
| 10 | Independent Lab | 2:55–3:35 | 6. Elicit performance | Enhanced RAG: compare v1 vs enhanced (lab_sheet Ex 2.3 — 40 min) |
| 11 | Peer Review | 3:35–3:50 | 7. Feedback | Pair code review (lab_sheet Ex 2.4 — 15 min) |
| — | Buffer / questions | 3:50–4:00 | — | — |

---

### 1. Hook (0:00–0:10)

**Instructor script:**

> "Your RAG pipeline from Session 1 works. But try this query: 'What is the WFH policy?' The document says 'remote work policy' — the vector search finds it (synonyms), but a keyword search would have missed it. Now try: 'What is the reimbursement rate for travel?' — a rare technical term. The vector search might miss the exact phrase 'reimbursement rate' if it's buried in noise.
>
> "Vector search is great at *meaning*. Keyword search is great at *exact terms*. Today we combine both — plus two more optimisations:
> 1. **Hybrid search** (BM25 keywords + vector) — best of both worlds
> 2. **Re-ranking** (a careful second pass over fast first results)
> 3. **Metadata filtering** (narrow the search space before searching)"

**Cognitive load note:** The hook frames the three optimisations as *solutions to specific failures* (misses exact terms, misses buried phrases, searches too wide). Students see each technique's *purpose* before its *mechanics* — Merrill's problem-centered principle.

---

### 2. Objectives & Recall (0:10–0:20)

**State objectives:**

> "By the end of this session, you will be able to:
> 1. Combine BM25 keyword scores with vector scores (hybrid search).
> 2. Re-rank top results with a cross-encoder for better precision.
> 3. Pre-filter chunks by metadata before searching.
> 4. Compare your enhanced pipeline against the v1 baseline."

**Recall sprint (pair, 2 min):**

> "In 30 seconds, explain to your partner:
> 1. What does `store.search()` return? *(chunk, score) tuples sorted descending*
> 2. What's the trade-off of small vs large chunks? *(precision vs context)*
> 3. What was the hardest part of Session 1's lab?"

---

### 3. Concept 1 — Hybrid Search (0:20–0:45)

#### The Problem: Two Search Worlds

```
KEYWORD (BM25):                    VECTOR (Week 3):
  "reimbursement rate"               "what can I claim for business travel?"
   → exact term match                → semantic match
   → misses synonyms, typos          → misses rare exact terms
   → good for rare/technical terms   → good for paraphrase, meaning
```

#### Worked Example: The Score Fusion Formula

**Show on the projector:**

```
hybrid_score = (w_bm25 × normalized_bm25) + (w_vector × cosine_similarity)

where:
  w_bm25 + w_vector = 1.0           (weights sum to 1)
  normalized_bm25 ∈ [0, 1]          (BM25 scores are unbounded — must normalise)
  cosine_similarity ∈ [-1, 1]       (already bounded)

Default weights: w_bm25 = 0.3, w_vector = 0.7
(reason: vector is usually the better primary signal; BM25 catches the gaps)
```

> **Persistent reference card — keep visible for the whole session:**
>
> ```
> ┌─────────────────────────────────────────────────────────────┐
> │  HYBRID SEARCH — score fusion pipeline                     │
> │                                                            │
> │  BM25 scores (UNBOUNDED: 5.2, 13.7, ...)                   │
> │    │                                                       │
> │    ▼ NORMALISE (min-max → [0,1])  ← MUST DO!               │
> │  bm25_norm ∈ [0,1]                                         │
> │    │                                                       │
> │  cosine_similarity (bounded: [-1,1])  ← already OK         │
> │    │                                                       │
> │    ▼                                                       │
> │  hybrid = 0.3×bm25_norm + 0.7×cosine                      │
> │  (weights must sum to 1.0)                                 │
> └─────────────────────────────────────────────────────────────┘
> ```

**Key insight:** "BM25 scores are unbounded (they can be 5.2, 13.7, anything). Cosine similarity is bounded [-1, 1]. You MUST normalise BM25 before combining — otherwise BM25 dominates the sum. This is the single most common bug in hybrid search: forgetting the normalisation step."

#### Normalisation approach

```python
def normalise_scores(scores: list[float]) -> list[float]:
    """Normalise scores to [0, 1] using min-max scaling."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)  # All equal → all 1.0
    return [(s - min_s) / (max_s - min_s) for s in scores]
```

---

### 4. Code Along 1 — Hybrid Search (2:00–2:30) [PART B — LAB]

**File:** `ex2_1_hybrid_search.py`

```python
from rank_bm25 import BM25Okapi
import numpy as np

# ===== Step 1: Build a BM25 index over the chunks =====

def build_bm25_index(chunks: list) -> BM25Okapi:
    """Build a BM25 index from chunk content (tokenised by whitespace)."""
    tokenised = [chunk.content.lower().split() for chunk in chunks]
    return BM25Okapi(tokenised)

# ===== Step 2: Hybrid search =====

def hybrid_search(
    query: str,
    chunks: list,
    bm25_index: BM25Okapi,
    embeddings: list[list[float]],
    top_k: int = 3,
    w_bm25: float = 0.3,
    w_vector: float = 0.7
) -> list[tuple]:
    """
    Hybrid search: combine BM25 (keyword) and vector (semantic) scores.

    Returns: list of (chunk_index, hybrid_score) tuples sorted descending.
    """
    # BM25 scores (raw, unbounded)
    tokenised_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenised_query)

    # Vector scores (cosine similarity, bounded)
    query_vec = embed_text(query)
    vector_scores = [cosine_similarity(query_vec, e) for e in embeddings]

    # Normalise BM25 to [0, 1]
    bm25_norm = normalise_scores(list(bm25_scores))

    # Fuse
    hybrid = [
        w_bm25 * b + w_vector * v
        for b, v in zip(bm25_norm, vector_scores)
    ]

    # Sort by hybrid score descending, return top_k
    ranked = sorted(
        enumerate(hybrid), key=lambda x: x[1], reverse=True
    )[:top_k]

    return ranked  # list of (chunk_index, score)
```

**Checkpoint:** Test with a query containing a rare term (e.g., "reimbursement rate") and a paraphrase query (e.g., "what can I claim for travel"). Compare which search finds each best.

---

### 5. Knowledge Check 1 (0:45–0:50)

**MCQ:**

> When would BM25 outperform vector search?
>
> (a) When the query contains a rare technical term
> (b) When the query is a long paragraph
> (c) Vector search always outperforms BM25

**Correct answer: (a) Rare technical term**

**Explanation:** BM25 matches exact terms. A rare technical term like "reimbursement rate" or a model name "BERT-2024" appears exactly in the document — BM25 nails it. Vector search may miss it if the surrounding chunk is noisy or the term's embedding sits far from the query's paraphrase. Long paragraphs (b) favour vector search; (c) is false — neither always wins.

---

### 6. Concept 2 — Re-Ranking (0:50–1:15)

#### Mini-Recall Sprint (2 min, pair activity)

> Before we add re-ranking, consolidate what you just built. **In 30 seconds, tell your partner:**
>
> 1. What does normalisation do in hybrid search, and why is it needed? *(BM25 is unbounded; min-max scales it to [0,1] so it doesn't dominate cosine)*
> 2. What does `hybrid_search()` return? *(list of (chunk_index, hybrid_score) sorted descending)*
> 3. Which query type does BM25 handle better than vector? *(rare/technical exact terms)*
>
> **Why this helps:** Re-ranking builds on top of the hybrid search you just learned. Activating it now gives a consolidation beat between the two techniques and reduces the load of layering re-ranking on top.

#### The Problem: Fast Search is Rough, Careful Search is Slow

```
Stage 1 (FAST, cheap):  retrieve top-10 with hybrid search
                        → good recall, rough precision
Stage 2 (SLOW, careful): re-rank top-10 with a cross-encoder
                         → precise ordering, only 10 calls
```

**Key insight:** "The cross-encoder reads BOTH the query AND the chunk together (cross-attention) — it's much more accurate than comparing two embeddings (bi-encoder), but slower. The trick: only run it on the top-10, not the whole corpus."

#### Worked Example: CrossEncoder

```python
from sentence_transformers import CrossEncoder

# Load the re-ranker model (once, at startup)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def re_rank(query: str, candidates: list[tuple], top_k: int = 3) -> list[tuple]:
    """
    Re-rank candidates with a cross-encoder.

    Args:
        query: The original query
        candidates: list of (chunk, score) from fast search
        top_k: Final number to return

    Returns:
        Re-ranked list of (chunk, new_score)
    """
    # Build (query, chunk_content) pairs for the cross-encoder
    pairs = [(query, chunk.content) for chunk, _ in candidates]

    # Cross-encoder returns relevance scores
    scores = reranker.predict(pairs)

    # Pair chunks with new scores and sort
    reranked = sorted(
        zip([c for c, _ in candidates], scores),
        key=lambda x: x[1], reverse=True
    )

    return reranked[:top_k]
```

**Trade-off table (show on projector):**

| | Stage 1 (hybrid) | Stage 2 (cross-encoder) |
|---|---|---|
| Speed | Fast (embedding math) | Slow (model inference) |
| Accuracy | Good | Better (cross-attention) |
| Cost | Cheap (no model call) | One model call per pair |
| Use | Whole corpus | Only top-10 |

---

### 7. Code Along 2 — Add Re-Ranking (2:30–2:55) [PART B — LAB]

**File:** `ex2_2_rerank.py`

> **⚠️ First run downloads ~80MB** (the cross-encoder model). Do this once at the start of the session or during the break — not mid-exercise. If the network is slow, ask the instructor for the pre-downloaded model.

Extend the pipeline:

```python
def enhanced_retrieve(query: str, store, top_k_fast=10, top_k_final=3) -> list[tuple]:
    """
    Two-stage retrieval: fast hybrid search, then cross-encoder re-rank.
    """
    # Stage 1: fast — get top-10 with hybrid search
    fast_results = hybrid_search_full(query, store, top_k=top_k_fast)

    # Stage 2: careful — re-rank top-10, keep top-3
    final = re_rank(query, fast_results, top_k=top_k_final)
    return final
```

**Teaching tip:** Compare the SAME query with and without re-ranking. The top-3 should be *reordered* (often the #1 after re-rank differs from the #1 after fast search). This visible difference is the learning moment.

---

### 8. Break (1:15–1:20)

---

### 9. Concept 3 — Metadata Filtering (1:20–1:35)

#### The Problem: Searching the Whole Corpus When You Only Need a Slice

> "The user asks about 'the leave policy'. Your corpus has policy_manual, HR_handbook, and product_docs. The leave policy is in HR_handbook. Without filtering, you search ALL chunks — the policy chunks compete with product chunks. With filtering, you narrow to HR chunks first."

#### Worked Example: Pre-Filter by Metadata

```python
def search_with_filter(
    query: str,
    store,
    metadata_filter: dict,
    top_k: int = 3
) -> list[tuple]:
    """
    Search chunks, but only those matching the metadata filter.

    Args:
        query: The search query
        store: VectorStore
        metadata_filter: e.g. {"source": "hr_handbook"} or {"year": "2024"}
        top_k: Number of results

    Returns:
        List of (chunk, score) from matching chunks only
    """
    # Pre-filter: keep chunks whose metadata matches ALL filter conditions
    candidate_chunks = [
        chunk for chunk in store.chunks.values()
        if all(chunk.metadata.get(k) == v for k, v in metadata_filter.items())
    ]

    if not candidate_chunks:
        return []

    # Embed query once
    query_vec = embed_text(query)

    # Score only the filtered chunks
    scored = []
    for chunk in candidate_chunks:
        if chunk.embedding is None:
            continue
        score = cosine_similarity(query_vec, chunk.embedding)
        scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

**Key insight:** "Metadata filtering is 'search space narrowing' — you make the search easier by removing chunks you already know are irrelevant. It's cheap, predictable, and often the single biggest retrieval improvement for structured corpora."

**Filter types:**

| Filter | Example | Use case |
|---|---|---|
| Source | `{"source": "hr_handbook"}` | Multi-document corpora |
| Date | `{"year": "2024"}` | Policy versions, news |
| Category | `{"type": "faq"}` | Mixed corpora |
| Doc ID | `{"doc_id": "DOC003"}` | Scoped queries |

---

### 10. Independent Lab — Enhanced RAG (2:55–3:35) [PART B — LAB]

**File:** `ex2_3_enhanced_rag.py`

Students extend the v1 pipeline with metadata filtering, hybrid search, and (optional) re-ranking, then compare against the baseline. Full instructions in `lab_sheet.md`.

**Teaching tips:**
- The comparison is the point: run the same 3 questions through v1 and enhanced, record answers + scores side by side
- Metadata filtering is easiest to see improvement — make sure students test a query that should be filtered
- Re-ranking is optional (model download ~80MB); provide it as bonus

---

### 11. Peer Review (3:35–3:50) [PART B — LAB]

**Pair up. Review checklist:**

> 1. Does the enhanced pipeline produce better results than v1? (same queries, side-by-side)
> 2. Are the trade-offs documented? (which optimisation helped most, why)
> 3. Is BM25 normalised before fusion? (unbounded scores would dominate)
> 4. Does metadata filtering correctly narrow the search space?
> 5. Is the code readable?

---

### 12. Knowledge Check 2 + Mid-Course Practical Announcement (1:35–1:50)

**MCQ:**

> What is the purpose of re-ranking in RAG?
>
> (a) To reduce the number of API calls
> (b) To improve precision by applying a more accurate (but slower) model
> (c) To increase recall

**Correct answer: (b) Improve precision**

**Explanation:** Re-ranking takes the top-N from a fast, rough search and re-orders them with a more accurate (but slower) model — improving *precision* (the top results are more likely to be relevant). It doesn't reduce API calls (a) or increase recall (b) — recall is set by the first-stage search width.

**Mid-Course Practical announcement:**

> "**Next session: Mid-Course Practical — 15% of your grade.** You'll build a RAG pipeline from scratch on a provided dataset: chunk → embed → store → search → prompt → answer → evaluate groundedness. 2 hours, open-internet, open-notes. Everything you need you've built this week — the practical is a test of *wiring it together*, not new material.
>
> **Here's your concrete check:** the practical expects the same shape as your **Lab 4 `RAGSystem`** — `answer()` returning `{query, answer, sources, num_chunks, context_tokens}`. If you can complete Lab 4 this week, you can pass the practical. That's your preparation checklist: finish Lab 4."

---

## Mid-Course Practical — Instructor Guide

### Practical Overview

| Item | Detail |
|---|---|
| **Duration** | 2 hours (dedicated session) |
| **Weight** | 15% |
| **Format** | In-lab, open-internet, open-notes, individual |
| **Dataset** | 5 short documents (2–3 pages each), provided |
| **Tasks** | Chunk → VectorStore → RAG pipeline → answer 3 questions → assess groundedness |
| **Prepared** | Instructor creates the dataset + question set beforehand |

### The 3 Questions Design

The three questions must cover the failure-mode spectrum:

| Question | Type | Tests |
|---|---|---|
| Q1 | Directly answered by one document | Basic pipeline works |
| Q2 | Spans two documents (needs synthesis) | Retrieval recall + prompt grounding |
| Q3 | NOT answered by any document | The "I cannot find..." edge case |

### Marking Criteria (aligned to the grading rubric)

| Criterion | Weight | Check |
|---|---|---|
| Data modelling | 15% | @dataclass used, metadata present |
| RAG pipeline | 30% | 5 stages complete, sources returned |
| Groundedness assessment | 20% | Student correctly assesses whether Q3's answer is grounded |
| Edge cases | 15% | No-results handled, context budget respected |
| Code quality | 20% | Readable, error handling, no hard-coded paths |

### Sample Dataset (instructor creates)

Provide 5 documents covering a fictional company (e.g., "Acme Corp"):
1. `acme_leave_policy.txt` — leave, vacation, sick days
2. `acme_remote_policy.txt` — remote work, WFH
3. `acme_expense_policy.txt` — reimbursement, travel claims
4. `acme_benefits.txt` — insurance, wellness
5. `acme_handbook.txt` — culture, hours, dress code

**Q1** (direct): "How many vacation days does an employee get in their first year?" → acme_leave_policy
**Q2** (synthesis): "Can I work from home and claim home-office expenses?" → acme_remote + acme_expense
**Q3** (no answer): "What is the stock option vesting schedule?" → nowhere in the documents

### Anti-Cheating Note

Open-internet/open-notes means students may consult docs and code — but the pipeline must be *theirs*. Flag if two students' pipelines are byte-identical. The assessment targets "can you wire it together," not "can you recall code."

---

## Instructor Preparation Checklist

### Before Session 1
- [ ] Verify Lab 3 VectorStore solutions work (or provide reference solution)
- [ ] Create `data/company_policy.txt` (multi-section)
- [ ] Test Code Along 1 and 2 with API key
- [ ] `pip install tiktoken`
- [ ] Prepare hook demo (RAG with/without citations)
- [ ] Prepare KC slides

### Before Session 2
- [ ] `pip install rank_bm25 sentence-transformers`
- [ ] Test hybrid search (rare-term + paraphrase queries)
- [ ] Pre-download the cross-encoder model (optional; ~80MB) to avoid classroom network slowdown
- [ ] Test re-ranking comparison
- [ ] Prepare Mid-Course Practical dataset + 3 questions + marking sheet
- [ ] Review Session 1 reflections

### Mid-Course Practical
- [ ] Dataset files in `data/midcourse/` on all machines
- [ ] 3 questions provided (not shown beforehand)
- [ ] Marking sheet printed
- [ ] Timing enforced (2 hours)

---

## Common Student Mistakes & Interventions

| Mistake | When It Appears | Intervention |
|---|---|---|
| Returning only the answer, not sources | Code Along 2, S1 | "The answer is untrustworthy without sources — return `(answer, sources)`" |
| Using `temperature=1.0` for RAG | Code Along 2, S1 | "Grounded Q&A needs determinism — use 0.0" |
| Sending scores to the LLM in context | Code Along 1, S1 | "Scores are for you — the LLM should only see content" |
| Not handling empty search results | Code Along 2, S1 | "What if the query has no match? Return 'I cannot find...'" |
| Forgetting `load_dotenv()` | Any | "Same as Week 2 — call it before os.getenv" |
| BM25 scores not normalised before fusion | Code Along 1, S2 | "BM25 is unbounded — it would dominate the sum" |
| Weights not summing to 1.0 | Code Along 1, S2 | "w_bm25 + w_vector must equal 1.0" |
| Re-ranking the whole corpus | Code Along 2, S2 | "Cross-encoder is slow — only run on top-10" |
| Metadata filter value type mismatch | Concept 3, S2 | "If metadata stores '2024' as int, filter with 2024 not '2024'" |
| Skipping the comparison (v1 vs enhanced) | Independent Lab, S2 | "The comparison IS the lab — you must document what improved" |

---

## KC Answer Key (for instructor reference)

### Session 1
- **KC1:** (b) To prevent hallucination from the LLM's training data
- **KC2:** (b) Strengthen the prompt instruction

### Session 2
- **KC1:** (a) When the query contains a rare technical term
- **KC2:** (b) To improve precision by applying a more accurate (but slower) model

---

## CLT Compliance Checklist (Week 4)

> Đối chiếu sau mỗi lần chỉnh sửa nội dung. Tất cả mục dưới đây hiện đã **✅ đạt** (kiểm tra lần cuối: 2026-08-17, đối chiếu với `cognitive_load_analysis.md` — 5 đề xuất P1–P5). Nếu một mục bị đổi thành ❌, cần sửa lại trước khi dùng tài liệu.

| # | Kiểm tra | Vị trí | Trạng thái |
|---|---|---|---|
| 1 | Hybrid search: persistent reference card (fusion pipeline + normalisation bắt buộc) + "predict before run" | Concept 1, Session 2 | ✅ |
| 2 | Mini-recall sprint (2 phút) giữa Concept 1 và Concept 2 (trước re-ranking) | Session 2 | ✅ |
| 3 | Note tải model cross-encoder ~80MB (pre-download trước giờ) — có trong prep checklist | Concept 2, Session 2 + lab_sheet Ex 2.2 | ✅ |
| 4 | Before/after visual: response shape Week 2 (answer) vs Week 4 (answer + sources) | Concept 2, Session 1 | ✅ |
| 5 | Reassurance Mid-Course Practical: trỏ tới `RAGSystem` của Lab 4 làm "shape" của bài thi | Wrap, Session 2 | ✅ |
| 6 | Concept 3 Metadata Filtering nén còn 15 phút (1:20–1:35) — dành thời gian cho Independent Lab so sánh | Session 2 | ✅ |
| 7 | Cấu trúc session: Part A — Theory 2h + Part B — Lab 2h (tổng 4h/tuần lý thuyết) | Header + bảng Session Structure | ✅ |
| 8 | Chuỗi fading: worked example → code-along (guided) → lab độc lập → peer review | Cả 2 session | ✅ |
| 9 | KCs cung cấp vòng phản hồi trong Part A (KC1 sau Concept 1, KC2 sau Concept 3) | Part A, cả 2 session | ✅ |
| 10 | Hoạt động elaborative/retrieval đặt đúng chỗ (lab opener / homework) | Wrap & Reflect + file companion | ✅ |

**Cách kiểm tra nhanh (grep):**

```bash
cd syllabus/week4
# Các dấu hiệu CLT phải hiện diện:
grep -c "reference card\|recall sprint\|80MB\|before/after\|RAGSystem\|PART A — THEORY (2h)" lecture_notes.md
# Tài liệu gốc phân tích:
# cognitive_load_analysis.md — 5 đề xuất (P1–P5) đã được áp dụng đầy đủ (xem mục "Summary of Recommended Changes")
```