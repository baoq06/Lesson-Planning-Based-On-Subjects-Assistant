# Week 4: Building the RAG Pipeline & Hybrid Search

## 1. General Information & Learning Objectives (A4.1 to A4.4)
- **A4.1**: Design a Grounding System Prompt: force the LLM to answer ONLY from the context that is provided.
- **A4.2**: Assemble a complete 5-stage RAG pipeline: Query -> Retrieve -> Build Prompt -> LLM Call -> Return Answer.
- **A4.3**: Manage and format source citations `[doc_id]` in the answer so that users can verify them.
- **A4.4**: Handle edge cases thoroughly: empty context, off-topic questions, prompts that exceed the token limit.

## 2. Foundational Theory & System Architecture
### 2.1. Motivation for Hybrid Search (Why Hybrid Search?)
- **Sparse Retrieval (BM25 Okapi)**:
  - Based on Term Frequency (TF) and Inverse Document Frequency (IDF).
  - Extremely strong when searching for exact keywords, identifiers (`DOC001`, `LP_W04`), rare terms and abbreviations.
  - Weakness: fails when the user uses synonyms or phrases the question differently (Vocabulary Mismatch Problem).
- **Dense Retrieval (Vector Embedding)**:
  - Captures semantic similarity (Semantic meaning) and naturally phrased questions.
  - Weakness: less accurate with technical codes or very specific terminology.
- **Score Fusion with Min-Max Normalisation**:
  Because BM25 scores are unbounded while Vector scores lie in $[0, 1]$, the scores must be normalized before they are combined:
  $$\text{norm}(S) = \frac{S - S_{\min}}{S_{\max} - S_{\min}}$$
  $$\text{Score}_{\text{hybrid}} = w_{\text{bm25}} \times \text{norm}(BM25) + w_{\text{vector}} \times \text{Vector}$$
  (The instructor's standard weights: $w_{\text{bm25}} = 0.4$, $w_{\text{vector}} = 0.6$).

### 2.2. Grounding Prompt Architecture & Anti-Hallucination Guardrail
The prompt acts as the only "firewall" that stops the LLM from making things up:
- It must contain a clear instruction: *"Answer only from the Context below. If the data is insufficient, you must refuse: 'I could not find enough information in the course materials'."*
- It must attach a chunk citation tag such as `[LP_W04_chunk_0]` after every statement in the answer.

## 3. Lab Exercises & Step-by-step Sample Code
### Exercise 4.1 — Grounding Prompt Template (ex1_1_rag_prompt.py)
```python
RAG_SYSTEM_PROMPT = """You are a study assistant for the AIPR course.
CORE RULES:
1. Answer the student's question ONLY from the CONTEXT provided below.
2. Never speculate or use knowledge from outside the materials.
3. If the CONTEXT does not contain the information needed to answer, you must say:
   "This content is not covered in the course materials."
4. Always cite the lesson source in the format [LP_W...] at the end of the answer.

CONTEXT:
{context}
"""
```

### Exercise 4.2 — BM25 Algorithm & Score Fusion (ex2_1_hybrid_search.py)
```python
from rank_bm25 import BM25Okapi

def normalise_scores(scores: list[float]) -> list[float]:
    if not scores or max(scores) == min(scores):
        return [1.0] * len(scores)
    s_min, s_max = min(scores), max(scores)
    return [(s - s_min) / (s_max - s_min) for s in scores]

def hybrid_search(query: str, chunks: list, bm25: BM25Okapi, vector_scores: list[float], top_k=3, w_b=0.4, w_v=0.6):
    tokenized_query = query.lower().split()
    bm25_raw = bm25.get_scores(tokenized_query)
    bm25_norm = normalise_scores(list(bm25_raw))

    fused_scores = [w_b * b + w_v * v for b, v in zip(bm25_norm, vector_scores)]
    ranked = sorted(enumerate(fused_scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunks[i], score) for i, score in ranked]
```

### Exercise 4.3 — Complete RAG Pipeline v2 (ex2_2_rag_pipeline_v2.py)
- Integrate data loading, run Hybrid Search, insert the Context into the prompt, call the Groq Chat Completion and print the result with citations.

## 4. Common Pitfalls & Design Principles (Gotchas & Best Practices)
- **Division by zero in Min-Max pitfall**: When all chunks have the same BM25 score ($S_{\max} == S_{\min}$), the normalization formula raises `ZeroDivisionError`; you must check this boundary condition and assign a default score array.
- **Context Overflow pitfall**: Combining too many chunks (`top_k` too large) overflows the LLM's tokens or triggers the Lost-in-the-Middle effect.

## 5. Independent Tasks & Competency Assessment (Independent Labs)
- Implement a check for empty questions or off-topic questions so that they are refused immediately, before calling the LLM, to save system resources.
