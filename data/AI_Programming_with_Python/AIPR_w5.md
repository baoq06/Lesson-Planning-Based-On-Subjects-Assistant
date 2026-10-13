# Week 5: RAG Evaluation and Quantitative Measurement (Evaluation & Groundedness)

## 1. General Information & Learning Objectives (A5.1 to A5.4)
- **A5.1**: Quantitatively measure the 3 key metrics: Faithfulness (Groundedness), Answer Relevance, Context Precision.
- **A5.2**: Build an automated Evaluation Pipeline with a multi-group Testset and the LLM-as-a-judge technique.
- **A5.3**: Diagnose RAG failures through the metric matrix: distinguish a Retrieval failure (poor search) from a Generation failure (LLM Hallucination).
- **A5.4**: Make RAG improvement decisions from experimental figures instead of guesses.

## 2. Foundational Theory & System Architecture
### 2.1. The "Confident, Fluent, and Wrong" problem in RAG
A RAG system without an evaluation layer easily misleads users: the answer sounds fluent and confident but is actually fabricated information (Hallucination) because of:
- **Retrieval Failure**: The retriever returns wrong chunks or chunks without enough information.
- **Generation Failure**: The retriever returns the right chunks but the LLM ignores the context and answers from its external pre-trained data.

### 2.2. The RAGAS evaluation framework & quantitative metrics
- **Faithfulness (Groundedness)**:
  The percentage of claims in the answer that have direct evidence in the retrieved context:
  $$\text{Faithfulness} = \frac{|\text{Number of Claims supported by the Context}|}{|\text{Total number of Claims made}|}$$
  - Score $1.0$: the entire answer is fully based on the course material.
  - Score $<0.8$: the system is fabricating information.
- **Out-of-Scope Refusal Rate**:
  Measured on a set of questions outside the curriculum. The system must reach $100\%$ refusal and must not answer topics outside the course (cooking, sports, news).
- **Citation Precision**:
  Measures the accuracy of the lesson citation codes `[LP_W...]`. Ensures that the cited chunks are really among the top-k found by the retriever.

### 2.3. Diagnostic Matrix for the cause of RAG failures
| Faithfulness | Context Relevance | Diagnosis | Improvement |
| :---: | :---: | :--- | :--- |
| **LOW** | **HIGH** | LLM Hallucination (found the right chunk but made things up) | Strengthen the Grounding Prompt, set temperature = 0.0, use a Critic Verifier |
| **LOW** | **LOW** | Retrieval Gap (wrong chunks lead to a wrong answer) | Switch to Hybrid Search, adjust chunk size/overlap, tune $w_{bm25}$ |
| **HIGH** | **LOW** | The LLM correctly summarizes junk context it was given | Raise the similarity score filter (min_score threshold) |
| **HIGH** | **HIGH** | The RAG pipeline works perfectly | Maintain it and expand the testset |

## 3. Lab Exercises & Step-by-step Sample Code
### Exercise 5.1 — Designing a standardized Testset (data/testset.jsonl)
The testset is divided into 3 clearly separate groups of questions:
1. **Factual (direct retrieval)**: Questions whose answer lies entirely in 1 chunk (for example the cosine similarity formula).
2. **Synthesis**: Questions that require gathering information from several weeks (for example comparing the Week 4 and Week 7 tools).
3. **Out-of-Scope**: Questions deliberately outside the course scope to test the safe refusal mechanism.

### Exercise 5.2 — The LLM-as-a-judge technique for measuring Faithfulness
```python
FAITHFULNESS_JUDGE_PROMPT = """You are a RAG evaluation judge.
Task: Check every claim in the ANSWER against the CONTEXT.
If any detail is not supported by the context -> score 0.
If everything is traceable to the context -> score 1.

CONTEXT: {context}
ANSWER: {answer}

Return JSON in the format: {"score": 0 or 1, "unsupported_claims": [...]}
"""
```

### Exercise 5.3 — Automatic Refusal Rate computation
```python
def compute_refusal_metrics(predictions: list[dict]) -> float:
    refusals = 0
    out_of_scope = [p for p in predictions if p["type"] == "out_of_scope"]
    for item in out_of_scope:
        ans = item["answer"].lower()
        if any(kw in ans for kw in ["out of scope", "not covered in the course materials", "i only support the course"]):
            refusals += 1
    return refusals / len(out_of_scope) if out_of_scope else 1.0
```

## 4. Common Pitfalls & Design Principles (Gotchas & Best Practices)
- **Vibe Checking pitfall**: Never evaluate RAG by "trying a few questions and seeing that it reads smoothly"; you must run an automated testset and report quantitative metrics.
- **Judge Bias pitfall**: When an LLM acts as the judge it tends to favor longer answers over short ones; the judge prompt must define specific criteria and force JSON output.

## 5. Independent Tasks & Competency Assessment (Independent Labs)
- Build the file `tests/test_evaluation_pipeline.py` that automatically runs 12 test cases and outputs an accurate Benchmark table.
