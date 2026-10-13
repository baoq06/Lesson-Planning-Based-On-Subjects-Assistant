# Week 2: LLM API Access & Prompt Management (Communication Layer)

## 1. General Information & Learning Objectives (A2.1 to A2.4)
- **A2.1**: Call Chat Completions through an OpenAI-compatible API client (Groq SDK / OpenAI SDK).
- **A2.2**: Handle Streaming responses and assemble the token deltas into the complete text.
- **A2.3**: Handle API exceptions (RateLimitError 429, timeouts, network errors) with a Retry mechanism using Exponential Backoff and Jitter.
- **A2.4**: Design parameterized Prompt Templates, separating the System Prompt from the User Prompt.

## 2. Foundational Theory & System Architecture
### 2.1. Request structure & core hyperparameters (Three Knobs)
Every LLM API call is tuned through 3 key technical parameters:
- `temperature`: Controls how spread out the probability distribution of the next token is.
  - $0.0$: Deterministic, always picks the token with the highest probability -> mandatory for RAG extraction, math and tool calling.
  - $>0.7$: Diverse and creative, used for writing emails, poems or open-ended discussion.
- `max_tokens`: The upper limit on the number of tokens the LLM may generate in one answer.
- `top_p` (Nucleus Sampling): Restricts sampling to the set of tokens whose cumulative probability reaches the threshold $p$.

### 2.2. Network failure handling & Rate Limit Resilience
- **Exponential Backoff with Random Jitter**:
  On an HTTP 429 (Too Many Requests) error, retrying immediately overloads the network or gets the API key blocked. The exponential wait formula with random noise is:
  $$\text{delay} = 2^{\text{attempt}} + \text{random.uniform}(0.1, 0.5)$$
- **Production-grade classification of HTTP errors**:
  - *Retryable*: codes `429` (Rate Limit), `500` (Internal Server Error), `502` (Bad Gateway), `503` (Service Unavailable), `APIConnectionError`.
  - *Non-retryable (stop immediately)*: codes `401` (wrong API Key), `400` (malformed prompt / invalid JSON Schema), `404` (wrong model name).

## 3. Lab Exercises & Step-by-step Sample Code
### Exercise 2.1 — Basic Chat Completion (test_api.py)
```python
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[
        {"role": "system", "content": "You are a study assistant for the AIPR course."},
        {"role": "user", "content": "List 3 advantages of Dataclass in Python."}
    ],
    temperature=0.0
)
print(response.choices[0].message.content)
```

### Exercise 2.2 — Streaming Response (streaming.py)
```python
def stream_llm_response(prompt: str):
    stream = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    full_content = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        full_content.append(delta)
    print()
    return "".join(full_content)
```

### Exercise 2.3 — ChatSession with Backoff & Token Logging (lab2_chat_session.py)
```python
import time, random, json
from groq import RateLimitError

def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=messages,
                temperature=0.0
            )
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0.1, 0.5)
            time.sleep(wait)
```

## 4. Common Pitfalls & Design Principles (Gotchas & Best Practices)
- **Hardcoded API Key pitfall**: Never keep the key string in source code; always read it through `os.getenv("GROQ_API_KEY")` and load it from `.env`.
- **Conversation memory overflow pitfall**: The LLM context window is limited; the chat history must not be stuffed in without bound, so apply a Sliding Window or conversation summarization.

## 5. Independent Tasks & Competency Assessment (Independent Labs)
- Build an `LLMClient` module with a usage token counter (`prompt_tokens`, `completion_tokens`) that writes an audit log to a JSONL file for cost calculation.
