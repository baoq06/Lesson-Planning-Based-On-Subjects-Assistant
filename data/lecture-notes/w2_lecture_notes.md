# Week 2 — Lecture Notes (Instructor Guide)

## LLM API Access & Integration

> **Course:** AI Programming with Python
> **Week:** 2 of 8 — 2 sessions/week × 4 hours (2h theory + 2h lab per session; theory total 4h/week)
> **Prerequisites:** Week 1 (data structures, JSON, @dataclass, type hints)
> **Learning Objectives:**
> - A2.1 Send a chat completion request to an LLM API (OpenAI-compatible) using the Python client.
> - A2.2 Handle streaming responses token-by-token and assemble the full response.
> - A2.3 Implement error handling for API errors (rate limits, authentication, timeouts).
> - A2.4 Design a prompt template with parameterised inputs.

---

## Session 1 — HTTP APIs & Chat Completions

> **Session format:** 4 hours total = **Part A — Theory (2h)** (this guide) + **Part B — Lab (2h)** (`lab_sheet.md`).

### Session Structure

| # | Segment | Duration | Gagné Event | Activity |
|---|---|---|---|---|
| — | **PART A — THEORY (2h)** | 0:00–2:00 | — | Segments 1–3, 5–6, 8–9, 12–13 |
| 1 | Hook & Problem | 0:00–0:10 | 1. Gain attention | Live demo: Python → LLM → response |
| 2 | Objectives & Recall | 0:10–0:20 | 2–3. Objectives + Recall | Recap Week 1; HTTP basics |
| 3 | Concept 1: API Fundamentals | 0:20–0:45 | 4. Present content | Anatomy of an LLM API call |
| 5 | KC 1 | 0:45–0:50 | 7. Feedback | MCQ: system message |
| 6 | Concept 2: Request/Response | 0:50–1:15 | 4. Present content | Parsing the response object |
| — | Break | 1:15–1:20 | — | — |
| 9 | Concept 3: Parameter Tuning | 1:20–1:40 | 4. Present content | temperature, max_tokens, top_p |
| 12 | KC 2 | 1:40–1:45 | 8. Assess | MCQ: JSON response format |
| 13 | Wrap & Reflect | 1:45–2:00 | 9. Retention/Transfer | Preview Session 2 |
| — | **PART B — LAB (2h)** | 2:00–4:00 | — | Segments 4, 7, 10–11 — see `lab_sheet.md` |
| 4 | Code Along 1 (guided) | 2:00–2:30 | 5–6. Guidance + Practice | First LLM call (lab_sheet Ex 1.1) |
| 7 | Code Along 2 (guided) | 2:30–2:55 | 5–6. Guidance + Practice | Response parsing + cost & logging (lab_sheet Ex 1.2) |
| 10 | Independent Lab | 2:55–3:35 | 6. Elicit performance | API Explorer CLI tool (lab_sheet Ex 1.3 — 40 min) |
| 11 | Peer Review | 3:35–3:50 | 7. Feedback | Pair code review (lab_sheet Ex 1.4 — 15 min) |
| — | Buffer / questions | 3:50–4:00 | — | — |

---

### Before Class: API Key Setup

**Critical prerequisite:** Every student must have an OpenAI API key (or compatible provider) before Session 1. Send this email 48 hours before class:

> **Subject:** Prepare for Week 2 — you need an API key
>
> In Week 2, we'll write Python code that calls an LLM API. You need:
> 1. An OpenAI API key: https://platform.openai.com/api-keys
> 2. At least $5 credit on your account
> 3. Install: `pip install openai python-dotenv`
> 4. Create a `.env` file in your project folder: `OPENAI_API_KEY=sk-...`
>
> **If you cannot get an OpenAI key**, email the instructor for an alternative (e.g., Groq free tier, or a class proxy key).

**Fallback plan:** If any student lacks an API key, prepare a mock module:

```python
# mock_openai.py — mock the OpenAI client for students without API keys
import random
import time
from datetime import datetime

class MockMessage:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = MockMessage(content)
        self.finish_reason = finish_reason
        self.index = 0

class MockUsage:
    prompt_tokens = random.randint(20, 100)
    completion_tokens = random.randint(50, 200)
    total_tokens = prompt_tokens + completion_tokens

class MockResponse:
    id = f"chatcmpl-mock-{datetime.now().timestamp()}"
    model = "gpt-4o-mini"
    object = "chat.completion"
    created = int(time.time())
    choices = [MockChoice("This is a mock response. In production, this would come from the LLM API.")]
    usage = MockUsage()

class MockStreamChunk:
    def __init__(self, text, finish_reason=None):
        self.choices = [type('', (), {'delta': type('', (), {'content': text})(), 'finish_reason': finish_reason, 'index': 0})()]

class MockStream:
    def __init__(self, text="Mock streaming response...", tokens=20):
        self._tokens = text.split() + ["The", "end", "."]
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self._tokens):
            raise StopIteration
        t = self._tokens[self._idx]
        self._idx += 1
        fr = "stop" if self._idx >= len(self._tokens) else None
        return MockStreamChunk(t + " ", fr)

class MockOpenAI:
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                if kwargs.get("stream"):
                    return MockStream()
                return MockResponse()

    class embeddings:
        @staticmethod
        def create(*args, **kwargs):
            return type('', (), {'data': [type('', (), {'embedding': [random.uniform(-0.1, 0.1) for _ in range(384)]})()]})()
```

---

### 1. Hook & Problem (0:00–0:10)

**Live demo — do this on your machine with a real API key:**

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what an API is in one sentence."}
    ]
)

print(response.choices[0].message.content)
```

**Instructor script:**

> "Look at that. 15 lines of Python code, and we just talked to one of the most advanced AI systems on the planet. This is the simplest AI program you can write: input → API → output.
>
> "In Week 1, you learned how to model data with @dataclass and JSON. Now you're going to *send* that data to an LLM and *receive* responses.
>
> "By the end of this session, you'll be able to call any LLM API, parse the response, tune parameters, and log everything for debugging. This is the foundation for everything else in this course — RAG, agents, multi-agent systems."

**Cognitive load note:** The live demo is a *worked example* at the session level. Students see the complete solution before any explanation. This reduces extraneous load (they know where we're going) and activates prior knowledge (they recognise the `json.load()`, dict patterns from Week 1 in the response parsing).

---

### 2. Objectives & Recall (0:10–0:20)

**State objectives:**

> "By the end of this session, you will be able to:
> 1. Call any LLM API using the OpenAI Python client.
> 2. Parse the response object to extract content, tokens, and model info.
> 3. Calculate cost and log API calls for debugging.
> 4. Tune parameters like temperature, max_tokens, and top_p."

**Recall prompt:**

> "From Week 1:
> - How do you access a nested dict value safely? (`.get()`)
> - What does `json.dump()` do? (write Python object to JSON file)
> - What is a `@dataclass`? (auto-generates `__init__`, `__repr__`, `__eq__`)
>
> New recall — HTTP basics:
> - What is a POST request? (sends data to server, gets response)
> - What is an API endpoint? (URL where the API lives)
> - What is an API key? (authentication token)"

**Expected prior knowledge:** Students have used HTTP conceptually (web browsing). They may not have written HTTP client code. This is fine — the OpenAI client abstracts HTTP details.

---

### 3. Concept 1 — API Fundamentals (0:20–0:45)

#### Worked Example: Anatomy of an LLM API Call

**Show this diagram on the projector:**

```
┌─────────────────────────────────────────────────────────┐
│  Your Python Script           OpenAI API Server         │
│                              │                          │
│  POST /v1/chat/completions ──┤                          │
│  Headers:                    │                          │
│    Authorization: Bearer sk-...│                        │
│    Content-Type: application/json                       │
│  Body: {                     │                          │
│    model: "gpt-4o-mini",     │                          │
│    messages: [               │                          │
│      {role: "system", content: "..."},                  │
│      {role: "user", content: "..."}                     │
│    ],                        │                          │
│    temperature: 0.7          │                          │
│  }                           │                          │
│                              │                          │
│  Response ◄──────────────────┤                          │
│  {                           │                          │
│    id: "chatcmpl-...",       │                          │
│    choices: [{message: {content: "..."}, finish_reason: "stop"}],│
│    usage: {prompt_tokens: 25, completion_tokens: 50}    │
│  }                           │                          │
└─────────────────────────────────────────────────────────┘
```

**Key insight:** "Every LLM API follows the same pattern: you send a list of messages, you get back a completion. The model, the parameters, and the authentication method may differ — but the *structure* is universal."

**Worked example — recommended way to pass the API key (use this):**

```python
# RECOMMENDED: .env file (best for development)
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

**Alternative (if you can't use .env):**

```python
# Alternative: direct environment variable
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

> **⚠️ Never hard-code API keys in your code.** If you see `api_key="sk-..."` in shared code, that's a security leak. Always use `.env` or environment variables. Add `.env` to `.gitignore`.

**Teaching tip:** Emphasise that **API keys must never be committed to Git**. Show `.gitignore` with `.env` in it. This is a real-world security lesson.

---

### 4. Code Along 1 — First LLM Call (2:00–2:30) [PART B — LAB]

**File:** `ex1_1_first_call.py`

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found. Create a .env file.")
    print("Add: OPENAI_API_KEY=sk-...")
    exit(1)

# Step 2: Initialise client
client = OpenAI(api_key=api_key)

# Step 3: Prepare messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is RAG in AI?"}
]

# Step 4: Send request
print("Sending request...")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0.7
)

# Step 5: Print response
print("\n=== Response ===")
print(response.choices[0].message.content)
print("\n=== Metadata ===")
print(f"Model: {response.model}")
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

**Common mistakes to watch for:**

| Mistake | Correction |
|---|---|
| `load_dotenv()` not called → `os.getenv()` returns `None` | Call `load_dotenv()` before `os.getenv()` |
| API key has quotes in `.env`: `OPENAI_API_KEY="sk-..."` | Remove quotes: `OPENAI_API_KEY=sk-...` |
| `pip install openai` not run → `ModuleNotFoundError` | `pip install openai python-dotenv` |
| Typo: `messsages` instead of `messages` | Check spelling |
| `response.choices[0].message` but `choices` is empty | Check API response format |

**Check for understanding:** "Everyone should see the model's response printed. If you see an error about API key, raise your hand."

---

### 5. Knowledge Check 1 (0:45–0:50)

**MCQ:**

> What is the purpose of the 'system' message in a chat completion request?
>
> (a) To log the request on the server
> (b) To set the assistant's behaviour and persona
> (c) To authenticate the user making the request

**Correct answer: (b) Set the assistant's behaviour**

**Explanation:** The system message is a prompt-level instruction that sets the context, behaviour, and persona of the assistant. It's not a log, not authentication — it's the "system prompt" or "meta-instruction" that guides how the LLM should respond.

**Further discussion:** "What would happen if you set the system message to 'You are a pirate who only speaks in rhymes'?" (The assistant would respond as a rhyming pirate.)

---

### 6. Concept 2 — Request/Response Structure (0:50–1:15)

#### Worked Example: Parsing the Response Object

**Show the full response object structure:**

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello"}]
)

# The response object is NOT a dict — it's a Pydantic-like object
# But you access it with dot notation, similar to dict access

# === Top-level fields ===
print(response.id)          # "chatcmpl-9a8b7c6d5e"
print(response.object)      # "chat.completion"
print(response.created)     # 1712345678 (Unix timestamp)
print(response.model)       # "gpt-4o-mini"

# === Choices (list of completions) ===
# Most common: response.choices[0]
choice = response.choices[0]
print(choice.index)              # 0
print(choice.finish_reason)      # "stop", "length", "content_filter", "null"
print(choice.message.role)       # "assistant"
print(choice.message.content)    # "Hello! How can I help you today?"

# === Usage (token counts) ===
usage = response.usage
print(usage.prompt_tokens)       # 12
print(usage.completion_tokens)   # 8
print(usage.total_tokens)        # 20

# === Multiple choices ===
response_n = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in 3 different ways"}],
    n=3  # Return 3 completions
)
for i, choice in enumerate(response_n.choices):
    print(f"Option {i+1}: {choice.message.content}")
```

**Key insight:** "The response object is a nested object — it behaves like a dict but uses dot notation. The Week 1 skill of navigating nested dicts translates directly: `response -> choices[0] -> message -> content`."

**Teaching tip:** Use `print(type(response))` and `print(dir(response))` to show students how to explore any object. This is a skill they'll use constantly.

#### Cost Estimation

```python
# Pricing for gpt-4o-mini (as of 2026)
# Input: $0.15 per 1M tokens
# Output: $0.60 per 1M tokens

def estimate_cost(usage, input_price=0.15, output_price=0.60):
    """Estimate API call cost in USD."""
    input_cost = (usage.prompt_tokens / 1_000_000) * input_price
    output_cost = (usage.completion_tokens / 1_000_000) * output_price
    return input_cost + output_cost

cost = estimate_cost(response.usage)
print(f"Estimated cost: ${cost:.6f}")
```

---

### 7. Code Along 2 — Response Parsing (Core) + Cost & Logging (Optional) (2:30–2:55) [PART B — LAB]

> **Teaching note:** Split this into two steps. **Step 1 (primary skill):** parse the response and display fields. **Step 2 (production feature, optional):** add cost estimation and logging. Students should master Step 1 first; Step 2 is a progressive enhancement.

#### Step 1 (Core) — Parse the response and display the fields

```python
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Main: send a request and parse the response ===
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

# Step 1: Extract and display the core fields
content = response.choices[0].message.content
usage = response.usage

print(f"Response: {content}")
print(f"Model: {response.model}")
print(f"Finish reason: {response.choices[0].finish_reason}")
print(f"Tokens: {usage.prompt_tokens} in + {usage.completion_tokens} out = {usage.total_tokens} total")
```

**Checkpoint:** Everyone should see the response, model, finish reason, and token counts. Confirm this works before moving to Step 2.

#### Step 2 (Optional — Production Feature) — Cost estimation and logging

```python
# Pricing table (per 1M tokens, USD)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
}

def estimate_cost(model, prompt_tokens, completion_tokens):
    """Estimate cost in USD for an API call."""
    pricing = PRICING.get(model, {"input": 0.15, "output": 0.60})
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost

def log_api_call(messages, response, filename="data/api_log.jsonl"):
    """Log an API call to a JSONL file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": response.model,
        "prompt": messages,
        "response": response.choices[0].message.content,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "cost": estimate_cost(
            response.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        ),
        "finish_reason": response.choices[0].finish_reason
    }
    with open(filename, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    return log_entry

# Extend the main flow with cost + logging (only if Step 1 worked)
cost = estimate_cost(response.model, usage.prompt_tokens, usage.completion_tokens)
print(f"Cost: ${cost:.6f}")

entry = log_api_call(messages, response)
print(f"Logged to data/api_log.jsonl")
```

# Log to file
entry = log_api_call(messages, response)
print(f"Logged to data/api_log.jsonl")
```

**Teaching tip:** Show the JSONL file after running — each line is a JSON object. This is the standard format for AI observability (LangSmith, Weights & Biases, etc.).

---

### 8. Break (1:15–1:20)

---

### 9. Concept 3 — Parameter Tuning (1:20–1:40)

#### Worked Example: Temperature

> **Visual: How temperature reshapes the token probability distribution**
>
> ```
> temperature = 0.0 (sharp peak):          temperature = 1.5 (flat):
>   ┌──────────────────┐                    ┌──────────────────┐
>   │   ████████       │   top token       │ ██ ██  ██  ██ ██│  many tokens
>   │   ████████       │   almost always   │ ██ ██  ██  ██ ██│  are plausible
>   │   ████████       │   chosen          │██████████████████│
>   └──────────────────┘                    └──────────────────┘
>   → deterministic                          → creative but risky
> ```

```python
# Temperature controls randomness
# 0.0 = deterministic (same input → same output)
# 0.7 = balanced (default)
# 1.5 = highly creative (may be incoherent)

prompt = "Write a short tagline for a Python AI course."

for temp in [0.0, 0.7, 1.5]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"\n--- Temperature: {temp} ---")
    print(response.choices[0].message.content)
```

**Key insight:** "Temperature ∈ [0, 2]. At 0.0, the model picks the most likely token every time — deterministic but boring. At 1.0+, it picks less likely tokens — creative but risky. For production, use 0.0 if you need consistent JSON output."

#### Worked Example: max_tokens

```python
# max_tokens limits the response length (not the prompt length)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a 500-word essay on Python."}],
    max_tokens=50  # Hard cut at 50 tokens
)
print(response.choices[0].message.content)
print(f"Finish reason: {response.choices[0].finish_reason}")  # "length"
```

**Key insight:** "When `finish_reason` is `'length'`, it means the response was cut off by `max_tokens`. When it's `'stop'`, the model finished naturally. Always check `finish_reason`."

#### Worked Example: top_p (Nucleus Sampling)

```python
# top_p = 0.1: Only consider the top 10% most likely tokens
# top_p = 1.0: Consider all tokens (default)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Complete this: The quick brown fox..."}],
    top_p=0.1  # Very focused, deterministic
)
```

**Teaching tip:** "Don't use temperature AND top_p simultaneously — the API docs recommend using one or the other. Temperature is more intuitive for most use cases."

---

### 10. Independent Lab — API Explorer (2:55–3:35) [PART B — LAB]

**File:** `ex1_3_api_explorer.py`

Students build a CLI tool. See `lab_sheet.md` for full instructions.

**Setup checklist (before class):**
- [ ] Verify all students have API keys working
- [ ] Have mock module ready for students without keys
- [ ] Test the solution yourself

**Teaching tips:**
- Circulate during the first 5 minutes. Check that everyone's API key is loading.
- Common issue: `.env` file in wrong directory → `load_dotenv()` looks in current directory
- Common issue: `os.getenv("OPENAI_API_KEY")` returns `None` → check `.env` syntax

---

### 11. Peer Review (3:35–3:50) [PART B — LAB]

**Pair up. Review checklist:**

> 1. Does the script handle missing API keys gracefully? (not crash with AttributeError)
> 2. Are both temperatures run and compared?
> 3. Are token counts displayed?
> 4. Is the conversation logged to a JSON file?
> 5. Is the code readable? (variable names, comments)

---

### 12. Knowledge Check 2 (1:40–1:45)

**MCQ:**

> You need the model to ALWAYS respond in valid JSON format. Which parameter is most important?
>
> (a) `temperature=0.0`
> (b) `response_format={"type": "json_object"}`
> (c) `max_tokens=100`

**Correct answer: (b) `response_format={"type": "json_object"}`**

**Explanation:** The `response_format` parameter explicitly tells the model to output valid JSON. While `temperature=0.0` helps with consistency, it doesn't guarantee JSON. Without `response_format`, the model may output text with JSON embedded in it, which is harder to parse.

---

### 13. Wrap & Reflect (1:45–2:00)

**Instructor summary:**

> "Today you:
> - Called an LLM API for the first time
> - Parsed the response object (content, tokens, model info)
> - Estimated costs and logged API calls
> - Tuned temperature, max_tokens, and top_p
>
> **Session 2:** We'll add streaming (real-time token-by-token output), error handling for production robustness, and prompt templates. You'll build a complete chat assistant."

**Reflection prompt (3 min):**

> "What was the most surprising thing about calling an LLM API? What questions do you have about parameters?"

---

## Session 2 — Streaming, Error Handling & Prompt Templates

> **Session format:** 4 hours total = **Part A — Theory (2h)** (this guide) + **Part B — Lab (2h)** (`lab_sheet.md`).

### Session Structure

| # | Segment | Duration | Gagné Event | Activity |
|---|---|---|---|---|
| — | **PART A — THEORY (2h)** | 0:00–2:00 | — | Segments 1–3, 5–6, 8–9, 12–13 |
| 1 | Hook | 0:00–0:10 | 1. Gain attention | Streaming vs non-streaming comparison |
| 2 | Objectives & Recall | 0:10–0:20 | 2–3. Objectives + Recall | Recap Session 1 |
| 3 | Concept 1: Streaming | 0:20–0:45 | 4. Present content | Stream=True, chunk iteration |
| 5 | KC 1 | 0:45–0:50 | 7. Feedback | MCQ: finish_reason |
| 6 | Concept 2: Error Handling | 0:50–1:15 | 4. Present content | try/except, retry, backoff |
| — | Break | 1:15–1:20 | — | — |
| 9 | Concept 3: Prompt Templates | 1:20–1:40 | 4. Present content | string.Template, f-strings |
| 12 | KC 2 | 1:40–1:45 | 8. Assess | MCQ: prompt template |
| 13 | Wrap & Reflect | 1:45–2:00 | 9. Retention/Transfer | Preview Week 3 |
| — | **PART B — LAB (2h)** | 2:00–4:00 | — | Segments 4, 7, 10–11 — see `lab_sheet.md` |
| 4 | Code Along 1 (guided) | 2:00–2:30 | 5–6. Guidance + Practice | Streaming chat (lab_sheet Ex 2.1) |
| 7 | Code Along 2 (guided) | 2:30–2:55 | 5–6. Guidance + Practice | Retry wrapper + logging (lab_sheet Ex 2.2) |
| 10 | Independent Lab | 2:55–3:35 | 6. Elicit performance | Chat assistant (lab_sheet Ex 2.3 — 40 min) |
| 11 | Peer Review | 3:35–3:50 | 7. Feedback | Pair code review (lab_sheet Ex 2.4 — 15 min) |
| — | Buffer / questions | 3:50–4:00 | — | — |

---

### 1. Hook — Streaming vs Non-Streaming (0:00–0:10)

**Live demo — run both side by side:**

```python
# Non-streaming (wait for full response)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a 200-word story about a robot learning to paint."}]
)
print(response.choices[0].message.content)

# Streaming (tokens appear one by one)
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a 200-word story about a robot learning to paint."}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**Instructor script:**

> "Notice the difference? The first one waits 5 seconds, then shows everything. The second one starts showing words immediately. Stream makes the user experience feel instant.
>
> "But streaming adds complexity: instead of one clean response object, you get a stream of partial chunks. Each chunk has a tiny piece of the response. You need to assemble them manually.
>
> "This is what every AI chat interface does — ChatGPT, Claude, Copilot — they all stream. By the end of this session, you'll build your own streaming chat assistant."

---

### 2. Objectives & Recall (0:10–0:20)

**State objectives:**

> "By the end of this session, you will be able to:
> 1. Stream responses token-by-token.
> 2. Handle API errors gracefully (rate limits, timeouts, auth failures).
> 3. Implement exponential backoff for retries.
> 4. Design parameterised prompt templates.
> 5. Build a complete chat assistant with streaming + history."

**Recall prompt:**

> "From Session 1:
> - How do you call the chat completions API? (`client.chat.completions.create(...)`)
> - What does the response object contain? (choices, usage, model, id)
> - What does `temperature` control? (randomness of token selection)
> - What is the `.env` file for? (storing API keys securely)"

---

### 3. Concept 1 — Streaming Responses (0:20–0:45)

#### Worked Example: The Streaming Flow

**Show this diagram:**

```
Non-streaming:
  Request ────────────────────► Response (full) ───────► Print

Streaming:
  Request ──► Chunk 1 (partial) ──► Print
           ──► Chunk 2 (partial) ──► Print
           ──► Chunk 3 (partial) ──► Print
           ──► ... ──► Final chunk (finish_reason) ──► Done
```

> **Mental-model shift: non-streaming vs streaming** (this is the key idea of this concept)
>
> ```
> NON-STREAMING (Session 1)          STREAMING (Session 2)
> response = client...create()       stream = client...create(stream=True)
>                                     for chunk in stream:
> response.choices[0].message          chunk.choices[0].delta.content
>   .content  → ONE full string          → partial fragment each time
> response is complete                you must accumulate deltas yourself
> print(response...)                  append + print as they arrive
> ```

**Predict-the-output prompt (30 seconds, think-pair-share) — before looking at the code:**

> "Look at the streaming flow diagram above. The API sends chunks one at a time. What do you think happens on the **very last chunk** of the stream?
>
> 1. What will `delta.content` contain on the last chunk?
> 2. What will `finish_reason` be — and how will we know the stream is done?
>
> Discuss with your partner for 30 seconds, then we'll look at the code and see if your prediction was right."

**Expected answers (reveal AFTER discussion):** The last chunk has `delta.content = None` (no more text) and `finish_reason = "stop"` — this is how the loop knows to stop. If students predicted this correctly, highlight it; if not, use the mismatch as the teachable moment.

**Worked example — streaming with assembly:**

```python
# Each chunk is a partial delta
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    stream=True
)

full_response = ""
for chunk in stream:
    # Each chunk has choices[0].delta.content
    # The final chunk has choices[0].delta.content = None
    delta = chunk.choices[0].delta
    finish_reason = chunk.choices[0].finish_reason

    if delta.content:
        content = delta.content
        full_response += content
        print(content, end="", flush=True)

    if finish_reason:
        print(f"\n\n[Finished: {finish_reason}]")

print(f"\n\n=== Full response ({len(full_response)} chars) ===")
print(full_response)
```

**Key insight:** "Each chunk is a micro-delta — a tiny piece of the full response. The final chunk has `delta.content = None` and `finish_reason = 'stop'`. You must accumulate the deltas yourself."

#### Important: Choice Index in Streaming

```python
# In streaming, chunk.choices may be empty for certain event types
# Always check before accessing

for chunk in stream:
    if not chunk.choices:  # Some chunks have no choices (e.g., rate limit info)
        continue
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

---

### 4. Code Along 1 — Streaming Chat (2:00–2:30) [PART B — LAB]

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def stream_response(messages, model="gpt-4o-mini"):
    """Send messages and stream the response token by token."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    full_response = ""
    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        if delta.content:
            content = delta.content
            full_response += content
            print(content, end="", flush=True)

        if finish_reason:
            print(f"\n[Finish reason: {finish_reason}]")

    return full_response

# Test
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What are the three most important Python data types for AI?"}
]

print("Assistant: ", end="")
response = stream_response(messages)
```

**Checkpoint:** Students should see tokens appearing one by one, then a final assembled response.

---

### 5. Knowledge Check 1 (0:45–0:50)

**MCQ:**

> In a streaming response, what does `chunk.choices[0].finish_reason = 'stop'` indicate?
>
> (a) The model has finished generating the response naturally.
> (b) The response was cut off by `max_tokens`.
> (c) An error occurred during generation.

**Correct answer: (a) Natural completion**

**Explanation:** `finish_reason = 'stop'` means the model decided to stop (it generated an end-of-turn token). `'length'` means max_tokens was hit. `'content_filter'` means the response was flagged by the content moderation filter. `'null'` means the generation is still in progress.

---

### 6. Concept 2 — Error Handling & Robustness (0:50–1:15)

> **Teaching note:** This concept is split into **two stages** to manage cognitive load. Stage A (simple) introduces the loop + delay formula with only 2 retryable errors. Stage B (extend) adds the full error taxonomy. Master Stage A before moving to Stage B.

#### Stage A (Core) — Simple Retry Loop with Exponential Backoff

Start with the core idea: a loop that retries with increasing delay when a rate limit or connection error occurs.

> **Persistent reference card — keep visible for the whole session:**
>
> ```
> RETRY (transient)         DON'T RETRY (fix the root cause)
> ─────────────────────     ────────────────────────────────
> RateLimitError            AuthenticationError  → check API key
> APIConnectionError        BadRequestError      → fix request params
>                           APIStatusError (4xx)  → fix request
> ```

```python
import time
import random
from openai import RateLimitError, APIConnectionError

def call_with_retry_simple(messages, model="gpt-4o-mini", max_retries=3):
    """
    Simple retry loop with exponential backoff.
    Only handles RateLimitError and APIConnectionError (both are retryable).
    """
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages
            )
            return response  # Success — exit the loop

        except RateLimitError:
            if attempt < max_retries:
                delay = 2 ** attempt + random.uniform(0, 1)  # 1s, 2s, 4s
                print(f"Rate limit hit. Retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
            else:
                print("Max retries reached.")
                raise

        except APIConnectionError:
            if attempt < max_retries:
                delay = 2 ** attempt + random.uniform(0, 1)
                print(f"Connection error. Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                print("Max retries reached.")
                raise
```

**Key insight — the core formula:** `delay = base_delay * (2 ** attempt) + random_jitter`. The delay doubles each attempt (1s → 2s → 4s) plus a small random jitter to prevent thundering-herd problems.

#### Stage B (Extend) — Full Error Taxonomy with Logging

Now extend the pattern to handle all error types, distinguishing retryable from non-retryable errors.

```python
from openai import (
    RateLimitError, APIConnectionError, AuthenticationError,
    BadRequestError, APIStatusError
)

def call_with_retry(messages, model="gpt-4o-mini", max_retries=3):
    """
    Full retry loop with error taxonomy.

    RETRY:        RateLimitError, APIConnectionError, APIStatusError (5xx)
    DON'T RETRY:  AuthenticationError (wrong key), BadRequestError (fix code)
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages
            )
            return response

        except (RateLimitError, APIConnectionError) as e:
            last_error = e
            if attempt < max_retries:
                delay = 2 ** attempt + random.uniform(0, 1)
                print(f"{type(e).__name__}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                print(f"Failed after {max_retries} retries: {e}")
                raise

        except AuthenticationError:
            print("Authentication failed. Check your API key.")
            raise  # Don't retry — wrong key

        except BadRequestError as e:
            print(f"Bad request: {e}")
            raise  # Don't retry — fix the code

        except APIStatusError as e:
            if 500 <= e.status_code < 600 and attempt < max_retries:
                delay = 2 ** attempt + random.uniform(0, 1)
                print(f"Server error {e.status_code}. Retrying...")
                time.sleep(delay)
            else:
                raise

    raise last_error or Exception("Unknown error after retries")
```

**Key insight:** "Not all errors are retryable. Rate limits and connection errors → retry. Authentication errors → don't retry (the key is wrong). Bad requests → don't retry (fix the code). The reference card (above) is your guide."

---

### 7. Code Along 2 — Retry Wrapper + Logging (2:30–2:55) [PART B — LAB]

```python
import os
import time
import json
import random
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(messages, model="gpt-4o-mini", max_retries=3, stream=False):
    """Call LLM with retry logic and logging."""
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            start_time = time.time()

            if stream:
                # Return stream directly (can't retry mid-stream)
                return client.chat.completions.create(
                    model=model, messages=messages, stream=True
                )
            else:
                response = client.chat.completions.create(
                    model=model, messages=messages
                )

            # Log success
            elapsed = time.time() - start_time
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "latency_seconds": round(elapsed, 2),
                "success": True,
                "attempt": attempt + 1
            }
            with open("data/api_log.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            return response

        except (RateLimitError, APIConnectionError) as e:
            last_error = e
            if attempt < max_retries:
                delay = 2 ** attempt + random.uniform(0, 1)
                print(f"⚠ Error: {type(e).__name__}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                # Log failure
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "error": type(e).__name__,
                    "success": False,
                    "attempt": attempt + 1
                }
                with open("data/api_log.jsonl", "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            else:
                print(f"✗ Failed after {max_retries} retries: {e}")
                raise

        except Exception as e:
            print(f"✗ Non-retryable error: {type(e).__name__}: {e}")
            raise

# Test
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello in 5 words."}
]

print("Calling LLM...")
response = call_llm(messages)
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens: {response.usage.total_tokens}")
```

---

### 8. Break (1:15–1:20)

---

### 9. Concept 3 — Prompt Templates (1:20–1:40)

#### The Problem: Hard-Coded Prompts

```python
# Bad: hard-coded prompt
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a Python expert."},
        {"role": "user", "content": f"Explain {topic} in Python."}
    ]
)
# What if you want to change the persona? Change the topic format?
# You'd have to edit the code.
```

#### Worked Example: Parameterised Prompt Templates

```python
from string import Template

# Template-based approach
SYSTEM_PROMPT_TEMPLATE = Template("""
You are a $persona assistant.
Your tone should be $tone.
Keep responses under $max_words words.
""")

USER_PROMPT_TEMPLATE = Template("""
Context: $context

Question: $question

Answer based only on the context provided above.
""")

# Use the templates
system_prompt = SYSTEM_PROMPT_TEMPLATE.safe_substitute(
    persona="Python programming",
    tone="concise and technical",
    max_words=50
)

user_prompt = USER_PROMPT_TEMPLATE.safe_substitute(
    context="Python lists are mutable sequences. Tuples are immutable.",
    question="What is the difference between a list and a tuple?"
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

print("=== System Prompt ===")
print(system_prompt)
print("\n=== User Prompt ===")
print(user_prompt)
```

#### Alternative: f-string with validation

```python
def build_prompt(persona: str, question: str, context: str = "") -> list[dict]:
    """Build a chat message list with parameterised prompts."""
    # Validate inputs
    if not persona.strip():
        raise ValueError("Persona cannot be empty")
    if not question.strip():
        raise ValueError("Question cannot be empty")

    system = f"You are a {persona} assistant. Be helpful and accurate."
    user = question

    if context:
        user = f"Context: {context}\n\nQuestion: {question}\n\nAnswer based only on the context."

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

# Usage
messages = build_prompt(
    persona="Python expert",
    question="What is a decorator?",
    context="Decorators are functions that modify other functions."
)
```

**Key insight:** "Prompts are code. Version them, test them, parameterise them. A prompt template is like a function — it takes parameters and returns a structured output."

---

### 10. Independent Lab — Chat Assistant (2:55–3:35) [PART B — LAB]

**File:** `ex2_3_chat_assistant.py`

Students build a complete chat assistant. See `lab_sheet.md` for full instructions.

**Teaching tips:**
- The message history must be managed carefully — it grows unbounded
- Demonstrate the unbounded growth problem: run 10 turns, show the token count
- Then add sliding window: keep only last N messages
- Ctrl+C handling: `try/except KeyboardInterrupt`

---

### 11. Peer Review (3:35–3:50) [PART B — LAB]

**Pair up. Review checklist:**

> 1. Does the message history grow unbounded, or is there a size limit?
> 2. Is the streaming text smooth? (no buffering, no chunking artifacts)
> 3. Is Ctrl+C handled gracefully? (clean exit, not a traceback)
> 4. Is the prompt template parameterised? (not hard-coded)
> 5. Are API errors handled? (try/except around API calls)

---

### 12. Knowledge Check 2 (1:40–1:45)

**MCQ:**

> Which is the best practice for building a prompt with dynamic content?
>
> (a) String concatenation with `+`
> (b) Parameterised template with validation (f-string or Template)
> (c) Hard-code all prompts

**Correct answer: (b) Parameterised template with validation**

**Explanation:** Parameterised templates (f-strings with validation, `string.Template`) are testable, versionable, and safe. String concatenation is error-prone (missing spaces, type errors). Hard-coded prompts break whenever the content changes.

---

### 13. Wrap & Reflect (1:45–2:00)

**Instructor summary:**

> "This week you built the communication layer for AI programming:
> - **Session 1:** Called an LLM API, parsed responses, tuned parameters, logged calls.
> - **Session 2:** Streamed responses, handled errors with retries, built prompt templates.
>
> Next week: RAG. You have the data structures (Week 1) and the API access (Week 2). Now you'll combine them: store documents, embed them, and retrieve relevant chunks. This is the foundation of grounded AI.
>
> **Lab 2 (due before Session 5):** Build a `ChatSession` class that combines everything from this week — streaming, error handling, logging, prompt templates, and message history."

**Reflection prompt (3 min):**

> "What was the most technically challenging part of this week? What do you feel confident about?"

---

## Instructor Preparation Checklist

### Before Session 1

- [ ] Verify all students have API keys (email reminder 48h before)
- [ ] Prepare mock module for students without keys
- [ ] Test live demo with your API key
- [ ] Create `data/` directory
- [ ] Test all Code Along scripts
- [ ] Test the Independent Lab solution
- [ ] Prepare KC slides (3-option MCQs)
- [ ] Ensure `pip install openai python-dotenv` is run on all machines

### Before Session 2

- [ ] Test streaming demo (bandwidth may affect latency — test beforehand)
- [ ] Test rate limit simulation (or just show the error handling code)
- [ ] Test all Code Along scripts
- [ ] Test the Independent Lab solution
- [ ] Review Session 1 reflections to adjust pace

### API Key Management

- **Never commit API keys to Git.** Add `.env` to `.gitignore`.
- **Cost control:** gpt-4o-mini is very cheap (~$0.000001 per call). Even 100 calls per student per session costs < $0.01 per student.
- **Rate limits:** gpt-4o-mini has 500,000 tokens/minute on Tier 1. For a class of 30 students, that's fine.
- **If OpenAI is blocked:** Use Groq (free, no API key required for small usage), or set up a class proxy.

---

## Common Student Mistakes & Interventions

| Mistake | When It Appears | Intervention |
|---|---|---|
| `load_dotenv()` not called → `os.getenv()` returns None | Code Along 1, S1 | "Call `load_dotenv()` before `os.getenv()`" |
| `.env` file in wrong directory | Code Along 1, S1 | "Put `.env` in the same directory as your script" |
| `pip install openai` not run | Code Along 1, S1 | "Run `pip install openai python-dotenv`" |
| Typo: `messsages` instead of `messages` | Any API call | "Check spelling of `messages`" |
| Forgetting `import OpenAI` | Any API call | "Add `from openai import OpenAI`" |
| `response.choices[0]` but `choices` is empty | Code Along 1, S1 | "Check the API response format — use `print(dir(response))`" |
| Streaming: `delta.content` is `None` on final chunk | Code Along 1, S2 | "Check for `None` before printing" |
| No error handling → script crashes on rate limit | Lab, S2 | "Wrap the API call in `try/except`" |
| Message history grows unbounded | Lab, S2 | "Show token count, then add sliding window" |
| Prompt template has KeyError for missing variable | Concept 3, S2 | "Use `.safe_substitute()` instead of `.substitute()`" |

---

## KC Answer Key (for instructor reference)

### Session 1
- **KC1:** (b) Set the assistant's behaviour — the system message configures persona and constraints
- **KC2:** (b) `response_format={"type": "json_object"}` — guarantees valid JSON output

### Session 2
- **KC1:** (a) Natural completion — model finished generating on its own
- **KC2:** (b) Parameterised template with validation — testable, versionable, safe

---

## CLT Compliance Checklist (Week 2)

> Đối chiếu sau mỗi lần chỉnh sửa nội dung. Tất cả mục dưới đây hiện đã **✅ đạt** (kiểm tra lần cuối: 2026-08-17, đối chiếu với `cognitive_load_analysis.md`). Nếu một mục bị đổi thành ❌, cần sửa lại trước khi dùng tài liệu.

| # | Kiểm tra | Vị trí | Trạng thái |
|---|---|---|---|
| 1 | Exponential backoff tách 2 giai đoạn (Stage A: 2 lỗi retryable → Stage B: full taxonomy) | Concept 2, Session 2 + `lab_sheet.md` Ex 2.2 | ✅ |
| 2 | Streaming: bảng so sánh before/after + prompt "predict the output" (think-pair-share) | Concept 1, Session 2 | ✅ |
| 3 | Cost/logging tách Step 1 (core: parse) / Step 2 (optional: production feature) | Code Along 2, Session 1 + `lab_sheet.md` Ex 1.2 | ✅ |
| 4 | ASCII visual phân phối xác suất cho temperature (0.0 vs 1.5) | Concept 3, Session 1 | ✅ |
| 5 | Reference card RETRY / DON'T RETRY (giữ cố định trong suốt session) | Concept 2, Session 2 | ✅ |
| 6 | API key: dẫn bằng `.env` khuyến nghị; hard-coding chỉ còn là cảnh báo 1 dòng | Concept 1, Session 1 | ✅ |
| 7 | Cấu trúc session: Part A — Theory 2h + Part B — Lab 2h (tổng 4h/tuần lý thuyết) | Header + bảng Session Structure | ✅ |
| 8 | Chuỗi fading: worked example → code-along (guided) → lab độc lập → peer review | Cả 2 session | ✅ |
| 9 | Mock module dự phòng cho SV thiếu API key (không tạo blocking dependency) | Before Class, Session 1 | ✅ |
| 10 | Bảng "Common mistakes" xử lý trước các lỗi dự đoán | Cuối file | ✅ |
| 11 | KCs cung cấp vòng phản hồi trong Part A (KC1 sau Concept 1, KC2 sau Concept 3) | Part A, cả 2 session | ✅ |
| 12 | Hoạt động elaborative/retrieval đặt đúng chỗ (lab opener / homework) | Wrap & Reflect + file companion | ✅ |

**Cách kiểm tra nhanh (grep):**

```bash
cd syllabus/week2
# Các dấu hiệu CLT phải hiện diện:
grep -c "Stage A\|Stage B\|Predict-the-output\|Persistent reference card\|Step 1 (primary\|temperature = 0.0\|PART A — THEORY (2h)" lecture_notes.md
# Tài liệu gốc phân tích:
# cognitive_load_analysis.md — 6 đề xuất (P1–P6) đã được áp dụng đầy đủ (xem mục "Summary of Recommended Changes")
```