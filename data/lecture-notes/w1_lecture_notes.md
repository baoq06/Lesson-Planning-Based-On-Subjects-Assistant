# Week 1 — Lecture Notes (Instructor Guide)

## Python Data Structures for AI

> **Course:** AI Programming with Python
> **Week:** 1 of 8 — 2 sessions/week × 4 hours (2h theory + 2h lab per session; theory total 4h/week)
> **Prerequisites:** Python basics (variables, functions, loops, conditionals, file I/O); OOP in C++
> **Learning Objectives:**
> - A1.1 Select the appropriate data type (`list`, `tuple`, `dict`, `set`) for a given AI data-processing task and justify the choice.
> - A1.2 Read and write JSON data using Python's `json` module, including custom object serialisation.
> - A1.3 Define a `@dataclass` with type hints and default values to model structured AI data.
> - A1.4 Transform a list of `@dataclass` instances into a JSON-serialisable dictionary.

---

## Session 1 — Lists, Tuples, and Dictionaries: AI Patterns

> **Session format:** 4 hours total = **Part A — Theory (2h)** (this guide) + **Part B — Lab (2h)** (`lab_sheet.md`).

### Session Structure

| # | Segment | Duration | Gagné Event | Activity |
|---|---|---|---|---|
| — | **PART A — THEORY (2h)** | 0:00–2:00 | — | Segments 1–3, 5–6, 8–9, 12–13 |
| 1 | Hook & Problem | 0:00–0:10 | 1. Gain attention | Live demo: LLM movie recommendations |
| 2 | Objectives & Recall | 0:10–0:20 | 2–3. Objectives + Recall | State outcomes; activate prior Python knowledge |
| 3 | Concept 1: Lists for AI | 0:20–0:45 | 4. Present content | Worked example: list of embeddings, chat messages |
| 5 | KC 1 | 0:45–0:50 | 7. Feedback | MCQ: tuple vs list for embedding vector |
| 6 | Concept 2: Dicts as AI Records | 0:50–1:15 | 4. Present content | Worked example: LLM API response dict |
| — | Break | 1:15–1:20 | — | — |
| 9 | Concept 3: Sets & Tuples | 1:20–1:40 | 4. Present content | Worked example: dedup, composite keys |
| 12 | KC 2 | 1:40–1:45 | 8. Assess | MCQ: tuple as dict key |
| 13 | Wrap & Reflect | 1:45–2:00 | 9. Retention/Transfer | 1-paragraph reflection; preview Session 2 |
| — | **PART B — LAB (2h)** | 2:00–4:00 | — | Segments 4, 7, 10–11 — see `lab_sheet.md` |
| 4 | Code Along 1 (guided) | 2:00–2:30 | 5–6. Guidance + Practice | Load, filter, sort a list of dicts from JSON (lab_sheet Ex 1.1) |
| 7 | Code Along 2 (guided) | 2:30–2:55 | 5–6. Guidance + Practice | Parse mock API response, handle missing keys (lab_sheet Ex 1.2) |
| 10 | Independent Lab | 2:55–3:35 | 6. Elicit performance | Product review data pipeline (lab_sheet Ex 1.3 — 40 min) |
| 11 | Peer Review | 3:35–3:50 | 7. Feedback | Pair code review (lab_sheet Ex 1.4 — 15 min) |
| — | Buffer / questions | 3:50–4:00 | — | — |

---

### 1. Hook & Problem (0:00–0:10)

**Instructor script:**

> "Look at this output. I asked an LLM: 'Recommend 5 movies for someone who likes The Matrix.' Here's what it returned:"
>
> ```python
> recommendations = [
>     {"title": "Inception", "year": 2010, "genre": "Sci-Fi", "rating": 8.8},
>     {"title": "Blade Runner 2049", "year": 2017, "genre": "Sci-Fi", "rating": 8.0},
>     {"title": "Ghost in the Shell", "year": 1995, "genre": "Anime", "rating": 8.0},
>     {"title": "The Thirteenth Floor", "year": 1999, "genre": "Sci-Fi", "rating": 7.3},
>     {"title": "Dark City", "year": 1998, "genre": "Sci-Fi", "rating": 7.6}
> ]
> ```
>
> "Now I want to: (1) sort by rating descending, (2) filter only Sci-Fi, (3) send the top 3 to another service as JSON. **How would you store this data in Python?** "
>
> "This is not just an exercise — this exact pattern appears in every AI application. Every LLM API call returns nested dicts. Every vector database query returns lists of floats. Every RAG pipeline processes lists of document chunks. **By the end of this session, you'll be able to write this data pipeline confidently.** "

**Cognitive load note:** The hook creates a concrete mental model before any abstraction. Students see the *problem* before the *solution* (Merrill's Problem-Centered Principle).

---

### 2. Objectives & Recall (0:10–0:20)

**State objectives:**

> "By the end of this session, you will be able to:
> 1. Choose the right Python data type for any AI data scenario.
> 2. Process lists of AI records (filter, sort, transform).
> 3. Navigate nested dicts like LLM API responses.
> 4. Use sets for deduplication and tuples for composite keys."

**Activate prior knowledge (recall prompt):**

> "Quick recall — in Python:
> - What's the difference between `list` and `tuple`? (mutability)
> - When have you used `dict` before? (storing key-value pairs)
> - What does `set` do? (unique elements, O(1) membership)
> - How do you write a list comprehension? `[x**2 for x in range(10)]`"

**Expected student knowledge:** Students have used all four types in introductory Python. The gap is knowing *which* to use in AI-specific scenarios, not the syntax itself.

---

### 3. Concept 1 — Lists for AI Pipelines (0:20–0:45)

#### Worked Example: AI Data as Lists

**Instructor presents these three scenarios on the projector:**

```python
# Scenario 1: A list of embeddings (each is a vector of 384 floats)
embeddings = [
    [0.023, -0.045, 0.112, ...],  # 384 floats
    [0.067, 0.001, -0.089, ...],
]  # list of lists

# Scenario 2: A list of chat messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation..."}
]  # list of dicts

# Scenario 3: A list of document chunks
chunks = [
    {"id": "chunk_001", "text": "Python is a programming language...", "tokens": 120},
    {"id": "chunk_002", "text": "Lists are mutable sequences...", "tokens": 85},
]  # list of dicts
```

**Key insight:** "In AI programming, almost everything is a list of something. Lists are the *container type* — what you put inside depends on the problem."

**Worked example — list operations:**

```python
# Starting data
reviews = [
    {"product": "Laptop", "rating": 4.5, "text": "Great battery life"},
    {"product": "Mouse",  "rating": 3.0, "text": "Works but noisy"},
    {"product": "Laptop", "rating": 4.0, "text": "Good value"},
    {"product": "Keyboard", "rating": 5.0, "text": "Excellent"},
    {"product": "Mouse",  "rating": 2.0, "text": "Stopped working"},
]

# Filter: only products with rating >= 4.0
good_reviews = [r for r in reviews if r["rating"] >= 4.0]
# Result: [Laptop 4.5, Laptop 4.0, Keyboard 5.0]

# Sort by rating descending (highest first)
sorted_reviews = sorted(reviews, key=lambda r: r["rating"], reverse=True)
# Result: [Keyboard 5.0, Laptop 4.5, Laptop 4.0, Mouse 3.0, Mouse 2.0]

# Map: extract only product names
products = [r["product"] for r in reviews]
# Result: ['Laptop', 'Mouse', 'Laptop', 'Keyboard', 'Mouse']

# Chain: filter → sort → extract
result = sorted(
    [r for r in reviews if r["rating"] >= 4.0],
    key=lambda r: r["rating"],
    reverse=True
)
# Result: [Keyboard 5.0, Laptop 4.5, Laptop 4.0]
```

**Teaching tip:** Walk through each operation line by line. Show the input and output of each step. Don't assume students remember `sorted()` with `key` — many only know `.sort()`.

**Cognitive load note:** This is Concept 1 (lowest intrinsic load). Students already know list syntax. The new element is *which* operations to chain and *why* — this is germane load (schema building), not extraneous.

---

### 4. Code Along 1 — Load, Filter, Sort a List of Dicts (2:00–2:30) [PART B — LAB]

**Instructor writes this code line by line. Students type along.**

```python
import json

# Step 1: Load data from a JSON file
with open("data/reviews.json", "r") as f:
    reviews = json.load(f)

print(f"Loaded {len(reviews)} reviews")

# Step 2: Filter — only reviews with text
with_text = [r for r in reviews if r.get("text")]
print(f"Reviews with text: {len(with_text)}")

# Step 3: Sort by rating (descending)
top_rated = sorted(with_text, key=lambda r: r["rating"], reverse=True)

# Step 4: Show top 3
print("Top 3 reviews:")
for r in top_rated[:3]:
    print(f"  {r['product']}: {r['rating']} — {r['text'][:50]}...")

# Step 5: Save filtered results
with open("data/top_reviews.json", "w") as f:
    json.dump(top_rated[:5], f, indent=2)
```

**Common mistakes to watch for:**

| Mistake | Correction |
|---|---|
| `r["text"]` on missing key → KeyError | Use `r.get("text", "")` |
| `reviews.sort()` on loaded JSON (list of dicts) | `sorted()` returns new list; `.sort()` sorts in-place |
| `json.dump()` without `indent=2` → unreadable one-liner | Always use `indent=2` for debugging |
| Forgetting `f.close()` → use `with open(...)` | Context manager handles it |

**Check for understanding:** "Pause. Everyone should see 'Loaded N reviews' printed. If not, raise your hand."

---

### 5. Knowledge Check 1 (0:45–0:50)

**MCQ — Project on screen:**

> You are storing an embedding vector — 384 floating-point numbers — that represents the semantic meaning of a sentence. This vector will never change after creation.
>
> **Which data type should you use?**
>
> (a) `list` — [0.023, -0.045, 0.112, ...]
> (b) `tuple` — (0.023, -0.045, 0.112, ...)
> (c) `set` — {0.023, -0.045, 0.112, ...}

**Correct answer: (b) tuple**

**Justification:** The vector is *fixed-length* and *immutable* — it never changes. `tuple` guarantees immutability, which prevents accidental modification and allows the vector to be used as a dict key or stored in a `set`. A `list` would work but signals "this might change" to other developers. A `set` discards duplicates and has no ordering, which is wrong for a vector.

**Pitfall note (for instructor):** Many students will choose `list` because they're more familiar with it. Use this teachable moment to discuss *semantic intent*: "list says 'this is a sequence that may change'; tuple says 'this is a fixed record.' AI embeddings are fixed records."

---

### 6. Concept 2 — Dicts as AI Records (0:50–1:15)

#### Worked Example: LLM API Response

**Show this real (simplified) OpenAI API response:**

```python
response = {
    "id": "chatcmpl-9a8b7c6d5e",
    "object": "chat.completion",
    "created": 1712345678,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "RAG stands for Retrieval-Augmented Generation. It's a technique..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 45,
        "completion_tokens": 120,
        "total_tokens": 165
    }
}
```

**Key insight:** "This nested dict structure is the *universal interface* to every AI API on the planet. Learn to navigate it, and you can use any AI service."

**Worked example — dict navigation:**

```python
# Safe access with .get()
content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
print(content)

# Extract specific fields
model = response["model"]
tokens = response["usage"]["total_tokens"]
finish = response["choices"][0]["finish_reason"]

print(f"Model: {model}")
print(f"Tokens: {tokens}")
print(f"Finish reason: {finish}")

# Dict merging (useful for combining configs)
default_config = {"temperature": 0.7, "max_tokens": 1000}
user_config = {"temperature": 0.0}
final_config = {**default_config, **user_config}
# Result: {"temperature": 0.0, "max_tokens": 1000}
# Note: user_config.temperature overrides default_config.temperature
```

**Teaching tip:** The `[{}]` fallback pattern (`response.get("choices", [{}])`) is a professional idiom. Explain: "If 'choices' is missing, we get a list with one empty dict, so the next `.get()` still works instead of crashing."

**Alternative pattern (Python 3.9+):**

```python
# Using the merge operator |=
default_config = {"temperature": 0.7, "max_tokens": 1000}
user_config = {"temperature": 0.0}
final_config = default_config | user_config  # Python 3.9+
```

---

### 7. Code Along 2 — Parse Mock API Response (2:30–2:55) [PART B — LAB]

```python
import json
from datetime import datetime

# Mock API response (simulating what OpenAI returns)
mock_response = {
    "id": "chatcmpl-mock-001",
    "object": "chat.completion",
    "created": int(datetime.now().timestamp()),
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I'm an AI assistant. How can I help you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 25,
        "completion_tokens": 15,
        "total_tokens": 40
    }
}

# Task 1: Extract the assistant's message
def extract_content(response):
    """Safely extract the assistant's message content."""
    choices = response.get("choices", [])
    if not choices:
        return "No response"
    message = choices[0].get("message", {})
    return message.get("content", "Empty message")

# Task 2: Extract token usage
def extract_usage(response):
    """Extract and print token usage statistics."""
    usage = response.get("usage", {})
    return {
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0)
    }

# Task 3: Handle missing keys gracefully
def safe_extract(response, key, default=None):
    """Safely extract a nested value using dot-notation path."""
    keys = key.split(".")
    current = response
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return default
    return current if current is not None else default

# Test the functions
print("Message:", extract_content(mock_response))
print("Usage:", extract_usage(mock_response))
print("Safe 'choices.0.message.content':",
      safe_extract(mock_response, "choices.0.message.content"))
print("Safe 'choices.0.message.nonexistent':",
      safe_extract(mock_response, "choices.0.message.nonexistent"))
print("Safe 'missing.path':",
      safe_extract(mock_response, "missing.path"))
```

**Common mistakes to watch for:**

| Mistake | Correction |
|---|---|
| `response["choices"][0]` without checking if list is empty | Always check `if choices:` first |
| Assuming `finish_reason` is always `"stop"` | Could be `"length"`, `"content_filter"`, `"null"` |
| Hard-coding key paths | Use parameterised `safe_extract` pattern |

---

### 8. Break (1:15–1:20)

**Instructor tip:** Walk around during the break and answer individual questions.

---

### 9. Concept 3 — Sets & Tuples for AI Workflows (1:20–1:40)

#### Worked Example: Deduplication with Sets

```python
# Scenario: You have a list of document IDs from multiple sources
# Some IDs appear more than once — you need unique ones

doc_ids = ["doc_001", "doc_002", "doc_001", "doc_003", "doc_002", "doc_004"]

# Inefficient O(n²) approach
unique = []
for id_ in doc_ids:
    if id_ not in unique:
        unique.append(id_)

# Efficient O(n) approach using set
unique = list(set(doc_ids))
print(unique)  # ['doc_001', 'doc_002', 'doc_003', 'doc_004']

# Preserving order (Python 3.7+ dicts preserve insertion order)
unique_ordered = list(dict.fromkeys(doc_ids))
print(unique_ordered)  # ['doc_001', 'doc_002', 'doc_003', 'doc_004']
```

**Key insight:** "`set` gives you O(1) membership testing. For 10,000 items, `if x in set` is instant; `if x in list` takes 10,000 comparisons."

#### Worked Example: Tuples as Composite Keys

```python
# Scenario: You need to cache search results per (user_id, session_id)
# A dict key must be immutable — tuples work perfectly

cache = {}

# Composite key: (user_id, session_id)
key = ("user_42", "session_abc123")
cache[key] = {"query": "What is RAG?", "result": "RAG stands for..."}

# Retrieve
result = cache.get(("user_42", "session_abc123"))
print(result)  # {'query': 'What is RAG?', 'result': 'RAG stands for...'}

# List of tuples as a structured dataset
top_products = [
    ("Laptop", 4.5, 120),
    ("Keyboard", 4.2, 85),
    ("Mouse", 3.8, 200),
]

# Unpacking tuples
for product, rating, count in top_products:
    print(f"{product}: {rating} ({count} reviews)")
```

**Key insight:** "Use tuples when the data structure is *fixed* — a record with a known number of fields. Tuples are the lightest way to group related data without defining a class."

---

### 10. Independent Lab — Data Pipeline (2:55–3:35) [PART B — LAB]

Students work independently. See `lab_sheet.md` for full instructions.

**Task card (project on screen so students don't switch documents):**
> 1. Load `data/reviews.json`.
> 2. Deduplicate by `review_id` (use a `set`, keep first occurrence).
> 3. Filter out reviews with empty `text`.
> 4. Top 5 products by average rating → `(product, avg_rating)` tuples.
> 5. Save to `data/top_5_products.json`.

**Setup checklist (before class):**
- [ ] Create `data/reviews.json` with 20+ mock product reviews (see `lab_sheet.md` for content)
- [ ] Verify the file is accessible from student machines
- [ ] Test the solution yourself

**Teaching tips during lab:**
- Circulate. Look for students who are stuck on the first step.
- Common blocker: "I can't find the file" → check working directory, use absolute path
- Common blocker: "My sort is wrong" → `reverse=True` for descending
- Bonus: "What if two products have the same average rating?" → sort by name as tiebreaker

---

### 11. Peer Review (3:35–3:50) [PART B — LAB]

**Pair up students. Review checklist (project on screen):**

> **Code Review Checklist:**
> 1. Does the deduplication work? (run with duplicate review_ids)
> 2. Are edge cases handled? (empty file, missing fields)
> 3. Is the code readable? (variable names, comments)
> 4. Does the output match the expected format?

**2 min per person, then swap.**

---

### 12. Knowledge Check 2 (1:40–1:45)

**MCQ:**

> You're building a cache that maps `(user_id, session_id)` to session data. The cache must be fast and the keys must never change.
>
> **Which type should you use for the key?**
>
> Please explain your answer in one sentence.

**Expected answer: `tuple` — `("user_42", "session_abc")`**

**Explanation:** Tuples are immutable (can be used as dict keys) and hashable (O(1) lookup). Lists are mutable and unhashable — they would raise `TypeError: unhashable type: 'list'`.

---

### 13. Wrap & Reflect (1:45–2:00)

**Instructor summary:**

> "Today we covered the four fundamental data types for AI programming:
> - **Lists** for sequences (embeddings, messages, chunks)
> - **Dicts** for records (API responses, configs, metadata)
> - **Sets** for uniqueness (deduplication, membership testing)
> - **Tuples** for fixed composite keys and lightweight records
>
> In Session 2, we'll take this further: JSON serialisation, @dataclass (the Pythonic way to model AI data), and type hints. You'll build a proper data model for a RAG document chunk."

**Reflection prompt (students write 1 paragraph, 3 min):**

> "What was the most useful thing you learned today? What confused you? Write 3–5 sentences."

**Collect reflections** — use them to adjust Session 2 pace.

**Optional — elaborative interrogation:** Run prompts P1–P4 from `elaborative_prompts.md` as the opener of the Part B lab session (first 15–20 min, before Code Along 1) or as homework; alternatively use them as the recall opener of Session 2.

---

## Session 2 — JSON, @dataclass, and Type Hints

> **Session format:** 4 hours total = **Part A — Theory (2h)** (this guide) + **Part B — Lab (2h)** (`lab_sheet.md`).

### Session Structure

| # | Segment | Duration | Gagné Event | Activity |
|---|---|---|---|---|
| — | **PART A — THEORY (2h)** | 0:00–2:00 | — | Segments 1–3, 5–6, 10–11, 14–15 |
| 1 | Hook | 0:00–0:10 | 1. Gain attention | Raw JSON → clean Python objects |
| 2 | Objectives & Recall | 0:10–0:20 | 2–3. Objectives + Recall | Recap Session 1 dicts |
| 3 | Concept 1: JSON in AI | 0:20–0:45 | 4. Present content | Worked example: json.loads/dumps, custom serialisation |
| 5 | KC 1 | 0:45–0:50 | 7. Feedback | MCQ: json.dumps on datetime |
| 6 | Concept 2: @dataclass | 0:50–1:15 | 4. Present content | Worked example: Document, ChatMessage |
| — | Break | 1:15–1:20 | — | — |
| 11 | Concept 3: Type Hints | 1:20–1:40 | 4. Present content | Function signatures; TypedDict (optional/enrichment) |
| 14 | KC 2 | 1:40–1:45 | 8. Assess | MCQ: @dataclass decorator |
| 15 | Wrap & Reflect | 1:45–2:00 | 9. Retention/Transfer | Preview Week 2 |
| — | **PART B — LAB (2h)** | 2:00–4:00 | — | Segments 4, 7–9, 12–13 — see `lab_sheet.md` |
| 4 | Code Along 1 (guided) | 2:00–2:30 | 5–6. Guidance + Practice | Read JSON file, filter, write back (lab_sheet Ex 2.1) |
| 7 | Code Along 2a (guided) | 2:30–2:45 | 5–6. Guidance + Practice | EmbeddingResult dataclass + asdict + json.dump (lab_sheet Ex 2.2) |
| 8 | Stretch Break | 2:45–2:50 | — | 5-min stretch, hydrate |
| 9 | Code Along 2b (mini-demo) | 2:50–3:05 | 5–6. Guidance + Practice | Custom JSON encoder for @dataclass |
| 12 | Independent Lab | 3:05–3:45 | 6. Elicit performance | DocumentChunk model + JSON round-trip (lab_sheet Ex 2.3 — 40 min) |
| 13 | Peer Review | 3:45–4:00 | 7. Feedback | Pair code review (lab_sheet Ex 2.4 — 15 min) |

---

### 1. Hook — Raw JSON → Clean Objects (0:00–0:10)

**Instructor script:**

> "This is the actual response from the OpenAI API. Look at this raw JSON:"
>
> ```json
> {
>   "id": "chatcmpl-AbCdEf123456",
>   "object": "chat.completion",
>   "created": 1712345678,
>   "model": "gpt-4o-mini",
>   "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
>   "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
> }
> ```
>
> "Working with raw dicts is error-prone. You have to remember key names, handle missing keys manually, and your IDE gives you no autocomplete. What if we could convert this to a clean Python object — with attributes, autocomplete, and validation?
>
> "That's what `@dataclass` does. In this session, you'll learn to convert messy JSON into clean, type-safe Python objects. Every AI developer does this every day."

---

### 2. Objectives & Recall (0:10–0:20)

**State objectives:**

> "By the end of this session, you will be able to:
> 1. Read and write JSON with custom serialisation.
> 2. Define a `@dataclass` to model AI data structures.
> 3. Use type hints to make your AI code self-documenting.
> 4. Convert between `@dataclass` instances and JSON."

**Recall prompt:**

> "From Session 1: How do you access a nested dict value? What's the problem with using bare dicts everywhere in a large codebase? (No autocomplete, no validation, manual key management.)"

---

### 3. Concept 1 — JSON in AI Workflows (0:20–0:45)

#### Worked Example: JSON Serialisation

```python
import json
from datetime import datetime

# Basic serialisation
data = {
    "name": "AI Assistant",
    "version": 2.0,
    "features": ["chat", "search", "code"],
    "is_active": True,
    "metadata": None
}

# Serialise to JSON string
json_str = json.dumps(data, indent=2)
print(json_str)

# Write to file
with open("config.json", "w") as f:
    json.dump(data, f, indent=2)

# Read from file
with open("config.json", "r") as f:
    loaded = json.load(f)

print(loaded["name"])  # AI Assistant
```

#### Custom Serialisation — The Problem

```python
# This will FAIL
data_with_date = {
    "name": "Chat Log",
    "created_at": datetime.now(),
    "messages": 5
}

# json.dumps(data_with_date)  # TypeError: Object of type datetime is not JSON serialisable
```

**Worked Example — Custom Encoder:**

```python
from datetime import datetime
import json

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO 8601 string
        if isinstance(obj, set):
            return list(obj)  # Convert set to list
        return super().default(obj)  # Fallback to default

data = {
    "name": "Chat Log",
    "created_at": datetime.now(),
    "tags": {"urgent", "follow-up"}  # set
}

json_str = json.dumps(data, cls=CustomEncoder, indent=2)
print(json_str)
```

**Key insight:** "JSON has 6 types (string, number, boolean, null, object, array). Python has many more (datetime, set, Decimal, etc.). When you encounter a non-serialisable type, you write a custom encoder — or you convert to a serialisable type first."

---

### 4. Code Along 1 — Read, Filter, Write JSON (2:00–2:30) [PART B — LAB]

```python
import json
from datetime import datetime

# Step 1: Read a JSON file
with open("data/chat_messages.json", "r") as f:
    messages = json.load(f)

print(f"Loaded {len(messages)} messages")

# Step 2: Filter — only assistant messages
assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
print(f"Assistant messages: {len(assistant_msgs)}")

# Step 3: Pretty-print a single message
for msg in assistant_msgs[:2]:
    print(f"[{msg['role']}] {msg['content'][:60]}...")

# Step 4: Write filtered results back to JSON
with open("data/assistant_messages.json", "w") as f:
    json.dump(assistant_msgs, f, indent=2)

# Step 5: Read it back and verify
with open("data/assistant_messages.json", "r") as f:
    verified = json.load(f)
print(f"Verified: {len(verified)} messages saved")
```

---

### 5. Knowledge Check 1 (0:45–0:50)

**MCQ:**

> You call `json.dumps()` on a Python `datetime` object. What happens?
>
> (a) It works — `datetime` is automatically serialised.
> (b) TypeError — `datetime` is not JSON serialisable by default.
> (c) It auto-converts to a Unix timestamp.

**Correct answer: (b) TypeError**

**Explanation:** Python's `datetime` has no default JSON representation. You must provide a custom encoder (or convert to string first). This is a common gotcha that students will encounter repeatedly when logging AI API calls with timestamps.

---

### 6. Concept 2 — @dataclass for AI Models (0:50–1:15)

#### The Problem: Manual Class Boilerplate

```python
class Document:
    def __init__(self, content, metadata, doc_id):
        self.content = content
        self.metadata = metadata
        self.doc_id = doc_id

    def __repr__(self):
        return f"Document(content={self.content[:20]}..., doc_id={self.doc_id})"

    def __eq__(self, other):
        if not isinstance(other, Document):
            return False
        return self.doc_id == other.doc_id

# That's a lot of code for a simple data container
```

#### The Solution: @dataclass

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class Document:
    """A document to be indexed and searched."""
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ChatMessage:
    """A single message in a chat conversation."""
    role: str          # "system", "user", "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
```

**@dataclass auto-generates:** `__init__`, `__repr__`, `__eq__`, `__hash__` (if frozen)

**Key insight:** "`@dataclass` is the standard way to model AI data in Python. It replaces the Session 1 pattern of 'dict with string keys' with a proper typed object. Your IDE now knows `doc.content` is a string, `doc.doc_id` is a string, and `doc.metadata` is a dict."

#### Important: Mutable Defaults

```python
from dataclasses import dataclass, field

# WRONG — @dataclass rejects this at class definition:
# ValueError: mutable default <class 'list'> for field tags is not allowed
# (The danger it prevents: all instances would share the same list.)
@dataclass
class BadExample:
    tags: list = []  # Don't do this!

# RIGHT — each instance gets a new list
@dataclass
class GoodExample:
    tags: list = field(default_factory=list)
```

**Teaching tip:** Run the `BadExample` class definition live — `@dataclass` refuses it outright with `ValueError: mutable default <class 'list'> for field tags is not allowed: use default_factory`. This is Python protecting you: the default `[]` is evaluated ONCE at class-definition time, so if it were allowed, every instance would share the same list object. To show the actual bug the ValueError prevents, demo it with a plain class instead:

```python
class Shared:
    tags = []          # class attribute — evaluated once, shared by all
    def add(self, tag):
        self.tags.append(tag)

a, b = Shared(), Shared()
a.add("urgent")
print(b.tags)  # ['urgent'] — b sees a's tag!
```

Then show the fix: `field(default_factory=list)` runs a fresh factory per instance, so each object gets its own list.

---

### 7. Code Along 2a — EmbeddingResult + JSON Dump (2:30–2:45) [PART B — LAB]

**Instructor writes this code line by line. Students type along.** Build the dataclass first; the custom encoder comes in the next segment.

```python
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json

@dataclass
class EmbeddingResult:
    """The result of embedding a text with an embedding model."""
    vector: list[float]          # The embedding vector (e.g., 384 floats)
    model: str = "text-embedding-3-small"
    token_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    text: str = ""

# Create instances
result1 = EmbeddingResult(
    vector=[0.023, -0.045, 0.112, 0.067],
    model="text-embedding-3-small",
    token_count=10,
    text="What is RAG?"
)

result2 = EmbeddingResult(
    vector=[0.089, 0.001, -0.034, 0.055],
    model="text-embedding-3-small",
    token_count=8,
    text="Python programming"
)

# Convert instances to plain dicts with asdict
results = [asdict(result1), asdict(result2)]

# Now try serialising — this FAILS:
# json_str = json.dumps(results, indent=2)
# TypeError: Object of type datetime is not JSON serialisable
```

**Check for understanding:** "Pause. Everyone should have two `EmbeddingResult` instances and see the `asdict` output. What happens if we call `json.dumps` directly? (TypeError — datetime again.) That's the problem we solve next."

---

### 8. Stretch Break (2:45–2:50) [PART B — LAB]

**Instructor tip:** 5-minute stretch and hydrate between Code Along 2a and 2b — resets attention before the encoder demo.

---

### 9. Code Along 2b — Custom JSON Encoder (Mini-Demo, 2:50–3:05) [PART B — LAB]

**Instructor demonstrates; students can type along or watch.** Builds on the working dataclass from Code Along 2a — paste it above if you're in a new file.

```python
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json

# Reuse the EmbeddingResult dataclass from Code Along 2a
@dataclass
class EmbeddingResult:
    """The result of embedding a text with an embedding model."""
    vector: list[float]
    model: str = "text-embedding-3-small"
    token_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    text: str = ""

# Custom JSON encoder for @dataclass
class DataclassEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        return super().default(obj)

# Serialise to JSON
result1 = EmbeddingResult(vector=[0.023, -0.045, 0.112, 0.067], token_count=10, text="What is RAG?")
result2 = EmbeddingResult(vector=[0.089, 0.001, -0.034, 0.055], token_count=8, text="Python programming")
results = [result1, result2]
json_str = json.dumps(results, cls=DataclassEncoder, indent=2)
print(json_str)

# Save to file
with open("data/embeddings.json", "w") as f:
    json.dump(results, f, cls=DataclassEncoder, indent=2)
```

**Expected output:**

```json
[
  {
    "vector": [0.023, -0.045, 0.112, 0.067],
    "model": "text-embedding-3-small",
    "token_count": 10,
    "created_at": "2026-08-04T10:30:00",
    "text": "What is RAG?"
  },
  {
    "vector": [0.089, 0.001, -0.034, 0.055],
    "model": "text-embedding-3-small",
    "token_count": 8,
    "created_at": "2026-08-04T10:30:01",
    "text": "Python programming"
  }
]
```

**Teaching tip:** Anchor the encoder to Concept 1's `CustomEncoder`: `json.dumps` calls `default()` whenever it meets a type it doesn't know. `isinstance` handles `datetime` → ISO string, `asdict` converts dataclasses → dicts, `super().default(obj)` is the fallback. Same structure, one new branch.

---

### 10. Break (1:15–1:20)

---

### 11. Concept 3 — Type Hints for AI Code (1:20–1:40)

#### Worked Example: Annotated Functions

```python
from __future__ import annotations  # must be the first line of the file

from dataclasses import dataclass, field

@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

# Type hints on function signatures
def search_documents(
    query: str,
    top_k: int = 5,
    min_score: float = 0.7,
    filter_by: dict | None = None
) -> list[tuple[Document, float]]:
    """Search documents and return (document, score) pairs."""
    # ... implementation
    results: list[tuple[Document, float]] = []
    return results

def process_messages(
    messages: list[dict[str, str]]
) -> dict[str, int]:
    """Count messages by role. Returns {'user': 3, 'assistant': 5}."""
    counts: dict[str, int] = {}
    for msg in messages:
        role = msg.get("role", "unknown")
        counts[role] = counts.get(role, 0) + 1
    return counts
```

**Teaching tip:** This example uses the builtin generic style (`list[...]`, `dict[...]`, `tuple[...]`) with `from __future__ import annotations` — the same style as Code Along 2a. Keep one style across the course; the older `typing.List[...]` form still works but mixes two conventions for no benefit.

#### TypedDict for API Response Shapes (optional / enrichment)

```python
from typing import TypedDict, Any

class ChatCompletionResponse(TypedDict):
    """Shape of the OpenAI chat completion response."""
    id: str
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]

class Message(TypedDict):
    role: str
    content: str

class ChatCompletionChoice(TypedDict):
    index: int
    message: Message
    finish_reason: str
```

**Optional:** `TypedDict` is not required for the labs — it models the same shapes as `@dataclass`. If the session is running long, skip this block and mention it as a "read on your own" topic.

**Key insight:** "Type hints are not enforced at runtime — they're for *humans and tools*. Your IDE uses them for autocomplete, type checkers (mypy, pyright) catch bugs before runtime, and future developers (including you in 6 months) can understand the code without reading the implementation."

---

### 12. Independent Lab — Data Model Design (3:05–3:45) [PART B — LAB]

See `lab_sheet.md` for full instructions.

**Task card (project on screen so students don't switch documents):**
> 1. Define `DocumentChunk` @dataclass — `embedding: list[float] | None`.
> 2. Read `data/document_chunks.json` → convert each dict to a `DocumentChunk`.
> 3. Print all chunks where `embedding` is `None`.
> 4. Add `char_count` from `len(content)`.
> 5. Serialise back to JSON with `DataclassEncoder` → `data/document_chunks_updated.json`.

**Setup checklist:**
- [ ] Create `data/document_chunks.json` with 5+ mock chunks (see `lab_sheet.md`)
- [ ] Test the solution

**Common issues to watch for:**
- `embedding: list[float] | None` — students may forget to import `from __future__ import annotations` or use `Optional[list[float]]`
- Mutable defaults: `chunks: list = []` — must use `field(default_factory=list)`
- JSON deserialisation: `json.load()` returns dicts, not @dataclass instances — they must convert manually

---

### 13. Peer Review (3:45–4:00) [PART B — LAB]

**Checklist on screen:**

> 1. Are mutable defaults handled with `field(default_factory=...)`?
> 2. Is `None` used properly for optional embedding (not `[]` or `0`)?
> 3. Does the JSON round-trip work? (load → modify → save → load)
> 4. Are type hints present on all fields?

---

### 14. Knowledge Check 2 (1:40–1:45)

**MCQ:**

> Which decorator automatically generates `__init__`, `__repr__`, and `__eq__` for a class?
>
> (a) `@staticmethod`
> (b) `@dataclass`
> (c) `@property`

**Correct answer: (b) @dataclass**

**Explanation:** `@dataclass` generates the boilerplate for data-holding classes. `@staticmethod` defines a static method. `@property` creates a getter/setter for an attribute.

---

### 15. Wrap & Reflect (1:45–2:00)

**Instructor summary:**

> "This week you built the foundation for everything that follows:
> - **Session 1:** You can process AI data with lists, dicts, sets, and tuples.
> - **Session 2:** You can model AI data with @dataclass, serialise to JSON, and add type hints.
>
> Next week: we send this data to an LLM API and get responses back. You'll use your @dataclass skills to model the API request/response, your JSON skills to serialise/deserialise, and your dict skills to navigate the response.
>
> **Everything connects.** "

**Reflection prompt (3 min):**

> "What's the one thing from today that you'll use most in your AI programming? What's still unclear? Write 3–5 sentences."

**Preview Lab 1 (due before Session 3):** "Your homework is to build a `ProductReview` @dataclass and a data pipeline — combining everything from both sessions. It's in the lab sheet."

**Optional — elaborative interrogation:** Run prompts P5–P8 and connection prompts C1–C3 from `elaborative_prompts.md` as the opener of the Part B lab session or as homework before Lab 1 (15–20 min, individual → pairs → class share).

---

## Instructor Preparation Checklist

### Before Session 1

- [ ] Prepare demo: show LLM movie recommendations (in Hook)
- [ ] Create `data/reviews.json` with 20+ mock reviews
- [ ] Test all Code Along scripts
- [ ] Test the Independent Lab solution
- [ ] Prepare KC slides (3-option MCQs)
- [ ] Ensure Python 3.11+ is installed on all machines
- [ ] Ensure `pip install` is working (no extra packages needed for Week 1)

### Before Session 2

- [ ] Create `data/chat_messages.json` with 10+ mock messages
- [ ] Create `data/document_chunks.json` with 5+ mock chunks
- [ ] Test all Code Along scripts
- [ ] Test the Independent Lab solution
- [ ] Review Session 1 reflections to adjust pace

### Sample Data Files

#### `data/reviews.json` (for Session 1 Lab)

```json
[
  {"review_id": "R001", "product": "Laptop", "rating": 5, "text": "Excellent performance"},
  {"review_id": "R002", "product": "Mouse", "rating": 3, "text": "Works but noisy"},
  {"review_id": "R003", "product": "Keyboard", "rating": 4, "text": "Good feel"},
  {"review_id": "R004", "product": "Laptop", "rating": 4, "text": "Good battery life"},
  {"review_id": "R005", "product": "Monitor", "rating": 5, "text": "Crystal clear display"},
  {"review_id": "R001", "product": "Laptop", "rating": 5, "text": "Excellent performance"},
  {"review_id": "R006", "product": "Mouse", "rating": 2, "text": "Stopped after 1 month"},
  {"review_id": "R007", "product": "Keyboard", "rating": 5, "text": "Love the mechanical switches"},
  {"review_id": "R008", "product": "Monitor", "rating": 4, "text": "Good value for money"},
  {"review_id": "R009", "product": "Laptop", "rating": 3, "text": ""},
  {"review_id": "R010", "product": "Mouse", "rating": 4, "text": "Comfortable grip"},
  {"review_id": "R011", "product": "Headphones", "rating": 5, "text": "Amazing sound quality"},
  {"review_id": "R012", "product": "Headphones", "rating": 3, "text": "Average noise cancelling"},
  {"review_id": "R013", "product": "Laptop", "rating": 2, "text": "Overheats quickly"},
  {"review_id": "R014", "product": "Keyboard", "rating": 4, "text": "Nice RGB lighting"},
  {"review_id": "R015", "product": "Monitor", "rating": 3, "text": ""},
  {"review_id": "R016", "product": "Mouse", "rating": 5, "text": "Best mouse I've ever used"},
  {"review_id": "R017", "product": "Headphones", "rating": 4, "text": "Comfortable for long sessions"},
  {"review_id": "R018", "product": "Laptop", "rating": 4, "text": "Fast and reliable"},
  {"review_id": "R019", "product": "Keyboard", "rating": 3, "text": "Keys are a bit stiff"},
  {"review_id": "R020", "product": "Monitor", "rating": 5, "text": "Perfect for programming"}
]
```

#### `data/chat_messages.json` (for Session 2 Code Along 1)

```json
[
  {"role": "system", "content": "You are a helpful assistant.", "timestamp": "2026-08-04T09:00:00"},
  {"role": "user", "content": "What is Python?", "timestamp": "2026-08-04T09:00:05"},
  {"role": "assistant", "content": "Python is a high-level, interpreted programming language known for its readability.", "timestamp": "2026-08-04T09:00:07"},
  {"role": "user", "content": "What is RAG?", "timestamp": "2026-08-04T09:01:00"},
  {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation. It combines retrieval from a knowledge base with text generation.", "timestamp": "2026-08-04T09:01:03"},
  {"role": "user", "content": "Give me an example", "timestamp": "2026-08-04T09:01:30"},
  {"role": "assistant", "content": "A user asks about a company policy. The system retrieves relevant policy documents, then generates an answer based on them.", "timestamp": "2026-08-04T09:01:35"},
  {"role": "user", "content": "Thanks!", "timestamp": "2026-08-04T09:02:00"}
]
```

#### `data/document_chunks.json` (for Session 2 Independent Lab)

```json
[
  {"chunk_id": "C001", "doc_id": "DOC001", "content": "Python is a high-level programming language.", "chunk_index": 0, "embedding": null},
  {"chunk_id": "C002", "doc_id": "DOC001", "content": "It was created by Guido van Rossum in 1991.", "chunk_index": 1, "embedding": null},
  {"chunk_id": "C003", "doc_id": "DOC002", "content": "Lists are mutable sequences in Python.", "chunk_index": 0, "embedding": [0.023, -0.045, 0.112]},
  {"chunk_id": "C004", "doc_id": "DOC002", "content": "Tuples are immutable sequences.", "chunk_index": 1, "embedding": [0.067, 0.001, -0.089]},
  {"chunk_id": "C005", "doc_id": "DOC003", "content": "RAG combines retrieval and generation.", "chunk_index": 0, "embedding": null},
  {"chunk_id": "C006", "doc_id": "DOC003", "content": "The retriever finds relevant documents.", "chunk_index": 1, "embedding": null}
]
```

---

## Common Student Mistakes & Interventions

| Mistake | When It Appears | Intervention |
|---|---|---|
| Using `r["text"]` instead of `r.get("text")` | Code Along 1, S1 | Show KeyError, then demonstrate `.get()` |
| Forgetting `reverse=True` in `sorted()` | Code Along 1, S1 | "What's the default? Ascending. How do we flip it?" |
| `json.dumps()` on datetime → TypeError | Code Along 2, S2 | "JSON doesn't know datetime. What can we convert it to?" |
| Mutable default: `tags: list = []` | Concept 2, S2 | Demo the shared-list bug |
| `from dataclasses import dataclass` misspelling | Code Along 2, S2 | "Check your spelling — `dataclass` not `dataclasses`" |
| Assuming `@dataclass` fields are type-checked at runtime | Concept 3, S2 | "Type hints are for humans and tools, not Python runtime" |
| Using `list` as a variable name (shadowing) | Any lab | "`list` is a built-in — don't use it as a variable name" |
| Not handling the `embedding: None` case | Lab 2 | "Check for `None` before processing the vector" |

---

## KC Answer Key (for instructor reference)

### Session 1
- **KC1:** (b) `tuple` — fixed-length, immutable, hashable
- **KC2:** (b) `tuple` — immutable and hashable; `list` is unhashable

### Session 2
- **KC1:** (b) TypeError — datetime is not JSON-serialisable by default
- **KC2:** (b) `@dataclass` — auto-generates `__init__`, `__repr__`, `__eq__`