# Week 1: Data Structures and Typing in Python (Data Layer)

## 1. General Information & Learning Objectives (A1.1 to A1.4)
- **A1.1**: Choose the appropriate data type (`list`, `tuple`, `dict`, `set`) for an AI data-processing task and justify the choice.
- **A1.2**: Read and write JSON data with Python's `json` module, including Custom Object Serialization.
- **A1.3**: Define a `@dataclass` with full type hints and default values (`field(default_factory=...)`) to model strictly structured AI data.
- **A1.4**: Convert a list of `@dataclass` instances into dictionaries for two-way serialization (JSON serialization with `asdict`, `to_dict`, `from_dict`).

## 2. Foundational Theory & System Architecture
### 2.1. Comparing the core data structures in an AI pipeline
- **List vs Set**:
  - A `list` preserves insertion order, allows $O(1)$ access by index and stores duplicate elements; it suits token sequences or message arrays in a conversation context.
  - A `set` does not preserve order but is optimized for lookup and membership tests `x in s` with $O(1)$ complexity; it is commonly used to filter stop-words, store the vocabulary and perform deduplication.
- **Dict vs Dataclass**:
  - A raw `dict` is very flexible but has no enforced structure (schema-less), so it easily causes `KeyError` and has no IDE autocomplete.
  - A `@dataclass` provides Type Safety, automatically generates the special methods (`__init__`, `__repr__`, `__eq__`), supports Static Type Checking and offers the `__post_init__` validation hook.
- **Tuple**:
  - An immutable data structure; it is safe to use as a dictionary key (for example `(user_id, session_id)`), memory-efficient, and guarantees integrity when storing fixed coordinate vectors.

### 2.2. Defensive Data Validation
- The `__post_init__` method is called automatically right after the `@dataclass` initializer. It is an important checkpoint to:
  - Check that text is not empty (`content.strip()`).
  - Validate value ranges (for example week number `1 <= week <= 8`, rating `1 <= rating <= 5`).
  - Raise a `ValueError` immediately to stop bad data from going deeper into the RAG pipeline.

## 3. Lab Exercises & Step-by-step Sample Code
### Exercise 1.1 — Code Along 1: Load, Filter & Sort Reviews (ex1_1_filter_sort.py)
```python
import json

with open("data/reviews.json", "r", encoding="utf-8") as f:
    reviews = json.load(f)

# Keep only the reviews that have text
with_text = [r for r in reviews if r.get("text")]
print(f"Loaded: {len(reviews)}, With text: {len(with_text)}")

# Sort by rating, descending
top_rated = sorted(with_text, key=lambda r: r["rating"], reverse=True)
for r in top_rated[:3]:
    print(f"  {r['product']}: {r['rating']} - {r['text'][:50]}...")
```

### Exercise 1.2 — Code Along 2: Parse Mock API Response (ex1_2_parse_api.py)
```python
def extract_assistant_content(api_response: dict) -> str:
    choices = api_response.get("choices", [])
    if not choices:
        return "No response"
    return choices[0].get("message", {}).get("content", "")

def safe_extract_nested(data: dict, path: str, default=None):
    keys = path.split(".")
    curr = data
    for k in keys:
        if isinstance(curr, dict):
            curr = curr.get(k)
        else:
            return default
    return curr if curr is not None else default
```

### Exercise 2.2 — Custom JSON Encoder for a Dataclass (ex2_2_dataclass_json.py)
```python
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json

@dataclass
class EmbeddingResult:
    vector: list[float]
    model: str = "text-embedding-3-small"
    token_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    text: str = ""

class DataclassEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return super().default(obj)
```

## 4. Common Pitfalls & Design Principles (Gotchas & Best Practices)
- **Default Mutable Arguments pitfall**: Never write `topics: list = []` in a `@dataclass`; use `field(default_factory=list)` so that objects do not share the same list in memory.
- **Datetime Serialization pitfall**: `json.dumps()` cannot serialize a `datetime` object by default; convert it to an ISO string (`isoformat()`) before dumping.
- **Missing Dictionary Key pitfall**: When accessing raw JSON data, always prefer `dict.get(key, default)` over `dict[key]` to avoid a `KeyError` crashing the program.

## 5. Independent Tasks & Competency Assessment (Independent Labs)
- Write the pipeline `ex1_3_data_pipeline.py` that reads a list of reviews, removes duplicates by `review_id` using `set()`, groups them to compute the average rating per product, and saves the result to `data/top_5_products.json`.
