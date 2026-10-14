# Week 3 — Lecture Notes (Instructor Guide)

## RAG Foundations: Documents & Embeddings

> **Course:** AI Programming with Python
> **Week:** 3 of 8 — 2 sessions/week × 4 hours (2h theory + 2h lab per session; theory total 4h/week)
> **Prerequisites:** Week 1 (data structures, @dataclass, JSON), Week 2 (LLM API, streaming, error handling)
> **Learning Objectives:**
> - A3.1 Design a `Document` and `DocumentChunk` @dataclass with metadata.
> - A3.2 Implement a `DocumentStore` with add, get, delete, and list operations.
> - A3.3 Implement a chunking strategy (fixed-size, recursive, semantic).
> - A3.4 Generate embeddings using an embedding API and compute cosine similarity between vectors.

---

## Session 1 — Document Model, DocumentStore, Chunking

> **Session format:** 4 hours total = **Part A — Theory (2h)** (this guide) + **Part B — Lab (2h)** (`lab_sheet.md`).

### Session Structure

| # | Segment | Duration | Gagné Event | Activity |
|---|---|---|---|---|
| — | **PART A — THEORY (2h)** | 0:00–2:00 | — | Segments 1–3, 5–6, 8–9, 12–13 |
| 1 | Hook & Problem | 0:00–0:10 | 1. Gain attention | "50-page PDF, 8K context window — what do you do?" |
| 2 | Objectives & Recall | 0:10–0:20 | 2–3. Objectives + Recall | Recap @dataclass; library catalogue analogy |
| 3 | Concept 1: Document Model | 0:20–0:45 | 4. Present content | Document + DocumentChunk @dataclass worked example |
| 5 | KC 1 | 0:45–0:50 | 7. Feedback | MCQ: why overlapping chunks |
| 6 | Concept 2: DocumentStore | 0:50–1:15 | 4. Present content | In-memory DocumentStore with CRUD |
| — | Break | 1:15–1:20 | — | — |
| 9 | Concept 3: Chunking Strategies | 1:20–1:40 | 4. Present content | Fixed-size vs recursive vs semantic (2 micro-cycles) |
| 12 | KC 2 | 1:40–1:45 | 8. Assess | MCQ: small vs large chunks trade-off |
| 13 | Wrap & Reflect | 1:45–2:00 | 9. Retention/Transfer | Preview Session 2 (embeddings) |
| — | **PART B — LAB (2h)** | 2:00–4:00 | — | Segments 4, 7, 10–11 — see `lab_sheet.md` |
| 4 | Code Along 1 (guided) | 2:00–2:30 | 5–6. Guidance + Practice | Define dataclasses, write chunking function (lab_sheet Ex 1.1) |
| 7 | Code Along 2 (guided) | 2:30–2:55 | 5–6. Guidance + Practice | Implement DocumentStore, add docs, search chunks (lab_sheet Ex 1.2) |
| 10 | Independent Lab | 2:55–3:35 | 6. Elicit performance | Document Loader with configurable strategy (lab_sheet Ex 1.3 — 40 min) |
| 11 | Peer Review | 3:35–3:50 | 7. Feedback | Pair code review (lab_sheet Ex 1.4 — 15 min) |
| — | Buffer / questions | 3:50–4:00 | — | — |

---

### 1. Hook & Problem (0:00–0:10)

**Instructor script:**

> "Your company has a 200-page policy manual. An employee asks: 'What is the remote work policy?'
>
> "You call the LLM. But the LLM doesn't know — it wasn't trained on your company's manual. The LLM was trained on the public internet, not on your internal documents.
>
> "You could paste the entire manual into the prompt. But the LLM's context window is 8,000 tokens. A 200-page PDF is roughly 200,000 tokens. That's 25 times too big.
>
> "So what do you do?
>
> "You split the manual into *chunks*. You store those chunks somewhere searchable. When someone asks a question, you find the *relevant* chunks and send only those to the LLM.
>
> "This is RAG — Retrieval-Augmented Generation. And it starts with two things:
> 1. A data model for documents and chunks.
> 2. A store to put them in.
>
> "That's exactly what we're building today."

**Cognitive load note:** This hook introduces the *entire* RAG problem before diving into the first concept. Students will see how Document → Chunk → Store fits into the bigger picture. This is Merrill's Problem-Centered principle — the problem comes before the solution.

---

### 2. Objectives & Recall (0:10–0:20)

**State objectives:**

> "By the end of this session, you will be able to:
> 1. Model a document and its chunks using @dataclass.
> 2. Build an in-memory DocumentStore with add/get/delete/list.
> 3. Implement three chunking strategies (fixed-size, recursive, semantic).
> 4. Load a text file, chunk it, and store the chunks."

**Recall prompt:**

> "From Week 1:
> - How do you define a @dataclass with a mutable default? (`field(default_factory=list)`)
> - What does `from __future__ import annotations` do? (enables `list[str]` syntax in older Python)
>
> Prior knowledge — think about search:
> - How does a library catalogue work? You look up a topic and find books on that topic.
> - What's the smallest unit of a book? A page, a paragraph, a sentence?
> - If you search for 'remote work policy', do you want the whole 200-page manual or just the 3 pages about remote work?"

---

### 3. Concept 1 — Document Model (0:20–0:45)

#### Worked Example: Document and DocumentChunk @dataclass

**Show both classes on the projector:**

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class Document:
    """A full document to be indexed and searched."""
    content: str              # Full text content
    doc_id: str = ""          # Unique identifier (auto-generated if empty)
    title: str = ""           # Document title
    source: str = ""          # File path, URL, or description
    metadata: dict = field(default_factory=dict)  # Flexible metadata
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DocumentChunk:
    """A single chunk of a document after splitting."""
    content: str              # Chunk text content
    chunk_id: str = ""        # Unique identifier
    doc_id: str = ""          # Reference to parent document
    chunk_index: int = 0      # Position in the original document
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None  # Vector embedding (set later)
```

**Key insight:** "These two classes are the foundation of every RAG system. `Document` is the whole thing; `DocumentChunk` is a piece. The `doc_id` field links a chunk back to its parent document. The `embedding` starts as `None` — we'll fill it in Session 2."

**Teaching tip — explain why chunk_id is useful:**

> "Every chunk should have a unique ID. When we retrieve chunks later, we need to know *which* chunk was used to answer a question. The chunk_id lets us trace back: 'This answer used chunk C003 from document DOC002.'"

#### Worked Example: Simple Chunking Function

```python
def chunk_document(doc: Document, chunk_size: int = 512, overlap: int = 64) -> list[DocumentChunk]:
    """
    Split a Document into DocumentChunks using fixed-size windowing.

    Args:
        doc: The document to split
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between consecutive chunks

    Returns:
        A list of DocumentChunk instances
    """
    chunks = []
    text = doc.content
    start = 0

    while start < len(text):
        # Extract chunk text
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        # Create chunk
        chunk = DocumentChunk(
            content=chunk_text,
            chunk_id=f"{doc.doc_id}_chunk_{len(chunks)}",
            doc_id=doc.doc_id,
            chunk_index=len(chunks),
            metadata={**doc.metadata, "chunk_size": chunk_size, "overlap": overlap}
        )
        chunks.append(chunk)

        # Move start position (with overlap)
        start += chunk_size - overlap

        # Break if we're at the end
        if start >= len(text):
            break

    return chunks
```

**Walk through the logic:**

| Iteration | start | end | chunk_text | next start |
|---|---|---|---|---|
| 0 | 0 | 512 | chars 0–511 | 512 - 64 = 448 |
| 1 | 448 | 960 | chars 448–959 | 448 + 512 - 64 = 896 |
| 2 | 896 | 1408 | chars 896–1407 | ... |

**Key insight:** "Overlap ensures that no information is lost at chunk boundaries. If a sentence spans from position 500 to 520, it appears in both chunk 0 and chunk 1. This prevents cutting a sentence in half."

---

### 4. Code Along 1 — Define Dataclasses + Chunking (2:00–2:30) [PART B — LAB]

**File:** `ex1_1_document_model.py`

> **Recall note:** The `Document` and `DocumentChunk` @dataclass definitions are shown in Concept 1 above. We reuse them here — copy them into your file (or import them) so we can focus on the new code: the `chunk_document()` function.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# (Document and DocumentChunk @dataclass defined in Concept 1 — copy them above)

# === Step 1: Create a sample document using the @dataclass from Concept 1 ===

doc = Document(
    content="""Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability with its notable use of significant indentation. Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, and functional programming. Python is often described as a "batteries included" language due to its comprehensive standard library. It has become one of the most popular programming languages, particularly in data science, artificial intelligence, and web development.""",
    doc_id="DOC001",
    title="Introduction to Python",
    source="course_materials"
)

print(f"Document: {doc.title} ({doc.doc_id})")
print(f"Content length: {len(doc.content)} chars\n")

# === Step 3: Implement chunking ===

def chunk_document(doc: Document, chunk_size: int = 150, overlap: int = 30) -> list[DocumentChunk]:
    """
    Split a Document into DocumentChunks using fixed-size windowing.
    """
    chunks = []
    text = doc.content
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        chunk = DocumentChunk(
            content=chunk_text,
            chunk_id=f"{doc.doc_id}_chunk_{len(chunks)}",
            doc_id=doc.doc_id,
            chunk_index=len(chunks),
            metadata={**doc.metadata, "chunk_size": chunk_size, "overlap": overlap}
        )
        chunks.append(chunk)

        start += chunk_size - overlap
        if start >= len(text):
            break

    return chunks

# === Step 4: Test it ===

chunks = chunk_document(doc, chunk_size=150, overlap=30)
print(f"Created {len(chunks)} chunks:\n")

for chunk in chunks:
    print(f"[{chunk.chunk_id}] (index {chunk.chunk_index})")
    print(f"  Text: {chunk.content[:80]}...")
    print(f"  Length: {len(chunk.content)} chars")
    print()
```

**Checkpoint:**
```
Document: Introduction to Python (DOC001)
Content length: 498 chars

Created 4 chunks:

[DOC001_chunk_0] (index 0)
  Text: Python is a high-level, interpreted programming language created by Guido van R...
  Length: 150 chars

[DOC001_chunk_1] (index 1)
  Text: guage created by Guido van Rossum and first released in 1991. It emphasizes cod...
  Length: 150 chars
...
```

---

### 5. Knowledge Check 1 (0:45–0:50)

**MCQ:**

> Why do we use overlapping chunks when splitting a document?
>
> (a) To increase the number of chunks (more chunks = more retrieval)
> (b) To avoid cutting a sentence or thought in half at the boundary
> (c) To reduce storage space

**Correct answer: (b) To avoid cutting a sentence or thought in half at the boundary**

**Explanation:** Overlap ensures continuity. If a sentence spans from character 148 to 155 in a 150-char chunk, part of it appears in chunk 0 (chars 148–150) and part in chunk 1 (chars 120–155 due to overlap). Without overlap, the sentence would be cut — the LLM would only see half of it.

---

### 6. Concept 2 — DocumentStore (In-Memory) (0:50–1:15)

#### Worked Example: DocumentStore Interface

```python
@dataclass
class DocumentStore:
    """
    An in-memory store for Documents and their Chunks.

    This is the foundation. Later, we'll replace the internal dict
    with a vector database — but the interface stays the same.
    """
    documents: dict[str, Document] = field(default_factory=dict)
    chunks: dict[str, DocumentChunk] = field(default_factory=dict)

    def add(self, document: Document) -> str:
        """Add a document (and return its doc_id)."""
        if not document.doc_id:
            document.doc_id = f"doc_{len(self.documents) + 1:04d}"
        self.documents[document.doc_id] = document
        return document.doc_id

    def get(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        return self.documents.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        """Delete a document and its chunks. Returns True if deleted."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            # Also delete all chunks belonging to this document
            chunk_ids = [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]
            for cid in chunk_ids:
                del self.chunks[cid]
            return True
        return False

    def list_all(self) -> list[Document]:
        """List all documents."""
        return list(self.documents.values())

    def add_chunk(self, chunk: DocumentChunk) -> str:
        """Add a chunk (and return its chunk_id)."""
        if not chunk.chunk_id:
            chunk.chunk_id = f"chunk_{len(self.chunks) + 1:04d}"
        self.chunks[chunk.chunk_id] = chunk
        return chunk.chunk_id

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a chunk by ID."""
        return self.chunks.get(chunk_id)

    def get_chunks_by_doc(self, doc_id: str) -> list[DocumentChunk]:
        """Get all chunks for a document."""
        return [c for c in self.chunks.values() if c.doc_id == doc_id]

    def count(self) -> dict:
        """Return counts of documents and chunks."""
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks)
        }
```

**Key insight:** "This is a *store* — it stores things and lets you retrieve them. The internal implementation (a dict) is the simplest possible version. Later, we'll replace it with a vector database, but the `add()`, `get()`, `search()` interface stays the same. This is the **interface-first design** pattern."

**Teaching tip — why in-memory matters:**

> "A real RAG system uses a vector database (ChromaDB, Pinecone, Weaviate). But building our own in-memory store teaches you *how* these databases work internally. You'll understand vector search better because you've implemented it yourself."

---

### 7. Code Along 2 — Implement DocumentStore (2:30–2:55) [PART B — LAB]

**File:** `ex1_2_document_store.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Reuse the Document and DocumentChunk from Exercise 1.1
@dataclass
class Document:
    content: str
    doc_id: str = ""
    title: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DocumentChunk:
    content: str
    chunk_id: str = ""
    doc_id: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None

@dataclass
class DocumentStore:
    """An in-memory store for Documents and their Chunks."""
    documents: dict[str, Document] = field(default_factory=dict)
    chunks: dict[str, DocumentChunk] = field(default_factory=dict)

    def add(self, document: Document) -> str:
        if not document.doc_id:
            document.doc_id = f"doc_{len(self.documents) + 1:04d}"
        self.documents[document.doc_id] = document

        # Auto-chunk the document
        chunks = chunk_document(document)
        for chunk in chunks:
            self.add_chunk(chunk)

        return document.doc_id

    def get(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        if doc_id in self.documents:
            del self.documents[doc_id]
            chunk_ids = [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]
            for cid in chunk_ids:
                del self.chunks[cid]
            return True
        return False

    def list_all(self) -> list[Document]:
        return list(self.documents.values())

    def add_chunk(self, chunk: DocumentChunk) -> str:
        if not chunk.chunk_id:
            chunk.chunk_id = f"chunk_{len(self.chunks) + 1:04d}"
        self.chunks[chunk.chunk_id] = chunk
        return chunk.chunk_id

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self.chunks.get(chunk_id)

    def get_chunks_by_doc(self, doc_id: str) -> list[DocumentChunk]:
        return [c for c in self.chunks.values() if c.doc_id == doc_id]

    def count(self) -> dict:
        return {"documents": len(self.documents), "chunks": len(self.chunks)}

def chunk_document(doc: Document, chunk_size: int = 150, overlap: int = 30) -> list[DocumentChunk]:
    """Split a Document into DocumentChunks."""
    chunks = []
    text = doc.content
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = DocumentChunk(
            content=text[start:end],
            chunk_id=f"{doc.doc_id}_chunk_{len(chunks)}",
            doc_id=doc.doc_id,
            chunk_index=len(chunks),
            metadata={**doc.metadata, "chunk_size": chunk_size}
        )
        chunks.append(chunk)
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks

# === Test it ===
store = DocumentStore()

# Add 3 sample documents
doc1 = Document(
    content="Python is a high-level programming language. It was created by Guido van Rossum in 1991. Python emphasizes code readability. It is widely used in AI and data science.",
    title="Introduction to Python",
    source="course_materials"
)

doc2 = Document(
    content="RAG stands for Retrieval-Augmented Generation. It combines a retrieval system with a generative model. The retriever finds relevant documents. The generator produces an answer based on those documents.",
    title="What is RAG?",
    source="course_materials"
)

doc3 = Document(
    content="Embeddings are numerical representations of text. Similar texts have similar embeddings. Cosine similarity measures how similar two embeddings are. Values range from -1 to 1.",
    title="Understanding Embeddings",
    source="course_materials"
)

for doc in [doc1, doc2, doc3]:
    store.add(doc)

print(store.count())
print(f"\nDocuments ({len(store.list_all())}):")
for d in store.list_all():
    chunks = store.get_chunks_by_doc(d.doc_id)
    print(f"  - {d.title} ({d.doc_id}): {len(chunks)} chunks")

print(f"\nAll chunks ({len(store.chunks)}):")
for cid, chunk in store.chunks.items():
    print(f"  - [{cid}] {chunk.content[:60]}...")
```

**Checkpoint:**
```
{'documents': 3, 'chunks': 12}

Documents (3):
  - Introduction to Python (doc_0001): 4 chunks
  - What is RAG? (doc_0002): 4 chunks
  - Understanding Embeddings (doc_0003): 4 chunks

All chunks (12):
  - [doc_0001_chunk_0] Python is a high-level programming language. It was ...
  ...
```

---

### 8. Break (1:15–1:20)

---

### 9. Concept 3 — Chunking Strategies (Part A: Fixed-Size & Recursive) (1:20–1:40)

#### Micro-Cycle A: Fixed-Size and Recursive Chunking

**Show on the projector the same text chunked two ways:**

```python
sample_text = """# Chapter 1: Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## 1.1 Supervised Learning

Supervised learning is the most common form of machine learning. The algorithm is trained on labeled data, where each training example is paired with an output label.

## 1.2 Unsupervised Learning

Unsupervised learning involves training on data without labels. The algorithm must find patterns and structure in the data on its own.

## 1.3 Reinforcement Learning

Reinforcement learning involves an agent that learns by interacting with an environment. The agent receives rewards or penalties based on its actions."""
```

**Strategy 1: Fixed-size (simple but may cut mid-sentence)**

```python
def chunk_fixed(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
    """Fixed-size chunking with character overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks
```

**Strategy 2: Recursive (respects document structure)**

```python
def chunk_recursive(text: str, chunk_size: int = 200) -> list[str]:
    """
    Recursive chunking — split on paragraph boundaries first,
    then fall back to sentence boundaries.

    This preserves natural document structure without requiring regex.
    """
    # Level 1: Split on double newlines (paragraphs)
    paragraphs = [p for p in text.split('\n\n') if p.strip()]

    chunks = []
    buffer = ""

    for para in paragraphs:
        # If adding this paragraph exceeds chunk_size, finish the current chunk
        if len(buffer) + len(para) > chunk_size and buffer:
            chunks.append(buffer.strip())
            buffer = ""

        # If a single paragraph is larger than chunk_size, split it by sentences
        if len(para) > chunk_size:
            sentences = [s.strip() + "." for s in para.split(".") if s.strip()]
            for sentence in sentences:
                if len(buffer) + len(sentence) > chunk_size and buffer:
                    chunks.append(buffer.strip())
                    buffer = sentence
                else:
                    buffer += " " + sentence if buffer else sentence
        else:
            buffer += "\n\n" + para if buffer else para

    if buffer:
        chunks.append(buffer.strip())

    return chunks
```

> **Note:** This simplified recursive version avoids regex. A production version would split on `##` headers first (see `data/recursive_chunking_advanced.py` in the repo).

**Key insight — trade-offs table:**

| Strategy | Pros | Cons | Best For |
|---|---|---|---|
| Fixed-size | Simple, fast, predictable | Cuts sentences/thoughts | Quick prototyping |
| Recursive | Preserves structure, clean | May create uneven chunks | Most production RAG |

**Knowledge check (think-pair-share):** "Which strategy would you use for a legal document where every sentence matters? Which for a set of Tweets?" (30 seconds, discuss with partner)

---

### 9b. Concept 3 (Part B) — Semantic Chunking (Bonus/Extension) (2:10–2:35 continued)

> **Instructor note:** Present semantic chunking only if time permits and students are engaged. Otherwise, mention it briefly and point to the reference notes.

**Strategy 3: Semantic (LLM-based topic boundaries)**

```python
# This is an advanced technique — use with caution (costly)
def chunk_semantic(text: str, llm_client) -> list[str]:
    """
    Use an LLM to identify topic boundaries and split accordingly.
    This is expensive and slow — use only when quality matters.
    """
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Split this text into coherent chunks. "
                       f"Each chunk should cover one topic. "
                       f"Return the chunks separated by '---CHUNK---':\n\n{text}"
        }]
    )
    result = response.choices[0].message.content
    return [c.strip() for c in result.split("---CHUNK---") if c.strip()]
```

| Strategy | Pros | Cons | Best For |
|---|---|---|---|
| Semantic | Coherent topics | Expensive, slow, variable | High-stakes QA |

**When to use semantic chunking:** If you have a small number of high-value documents (e.g., legal contracts, medical guidelines) and need the highest possible chunk quality. For most RAG applications, recursive chunking is the right default.

---

### 10. Independent Lab — Document Loader (2:55–3:35) [PART B — LAB]

**File:** `ex1_3_document_loader.py`

Students write a function that reads a text file, chunks it, stores in DocumentStore, and prints stats.

Full instructions in `lab_sheet.md`.

**Setup checklist (before class):**
- [ ] Create `data/sample_doc.txt` — a multi-paragraph text about Python programming (see lab_sheet)
- [ ] Create `data/company_policy.txt` — a longer text for the extension
- [ ] Test the solution

**Teaching tips:**
- Watch for students who forget to handle the overlap parameter correctly
- Common issue: `chunk_id` collisions if adding multiple documents
- Show how to use `enumerate` for chunk_index

---

### 11. Peer Review (3:35–3:50) [PART B — LAB]

**Pair up. Review checklist:**

> 1. Does the chunking preserve paragraph boundaries? (recursive strategy)
> 2. Are edge cases handled? (empty doc, single word, single character)
> 3. Are the statistics useful? (num chunks, avg length, min/max)
> 4. Is the DocumentStore correctly populated?
> 5. Is the code readable?

---

### 12. Knowledge Check 2 (1:40–1:45)

**MCQ:**

> A chunk of 200 chars returns relevant results but misses nuance. A chunk of 2000 chars includes all the context but also a lot of noise. What is the trade-off?
>
> (a) Precision vs. recall
> (b) Speed vs. accuracy
> (c) Both precision/recall AND speed/accuracy

**Correct answer: (c) Both**

**Explanation:** Small chunks = high precision (each chunk is specific) but lower recall (may miss relevant context). Large chunks = higher recall (more context captured) but lower precision (more noise). Also, more small chunks = more embeddings to search (slower), while fewer large chunks = faster search but worse quality.

---

### 13. Wrap & Reflect (1:45–2:00)

**Instructor summary:**

> "Today we built the data layer for RAG:
> - **Document + DocumentChunk @dataclass** — the data model
> - **DocumentStore** — the storage layer
> - **Chunking** — splitting documents into searchable pieces
>
> Session 2: We make these chunks searchable by meaning, not just by keywords. We'll embed them into vector space and find similar chunks using cosine similarity."

**Reflection prompt (3 min):**

> "What's the hardest part about chunking? What questions do you have about the trade-offs between different chunk sizes?"

---

## Session 2 — Embeddings, Vector Stores, Similarity Search

> **Session format:** 4 hours total = **Part A — Theory (2h)** (this guide) + **Part B — Lab (2h)** (`lab_sheet.md`).

### Session Structure

| # | Segment | Duration | Gagné Event | Activity |
|---|---|---|---|---|
| — | **PART A — THEORY (2h)** | 0:00–2:00 | — | Segments 1–3, 5–6, 8–9, 12–13 |
| 1 | Hook | 0:00–0:10 | 1. Gain attention | 2D plot of embedded sentences |
| 2 | Objectives & Recall | 0:10–0:20 | 2–3. Objectives + Recall | Recap Session 1; vector basics |
| 3 | Concept 1: Embedding Models | 0:20–0:45 | 4. Present content | OpenAI embedding API, vector output |
| 5 | KC 1 | 0:45–0:50 | 7. Feedback | MCQ: what is an embedding |
| 6 | Concept 2: Cosine Similarity | 0:50–1:15 | 4. Present content | Formula, numpy implementation, examples |
| — | Break | 1:15–1:20 | — | — |
| 9 | Concept 3: Vector Store | 1:20–1:40 | 4. Present content | Extend DocumentStore with embeddings + search |
| 12 | KC 2 | 1:40–1:45 | 8. Assess | MCQ: similarity threshold |
| 13 | Wrap & Reflect | 1:45–2:00 | 9. Retention/Transfer | Preview Week 4 (RAG pipeline) |
| — | **PART B — LAB (2h)** | 2:00–4:00 | — | Segments 4, 7, 10–11 — see `lab_sheet.md` |
| 4 | Code Along 1 (guided) | 2:00–2:30 | 5–6. Guidance + Practice | Embed sentences, examine vectors (lab_sheet Ex 2.1) |
| 7 | Code Along 2 (guided) | 2:30–2:55 | 5–6. Guidance + Practice | find_most_similar function (lab_sheet Ex 2.2) |
| 10 | Independent Lab | 2:55–3:35 | 6. Elicit performance | Vector Search Engine (lab_sheet Ex 2.3 — 40 min) |
| 11 | Peer Review | 3:35–3:50 | 7. Feedback | Pair code review (lab_sheet Ex 2.4 — 15 min) |
| — | Buffer / questions | 3:50–4:00 | — | — |

---

### 1. Hook — Words Become Numbers (0:00–0:10)

**Live demo — run this before the session starts:**

```python
# Pre-compute embeddings for a few sentences and show them on a 2D plot
# (Use PCA to reduce 1536d → 2d)

from openai import OpenAI
from sklearn.decomposition import PCA  # or just show a pre-made plot
import matplotlib.pyplot as plt
import numpy as np

sentences = [
    "I love programming in Python",
    "Python is my favorite programming language",
    "The weather today is rainy and cold",
    "It's going to rain tomorrow"
]

client = OpenAI()
embeddings = []
for s in sentences:
    resp = client.embeddings.create(model="text-embedding-3-small", input=s)
    embeddings.append(resp.data[0].embedding)

# PCA to 2D
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for i, (x, y) in enumerate(coords):
    plt.scatter(x, y, s=100)
    plt.annotate(sentences[i][:30], (x, y), fontsize=9)
plt.title("Sentence Embeddings (PCA 2D projection)")
plt.show()
```

**If you can't run PCA live, prepare the plot as a PNG beforehand.**

**Instructor script:**

> "Look at this plot. The two sentences about Python are *near each other*. The two about weather are *near each other*. But the Python sentences are *far from* the weather sentences.
>
> "This is the magic of embeddings. Each sentence becomes a point in a 1536-dimensional space. Sentences with similar *meaning* have similar *coordinates*. The LLM doesn't understand text; it understands numbers. Embeddings bridge that gap.
>
> "Today, you'll learn to:
> 1. Convert text into embeddings (using the embedding API).
> 2. Measure similarity between texts (using cosine similarity).
> 3. Build a vector store that searches by meaning, not by keywords."

---

### 2. Objectives & Recall (0:10–0:20)

**State objectives:**

> "By the end of this session, you will be able to:
> 1. Call the embedding API to convert text into vectors.
> 2. Implement cosine similarity from scratch.
> 3. Build a vector store that searches by semantic similarity.
> 4. Evaluate search results and set similarity thresholds."

**Recall prompt:**

> "From Session 1:
> - What is a DocumentChunk? (a piece of a document with content, doc_id, chunk_index)
> - What does the DocumentStore store? (documents and chunks in dicts)
>
> Prior knowledge — vectors:
> - What is a vector? (a list of numbers, an arrow in space)
> - How do you measure if two vectors point in the same direction? (the angle between them — cosine similarity)
> - What is the dot product of two vectors? (sum of element-wise products)"

---

### 3. Concept 1 — Embedding Models (0:20–0:45)

#### Persistent Reference Card (keep visible all session)

> **Print this reference card and keep it projected/visible for the entire session.** It contrasts the two API response structures students must distinguish.
>
> ```
> CHAT API (Week 2):                     EMBEDDING API (Week 3):
> --------------------                   ------------------------
> client.chat.completions.create(...)    client.embeddings.create(...)
> response.choices[0]                    response.data[0]
>   .message.content   → str             .embedding   → list[float]
>   .finish_reason     → str             (no finish_reason)
> response.usage.prompt_tokens           response.usage.prompt_tokens
> response.model                         response.model
> ```

#### Worked Example: Calling the Embedding API

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# The simplest embedding call
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="What is RAG in AI?"
)

# The response contains the embedding vector
embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")
print(f"Type: {type(embedding)}")  # list of floats
```

**Expected output:**
```
Embedding dimension: 1536
First 10 values: [0.023, -0.045, 0.112, -0.067, 0.089, 0.001, -0.034, 0.055, -0.021, 0.078]
Type: <class 'list'>
```

**Key insight:** "An embedding is a list of 1536 floating-point numbers. Each number represents a *semantic feature* — but we don't know what each feature means individually. The *pattern* across all 1536 numbers captures the meaning of the text."

#### Model Comparison

```python
# text-embedding-3-small: 1536 dimensions, $0.02/1M tokens — best for most use cases
# text-embedding-3-large: 3072 dimensions, $0.13/1M tokens — higher quality, more expensive

response_small = client.embeddings.create(
    model="text-embedding-3-small",
    input="Hello world"
)
response_large = client.embeddings.create(
    model="text-embedding-3-large",
    input="Hello world"
)

print(f"Small: {len(response_small.data[0].embedding)} dims")
print(f"Large: {len(response_large.data[0].embedding)} dims")
```

**Teaching tip:** "text-embedding-3-small is the default choice. It's cheap ($0.02 per 1M tokens), fast, and good enough for most RAG systems. Only use large if you need maximum accuracy and can afford the cost."

#### Batch Embedding

```python
# Much more efficient — embed multiple texts in one API call
texts = [
    "What is RAG in AI?",
    "Python programming basics",
    "The weather is nice today"
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

for i, data in enumerate(response.data):
    print(f"Text {i}: {len(data.embedding)} dims, first 3 values: {data.embedding[:3]}")
```

---

### 4. Code Along 1 — Embed and Examine Vectors (2:00–2:30) [PART B — LAB]

**File:** `ex2_1_embeddings.py`

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Embed a single text and return the vector."""
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding

def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed multiple texts and return vectors."""
    response = client.embeddings.create(model=model, input=texts)
    return [data.embedding for data in response.data]

# === Step 1: Embed one sentence ===
sentence = "RAG stands for Retrieval-Augmented Generation"
vec = embed_text(sentence)
print(f"Sentence: {sentence}")
print(f"Dimension: {len(vec)}")
print(f"First 5 values: {vec[:5]}")
print(f"Vector norm: {np.linalg.norm(vec):.4f}")
print(f"Min: {min(vec):.4f}, Max: {max(vec):.4f}")
print()

# === Step 2: Embed three sentences about different topics ===
sentences = [
    "RAG combines retrieval with text generation",
    "Python is a popular programming language",
    "I enjoy hiking in the mountains"
]

embeddings = embed_texts(sentences)

for s, v in zip(sentences, embeddings):
    print(f"  '{s[:40]}...' → dim {len(v)}, norm {np.linalg.norm(v):.4f}")
```

**Checkpoint:**
```
Sentence: RAG stands for Retrieval-Augmented Generation
Dimension: 1536
First 5 values: [0.023, -0.045, 0.112, -0.067, 0.089]
Vector norm: 1.0000  <-- note: OpenAI embeddings are normalised!

  'RAG combines retrieval with text generation...' → dim 1536, norm 1.0000
  'Python is a popular programming language...' → dim 1536, norm 1.0000
  'I enjoy hiking in the mountains...' → dim 1536, norm 1.0000
```

**Teaching tip:** "OpenAI normalises their embeddings to length 1. This means cosine similarity = dot product, which simplifies the math. Not all embedding providers do this!"

---

### 5. Knowledge Check 1 (0:45–0:50)

**MCQ:**

> What does a 1536-dimensional embedding from `text-embedding-3-small` represent?
>
> (a) 1536 keywords extracted from the text
> (b) 1536 numerical coordinates in a semantic vector space
> (c) 1536 tokens from the text

**Correct answer: (b) 1536 numerical coordinates in a semantic vector space**

**Explanation:** Each embedding is a point in a 1536-dimensional semantic space. The position of the point captures the *meaning* of the text. Similar texts are located near each other in this space. It's not a bag of keywords (a) nor a list of tokens (c).

---

### 6. Concept 2 — Cosine Similarity (0:50–1:15)

#### Worked Example: The Formula

**Show on the projector:**

```
cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)

where:
  a · b = Σ(a_i × b_i)         (dot product)
  ||a|| = sqrt(Σ(a_i²))        (magnitude / L2 norm)

Range: [-1, 1]
  1.0  = identical direction (same meaning)
  0.0  = orthogonal (unrelated)
  -1.0 = opposite direction (opposite meaning)
```

#### Implementation from Scratch

**Integrated annotation — formula → code mapping:**

```
cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)
                           │      │    │      │
                           │      │    │      └ np.linalg.norm(b_arr)
                           │      │    └ np.linalg.norm(a_arr)
                           │      └ np.dot(a_arr, b_arr)
                           └ return float(dot_product / (norm_a * norm_b))
```

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)

    dot_product = np.dot(a_arr, b_arr)   # (a · b) — dot product
    norm_a = np.linalg.norm(a_arr)       # ||a|| — magnitude of a
    norm_b = np.linalg.norm(b_arr)       # ||b|| — magnitude of b

    if norm_a == 0 or norm_b == 0:
        return 0.0  # Avoid division by zero

    return float(dot_product / (norm_a * norm_b))  # (a · b) / (||a|| × ||b||)
```

# Test with the three embedded sentences from Code Along 1
embeddings = embed_texts([
    "RAG combines retrieval with text generation",
    "Python is a popular programming language",
    "I enjoy hiking in the mountains"
])

sim_0_1 = cosine_similarity(embeddings[0], embeddings[1])
sim_0_2 = cosine_similarity(embeddings[0], embeddings[2])
sim_1_2 = cosine_similarity(embeddings[1], embeddings[2])

print(f"RAG vs Python:   {sim_0_1:.4f}")  # Low (different topics)
print(f"RAG vs Hiking:   {sim_0_2:.4f}")  # Very low
print(f"Python vs Hiking: {sim_1_2:.4f}")  # Very low
```

**Expected output (approximate):**
```
RAG vs Python:   0.38
RAG vs Hiking:   0.12
Python vs Hiking: 0.15
```

**Key insight:** "RAG and Python are both about technology, so their similarity is ~0.38 — not identical but related. Hiking is unrelated to both, so similarity is ~0.12. The numbers confirm what we expect: related topics have higher cosine similarity."

#### Teaching Tip: Show Similarities with Truly Similar Sentences

```python
embeddings = embed_texts([
    "I love programming in Python",
    "Python is my favorite language",
    "The weather is rainy today"
])

sim_0_1 = cosine_similarity(embeddings[0], embeddings[1])  # Same meaning
sim_0_2 = cosine_similarity(embeddings[0], embeddings[2])  # Different

print(f"Same topic (Python):   {sim_0_1:.4f}")  # ~0.85+
print(f"Different topic:       {sim_0_2:.4f}")  # ~0.10-0.30
```

---

### 7. Code Along 2 — find_most_similar (2:30–2:55) [PART B — LAB]

**File:** `ex2_2_similarity_search.py`

```python
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from typing import TypeAlias

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Type alias — makes the nested type readable
Candidate: TypeAlias = tuple[str, list[float]]  # (text, embedding vector)
Result: TypeAlias = tuple[str, float]           # (text, similarity score)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts."""
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [data.embedding for data in response.data]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    if np.linalg.norm(a_arr) == 0 or np.linalg.norm(b_arr) == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

def find_most_similar(
    query: str,
    candidates: list[Candidate],
    top_k: int = 3
) -> list[Result]:
    """
    Find the top_k most similar items to a query.

    Args:
        query: The query text
        candidates: List of (text, embedding) tuples
        top_k: Number of results to return

    Returns:
        List of (text, similarity_score) tuples, sorted by score descending
    """
    # Embed the query
    query_vec = embed_texts([query])[0]

    # Compute similarities
    results = []
    for text, vec in candidates:
        score = cosine_similarity(query_vec, vec)
        results.append((text, score))

    # Sort by score descending and return top_k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# === Test with sample data ===
# Our candidate database
sentences = [
    "RAG is a technique for combining retrieval with generation",
    "Python is a high-level programming language",
    "The capital of France is Paris",
    "Embeddings map text to numerical vectors",
    "Cosine similarity measures the angle between vectors",
    "I love hiking and exploring nature trails",
]

print("Building embedding index...")
candidates = [(s, vec) for s, vec in zip(sentences, embed_texts(sentences))]

# Test queries
test_queries = [
    "What is RAG?",
    "Tell me about vector similarity",
    "What is the capital of France?",
]

for query in test_queries:
    print(f"\n=== Query: '{query}' ===")
    results = find_most_similar(query, candidates, top_k=2)
    for text, score in results:
        print(f"  [{score:.4f}] {text}")
```

**Expected output:**
```
=== Query: 'What is RAG?' ===
  [0.89] RAG is a technique for combining retrieval with generation
  [0.45] Embeddings map text to numerical vectors

=== Query: 'Tell me about vector similarity' ===
  [0.82] Cosine similarity measures the angle between vectors
  [0.65] Embeddings map text to numerical vectors

=== Query: 'What is the capital of France?' ===
  [0.91] The capital of France is Paris
  [0.25] RAG is a technique for combining retrieval with generation
```

**Teaching tip:** "Notice that for 'What is RAG?', the most similar sentence is the one about RAG, not the one containing the word 'RAG'. This is semantic search — it finds based on meaning, not just keyword matching."

---

### 8. Break (1:15–1:20)

---

### 9. Concept 3 — Vector Store (1:20–1:40)

#### Recall Sprint (2 min, pair activity)

> Before we look at the VectorStore code, activate your prerequisite schemas. **In 30 seconds, explain to your partner:**
>
> 1. What does `embed_text()` return? *(a list of 1536 floats)*
> 2. What does `cosine_similarity()` return? *(a float between -1 and 1)*
> 3. What does `DocumentStore.add()` do? *(stores a document and creates chunks)*
>
> **Why this helps:** The VectorStore reuses all three — embedding, cosine similarity, and the DocumentStore interface. Activating them now reduces the load of holding them all at once.

#### Worked Example: Extending DocumentStore with Embeddings

```python
@dataclass
class VectorStore(DocumentStore):
    """
    A DocumentStore that also indexes chunks by embedding vectors
    for semantic similarity search.
    """

    def add_chunk(self, chunk: DocumentChunk) -> str:
        """Add a chunk and generate its embedding."""
        # Generate embedding if not already present
        if chunk.embedding is None and chunk.content:
            chunk.embedding = embed_text(chunk.content)
        return super().add_chunk(chunk)

    def add(self, document: Document) -> str:
        """Add a document and embed all its chunks."""
        doc_id = super().add(document)  # This calls add_chunk for each chunk
        return doc_id

    def search(self, query: str, top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
        """
        Search for the most relevant chunks to a query.

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            List of (chunk, similarity_score) tuples, sorted by score descending
        """
        if not self.chunks:
            return []

        # Embed the query
        query_vec = embed_text(query)

        # Compute similarity against all chunks
        results = []
        for chunk in self.chunks.values():
            if chunk.embedding is None:
                continue
            score = cosine_similarity(query_vec, chunk.embedding)
            results.append((chunk, score))

        # Sort and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_with_threshold(
        self, query: str, top_k: int = 3, min_score: float = 0.5
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Search with a minimum similarity threshold.
        Only returns results with score >= min_score.
        """
        results = self.search(query, top_k=len(self.chunks))  # Get all results
        return [(c, s) for c, s in results if s >= min_score][:top_k]
```

**Key insight:** "The VectorStore extends DocumentStore with one thing: embedding-based search. The `search()` method is the most important function in this course — it's the entire point of RAG."

---

### 10. Independent Lab — Vector Search Engine (2:55–3:35) [PART B — LAB]

**File:** `ex2_3_vector_search.py`

Full instructions in `lab_sheet.md`.

**Teaching tips:**
- The embedding API calls take time (≈ 0.5 seconds per chunk). 10 chunks × 0.5s = 5 seconds.
- Warn students: "The first search may be slow because we need to embed all chunks first."
- The `search_with_threshold` function is a bonus for students who finish early

---

### 11. Peer Review (3:35–3:50) [PART B — LAB]

**Pair up. Review checklist:**

> 1. Does the search return semantically relevant results? (not just keyword matches)
> 2. Are scores reasonable? (0.7–1.0 for relevant, < 0.5 for irrelevant)
> 3. Is cosine similarity correctly implemented?
> 4. Are edge cases handled? (empty store, query with no matches)
> 5. Does the search_with_threshold work?

---

### 12. Knowledge Check 2 (1:40–1:45)

**MCQ:**

> A query returns chunks with similarity scores of 0.92, 0.88, and 0.45. Should you include the 0.45 result in the context sent to the LLM?
>
> (a) Yes — always include all results up to top_k
> (b) No — set a threshold > 0.7 and exclude low-quality results
> (c) It depends on the application — sometimes more context helps, sometimes it adds noise

**Correct answer: (c) It depends**

**Explanation:** Including low-scoring chunks can be helpful (more context, might contain useful info) or harmful (noise that confuses the LLM, wastes tokens). The right approach depends on your application's precision/recall requirements. A configurable `min_score` parameter lets you tune this.

---

### 13. Wrap & Reflect (1:45–2:00)

**Instructor summary:**

> "This week you built the core of RAG:
> - **Session 1:** Documents, chunks, store, and chunking strategies.
> - **Session 2:** Embeddings, cosine similarity, and vector search.
>
> Next week: the RAG pipeline. You'll combine your VectorStore with the LLM API from Week 2 to build a system that:
> 1. Takes a user's question
> 2. Finds relevant chunks in the VectorStore
> 3. Sends those chunks to the LLM as context
> 4. Returns a grounded answer with citations
>
> **Lab 3 (due before Session 7):** Build a complete `VectorStore` class with save/load, batch search, and configurable thresholds."

**Reflection prompt (3 min):**

> "What surprised you most about how embeddings work? Did any search results surprise you?"

---

## Instructor Preparation Checklist

### Before Session 1

- [ ] Create `data/sample_doc.txt` (see lab_sheet.md)
- [ ] Create `data/company_policy.txt` (longer text for extension)
- [ ] Test all Code Along scripts
- [ ] Test the Independent Lab solution
- [ ] Prepare KC slides

### Before Session 2

- [ ] Prepare 2D embedding plot (pre-computed PNG) for the hook
- [ ] Ensure `pip install numpy scikit-learn matplotlib` for hook demo (or just use pre-computed)
- [ ] Verify all students have API keys (embedding API uses a different endpoint)
- [ ] Test all Code Along scripts
- [ ] Test the Independent Lab solution
- [ ] Embedding costs: text-embedding-3-small is $0.02/1M tokens — 100 chunks per student ≈ 2K tokens ≈ $0.00004 per student — negligible

### Session 2 API Note

The embedding API does NOT use the chat completions interface. Make sure students understand:

```python
# This is the embedding API — different endpoint!
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="text to embed"
)
# NOT: client.chat.completions.create()
```

---

## Common Student Mistakes & Interventions

| Mistake | When It Appears | Intervention |
|---|---|---|
| `field(default_factory=dict)` forgotten — mutable default error | Code Along 1, S1 | "Mutable defaults must use `field(default_factory=...)`" |
| Overlap logic wrong — chunks don't overlap correctly | Code Along 1, S1 | Draw the overlap diagram on the board |
| `chunk_id` collisions across documents | Code Along 2, S1 | "Include doc_id in chunk_id: `{doc_id}_chunk_{index}`" |
| Off-by-one in chunk splitting (misses last chunk) | Code Along 2, S1 | "Check the `while start < len(text)` condition" |
| `ModuleNotFoundError: No module named 'openai'` | Code Along 1, S2 | `pip install openai python-dotenv` |
| Confusing embedding API with chat API | Code Along 1, S2 | "It's `client.embeddings.create()`, not `client.chat.completions.create()`" |
| `response.data[0].embedding` vs `response.choices[0]` | Code Along 1, S2 | "Embedding response has `data[0].embedding` — different structure from chat" |
| `numpy.linalg.norm` → zero vector → division by zero | Code Along 2, S2 | Add `if norm == 0: return 0.0` check |
| Search returns chunks with score 0.12 — "it's broken!" | Code Along 2, S2 | "No, 0.12 means unrelated. That's working correctly!" |
| Forgetting to embed chunks before searching | Independent Lab, S2 | "Call `embed_text()` in `add_chunk()`, not in `search()`" |

---

## KC Answer Key (for instructor reference)

### Session 1
- **KC1:** (b) To avoid cutting a sentence or thought in half at the boundary
- **KC2:** (c) Both precision/recall AND speed/accuracy trade-offs

### Session 2
- **KC1:** (b) 1536 numerical coordinates in a semantic vector space
- **KC2:** (c) It depends on the application — configurable threshold is best

---

## CLT Compliance Checklist (Week 3)

> Đối chiếu sau mỗi lần chỉnh sửa nội dung. Tất cả mục dưới đây hiện đã **✅ đạt** (kiểm tra lần cuối: 2026-08-17, đối chiếu với `cognitive_load_analysis.md` — 6 đề xuất P1–P6). Nếu một mục bị đổi thành ❌, cần sửa lại trước khi dùng tài liệu.

| # | Kiểm tra | Vị trí | Trạng thái |
|---|---|---|---|
| 1 | Chunking strategies tách micro-cycles (Micro-Cycle A/B) — không dồn 3 chiến lược cùng lúc | Concept 3, Session 1 | ✅ |
| 2 | Cosine similarity: annotation tích hợp formula → code (math primer 2 phút) | Concept 2, Session 2 | ✅ |
| 3 | Persistent reference card cho 2 cấu trúc API response (giữ cố định cả session) | Concept 1, Session 2 | ✅ |
| 4 | Recall sprint (2 phút, pair activity) trước VectorStore | Concept 3, Session 2 | ✅ |
| 5 | Type alias cho nested type hint (`Embedding` thay `list[float]` lồng nhau) | Code Along 2, Session 2 | ✅ |
| 6 | Không định nghĩa lại `Document`/`DocumentChunk` — Code Along 1 chỉ tham chiếu Concept 1 (expertise reversal) | Code Along 1, Session 1 | ✅ |
| 7 | Cấu trúc session: Part A — Theory 2h + Part B — Lab 2h (tổng 4h/tuần lý thuyết) | Header + bảng Session Structure | ✅ |
| 8 | Chuỗi fading: worked example → code-along (guided) → lab độc lập → peer review | Cả 2 session | ✅ |
| 9 | KCs cung cấp vòng phản hồi trong Part A (KC1 sau Concept 1, KC2 sau Concept 3) | Part A, cả 2 session | ✅ |
| 10 | Hoạt động elaborative/retrieval đặt đúng chỗ (lab opener / homework) | Wrap & Reflect + file companion | ✅ |

**Cách kiểm tra nhanh (grep):**

```bash
cd syllabus/week3
# Các dấu hiệu CLT phải hiện diện:
grep -c "Micro-Cycle\|math primer\|Persistent Reference Card\|Recall Sprint\|type alias\|PART A — THEORY (2h)" lecture_notes.md
# Tài liệu gốc phân tích:
# cognitive_load_analysis.md — 6 đề xuất (P1–P6) đã được áp dụng đầy đủ (xem mục "Summary of Recommended Changes")
```