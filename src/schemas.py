from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime

# --- khối dữ liệu rag và chunking

# schema Document: nguyên văn tài liệu
@dataclass
class Document:
    content: str
    doc_id: str = ""
    title: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)
    created_at: Optional[datetime] = field(default_factory=datetime.now)

# schema DocumentChunk: chunk được cắt ra từ tài liệu
@dataclass
class DocumentChunk:
    content: str
    chunk_id: str = ""
    doc_id: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)
    # embedding: Optional[list[float]] = None

# schema RetrievalResult: kết quả trả về khi xong thuật tìm kiếm
@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    rank: int = 1
    provenance: str = "" # trích nguồn

# --- khối mpc server - từ từ viết

# --- khối vòng lặp agent react & memory (viết tạm thời, có thể không xài)

# schema ToolCall: lệnh gọi tool do llm quyết
@dataclass 
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

#schema ToolResult: llm trả về
@dataclass
class ToolResult:
    call_id:  str
    tool_name: str
    output: str
    id_error: str = False

#schema AgentStep: 1 vòng lặp ReAct
@dataclass
class AgentStep:
    step_num: int
    thought: str 
    tool_call: Optional[ToolCall] = None # Action 
    observation: Optional[str] = None

#schema AgentResponse: câu trả lời hoàn chỉnh
@dataclass 
class AgentResponse: 
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    verifired: bool = False



