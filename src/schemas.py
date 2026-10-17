from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Literal, List
from datetime import datetime
from typing_extensions import TypedDict

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
    embedding: Optional[list[float]] = None

# schema RetrievalResult: kết quả trả về khi xong thuật tìm kiếm
@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    rank: int = 1
    provenance: str = "" # trích nguồn
    mode: str = ""

# --- khối mpc server - từ từ viết

#schema DraftResult: kết quả của tool draft_template_submission
@dataclass
class DraftResult:
    path: str
    overwritten: bool
    chars: int
    note: str

#schema SearchHit: một đoạn bài giảng tìm được
@dataclass
class SearchHit:
    doc_id: str
    chunk_id: str
    week: int
    score: float
    text: str

#schema SearchResult: kết quả của tool search_lesson_content
@dataclass
class SearchResult:
    query: str
    mode: str
    found: bool
    best_relevance: float
    hits: list[SearchHit] = field(default_factory=list)

#schema TimelineItem: cấu trúc cho từng mốc hoạt động trong buổi học
class TimelineItem(TypedDict):
    time: str
    activity: str
    detail: str

# schema ReadDraftResult: kết quả của tool read_draft
@dataclass
class ReadDraftResult:
    path: str
    chars: int
    content: str
# schema ExportPlanResult: kết quả trả về của tool export_lesson_plan
@dataclass
class ExportPlanResult:
    status: str
    week: int
    title: str
    json_path: str
    md_path: str
    note: str
# schema UsageStats: thống kê token chuẩn hóa
class UsageStats(TypedDict):
    calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    max_prompt: int

# --- khối vòng lặp agent react & memory (viết tạm thời, có thể không xài)

# schema ToolCall: lệnh gọi tool do llm quyết
@dataclass 
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

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
    # sources: list[str] = field(default_factory=list)
    # verified: bool = False
    usage:  UsageStats = field(default_factory=dict)

@dataclass
# Handoff contract gói dữ liệu bàn giao từ Producer sang Critic
class ProducerHandoff:
    question: str
    draft: str
    evidence: str
    revision_budget: int = 1

    # chuyển contract thành prompt sạch cho Critic
    def to_critic_prompt(self) -> str:
        return(
            f"REQUEST:\n{self.question}\n\n"
            f"RAW EVIDENCE: \n{self.evidence if self.evidence.strip() else '(No tool evidence gathered)'}\n\n"
            f"DRAFT:\n{self.draft}"
        )


@dataclass
# Handoff contract kết quả đánh giá từ Critic Agent
class CriticReview:
    verdict: Literal["PASS", "REVISE"]
    critique: str
    issues: List[str] = field(default_factory=list)
    suggested_fix: Optional[str] = None
