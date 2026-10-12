# Tuần 8: Kiến trúc Đa tác tử (Multi-Agent) & Critic Verifier

## 1. Thông tin chung & Chuẩn đầu ra (Learning Objectives - Week 8)
- Thiết kế các mô hình phối hợp Đa tác tử: Pipeline, Handoff pattern, Orchestrator-Worker, Router, Critic-Verifier.
- Xây dựng lớp `Agent` độc lập có System Prompt chuyên biệt, tập công cụ riêng và quản lý chi phí Token Usage.
- Hiện thực hóa kiến trúc **Producer-Critic** với Handoff Contract rõ ràng (DRAFT + EVIDENCE).
- Áp dụng cơ chế **Bounded Revision** (giới hạn số vòng lặp sửa đổi, ví dụ: `REVISION_BUDGET=1`) để kiểm định chất lượng và ngăn chặn lặp vô tận.

## 2. Lý thuyết nền tảng & Kiến trúc hệ thống
### 2.1. Phân loại các mô hình Đa tác tử (Multi-Agent Patterns)
- **Pipeline Pattern (ex1_2)**: Luồng tuần tự cố định (Researcher -> Specialist -> Writer). Điểm yếu: Người viết cuối cùng có quyền quyết định tất cả, không có ai kiểm tra lại.
- **Delegate / Coordinator Pattern (ex1_3)**: Một mô hình điều phối quyết định gọi Worker nào trong lúc runtime.
- **Router Pattern (ex1_5)**: Phân loại câu hỏi thành các nhóm và định tuyến đến đúng 1 Agent duy nhất (tiết kiệm token).
- **Critic / Verifier Pattern (ex1_6)**: Kiến trúc tối thượng để chống Hallucination trong sản xuất:
  - Một **Worker Agent (Producer)** tạo ra Bản thảo (Draft) và thu thập Bằng chứng (Evidence).
  - Một **Critic Agent (Verifier)** hoạt động độc lập làm nhiệm vụ thẩm định.

### 2.2. Nguyên tắc độc lập & Handoff Contract của Critic Agent
- **Sự độc lập tuyệt đối**: Critic Agent **KHÔNG CÓ CÔNG CỤ TÌM KIẾM** và **TUYỆT ĐỐI KHÔNG TỰ VIẾT LẠI CÂU TRẢ LỜI**.
- **Handoff Contract**: Đầu vào của Critic chỉ bao gồm:
  1. Yêu cầu của người dùng (Request).
  2. Bản thảo câu trả lời (Draft Answer).
  3. Bằng chứng tài liệu đã truy xuất (Evidence).
- **Phán quyết của Critic (Critic Verdict)**:
  - `VERDICT: PASS`: Mọi khẳng định đều có nguồn gốc từ Evidence; câu hỏi ngoài phạm vi đã được từ chối an toàn.
  - `VERDICT: REVISE | REASON: <lỗi>`: Phát hiện thông tin bịa đặt không có trong tài liệu hoặc thiếu trích dẫn nguồn.
- **Bounded Revision Budget**: Tối đa cho phép sửa đổi $1$ vòng (`REVISION_BUDGET=1`). Nếu sau 1 vòng vẫn không đạt, dừng lại để tránh cạn kiệt ngân sách token và vòng lặp vô tận.

## 3. Các bài tập Lab & Mã nguồn mẫu từng bước (Step-by-step Exercises)
### Exercise 8.1 — Lớp Agent độc lập (ex1_1_agent_class.py)
```python
class Agent:
    def __init__(self, name: str, system_prompt: str, tools: list = None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []

    def run(self, task: str) -> str:
        # Vòng lặp Agent độc lập có theo dõi token usage
        return "Kết quả xử lý từ " + self.name
```

### Exercise 8.2 — Handoff Contract & Phán quyết Critic (ex1_6_critic.py)
```python
from dataclasses import dataclass, field

@dataclass
class CriticVerdict:
    verdict: str                  # "PASS" hoặc "REVISE"
    feedback: str = ""            # Chi tiết lỗi cần sửa nếu REVISE
    unsupported_claims: list[str] = field(default_factory=list)

CRITIC_PROMPT = """Bạn là Tác tử Thẩm định Độc lập.
Bạn KHÔNG tự viết câu trả lời.
Nhiệm vụ: Đối chiếu BẢN THẢO với BẰNG CHỨNG.
Dòng cuối cùng bắt buộc là:
VERDICT: PASS
hoặc
VERDICT: REVISE | REASON: <nêu ngắn gọn lỗi>
"""
```

### Exercise 8.3 — Điều phối Producer-Critic với Bounded Revision
```python
def run_critic_pipeline(user_question: str, revision_budget: int = 1):
    # Bước 1: Worker tạo bản thảo đầu tiên
    draft_response = worker_agent.run(user_question)

    # Bước 2: Thẩm định chất lượng có giới hạn
    for r in range(revision_budget):
        verdict = critic_agent.evaluate(
            request=user_question,
            draft=draft_response.answer,
            evidence=draft_response.citations
        )
        if verdict.verdict == "PASS":
            return draft_response  # Đạt chuẩn

        # Nếu REVISE: Worker sửa lại bản thảo dựa trên feedback của Critic
        revision_prompt = f"{user_question}\n[LƯU Ý TỪ CRITIC]: {verdict.feedback}"
        draft_response = worker_agent.run(revision_prompt)

    return draft_response
```

## 4. Các bẫy thường gặp & Nguyên tắc thiết kế (Gotchas & Best Practices)
- **Bẫy Critic tự ý viết lại**: Nếu để Critic tự sửa văn bản, Critic sẽ có xu hướng tự sinh ra ảo giác mới; Critic chỉ được phép đưa ra feedback để Worker tự sửa.
- **Bẫy Không giới hạn số vòng sửa (Infinite Debate)**: Nếu hai Agent tranh cãi qua lại mà không có `REVISION_BUDGET`, hệ thống sẽ cạn kiệt ngân sách API và treo phiên người dùng.

## 5. Nhiệm vụ tự học & Đánh giá năng lực (Independent Labs)
- Thử nghiệm kỹ thuật **Fault Injection (Cố tình tiêm lỗi bịa đặt)** vào bản thảo của Worker và kiểm chứng xem Critic Verifier có bắt đúng và yêu cầu sửa hay không.
