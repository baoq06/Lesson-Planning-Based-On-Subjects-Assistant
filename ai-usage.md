## Mai Quốc Bảo (24110080, git: baoq06)

**Tool:** Google Antigravity (Claude 3.7 Sonnet / Gemini 2.0 Flash), used in VS Code, October 2026.

**How I used AI**

1. **Đo lường RAG Triad & Logging theo Session (`src/rag/eval_metrics.py`, `logs/`):**
   - AI hỗ trợ viết module chấm điểm liên tục cho 3 chỉ số RAG Triad (`faithfulness`, `answer_relevance`, `context_precision`) bằng LLM Judge, thay thế cách chấm điểm rời rạc (1.0/0.5/0.0).
   - Tôi tự triển khai cấu trúc lưu log theo từng phiên làm việc riêng biệt (`CURRENT_SESSION_DIR`, `run_*.jsonl`, xuất đồ thị `.png` và file `.csv` khi kết thúc phiên `make demo`). AI kiểm tra và chỉ ra lỗi định dạng JSONL (bỏ `indent=2` để đảm bảo mỗi bản ghi là đúng 1 dòng).
2. **Thực nghiệm so sánh Retrieval Configurations (`tests/compare_configs.py` - Điểm cộng Rubric +1):**
   - Dựa trên phản hồi của giảng viên về lỗi "kết luận viết sẵn", AI hỗ trợ tôi thay thế toàn bộ nhận xét tĩnh bằng logic tính toán động theo số liệu thực tế đo được từ 8 testcase (Hit Rate, Độ liên quan Cosine, Độ trễ API Jina), đảm bảo tính trung thực thực nghiệm giữa Hybrid Search và Dense-only.
   - Hỗ trợ thêm cờ `--resume` trong `tests/eval.py` để có thể chạy tiếp khi bị ngắt quãng mà không phải chạy lại từ đầu.
3. **Kiểm thử độc lập MCP Server từ Host thứ hai (`scripts/test_second_host.py` - Điểm cộng Rubric +1):**
   - AI hướng dẫn viết script đóng vai trò Host Python thứ hai độc lập hoàn toàn với vòng lặp ReAct, kết nối tới MCP Server qua stdio để discover danh sách tool và gọi thử 3 công cụ (`calculator`, `get_curriculum_week`, `check_late_submission_penalty`), kiểm tra trạng thái thành công/thất bại tự động.

| Part / Module | AI role | Written by |
|---|---|---|
| `tests/compare_configs.py` (+1 điểm) | Hướng dẫn công thức đánh giá động theo số liệu thực tế (bỏ kết luận viết sẵn) | Tôi viết code và chạy thực nghiệm |
| `src/rag/eval_metrics.py` & `tests/eval.py` | Hỗ trợ prompt cho RAG Triad Judge liên tục, cờ `--resume` | Tôi tích hợp vào pipeline và xuất đồ thị/CSV |
| `scripts/test_second_host.py` (+1 điểm) | Gợi ý cấu trúc Client MCP độc lập qua StdioServerParameters | Tôi viết code và kiểm chứng |

**Main prompts (examples):**
- *"File compare_configs.py bị thầy nhận xét lỗi kết luận viết sẵn mục 1.2, hãy hướng dẫn tôi sửa thành đánh giá động dựa trên kết quả đo thật của các testcase."*
- *"File log của tôi có ghi USAGE nhưng bị báo không phải JSONL hợp lệ, lỗi do đâu và sửa như thế nào?"*
- *"Viết script test_second_host.py độc lập để chứng minh MCP Server dùng được từ client thứ hai đạt điểm cộng Rubric."*
- *"Làm sao để sửa eval.py mặc định chạy từ đầu, và chỉ chạy tiếp khi có tham số --resume?"*

I can explain every line of the code listed above.

## Lê Hải Đăng (24110087, git: Hai Dang)

**Tool:** Google Antigravity, used in VS Code, October 2026.

**How I used AI**

1. **MCP Tools implementation & review (`calculator`, `get_curriculum_week`, `get_gotchas_and_best_practices`, `calculate_course_grade`, `check_late_submission_penalty`).** AI guided the safe design logic (e.g., safe AST evaluation for `calculator` instead of `eval()`). I wrote the tools myself, and AI reviewed the code, verified logic edge cases, and checked input/output schema validation.
2. **Multi-Agent pattern (`team_critic.py`).** Using the single-agent skeleton from `loop.py` and the lecture code samples, AI guided me on how to architect a two-role Critic-Refiner workflow (`Producer` and `Critic`) to evaluate answer factuality, verify document citations (`[POLICY_AIPR]`, `[LP_Wxx]`), and iteratively refine responses if ungrounded claims are detected.
3. **Session Logging management.** AI guided how to structure session-isolated logs (`CURRENT_SESSION_DIR`, environment variable `SESSION_LOG_DIR`, and separate log files for the agent loop and MCP server per run timestamp). I implemented and integrated the logging setup into the pipeline myself.

| Part | AI role | Written by |
|---|---|---|
| MCP Tools (`calculator`, `get_curriculum_week`, `get_gotchas_and_best_practices`, `calculate_course_grade`, `check_late_submission_penalty`) | Guided safe design (safe AST evaluation, exponent limits), policy edge cases, and schema validation; reviewed code | Code by me, refined based on AI review |
| Multi-Agent pattern (`src/agent/team_critic.py`, `src/schemas.py`) | Guided Critic-Refiner multi-agent architecture, critic prompt design, and Handoff Contracts (`ProducerHandoff`, `CriticReview`) | Code by me, guided by AI |
| Session Logging (`CURRENT_SESSION_DIR`, `logs/`) | Guided timestamped folder structuring and passing `SESSION_LOG_DIR` to MCP server | Implemented by me, guided by AI |

**Main prompts (examples):** 
- *"Làm sao để sửa tool calculator bỏ hàm eval() và dùng ast để an toàn hơn?"*
- *"Làm thế nào để giới hạn số mũ trong _eval_ast nhằm tránh tràn số hoặc treo máy?"*
- *"Dựa vào file loop.py và bài mẫu, hãy hướng dẫn tôi viết team_critic.py với 2 vai trò Producer và Critic"*
- *"Làm sao để cấu trúc folder log theo timestamp của từng session và đồng bộ SESSION_LOG_DIR qua MCP server?"*

I can explain every line of the code listed above.

# AI usage declaration
## Lý Trần Gia Khang (24110098, git: kangkzxw)

**Tool:** Claude Code (Anthropic), used in VS Code, October 2026.

**How I used AI**

1. **Choosing the tools.** AI suggested which MCP tools the assistant needed for our topic (an extension-request writer `draft_template_submission`, its verification tool `read_draft`, and a lesson search tool `search_lesson_content`).
2. **Understanding how each tool works.** Using the AI's explanations together with my own reading, I learned how each tool should work: input validation, the student-ID check, writing and reading back the draft file, and the JSON result returned to the agent.
3. **Writing and fixing the code.** I wrote the tools myself. AI reviewed my code and helped me fix errors, for example the `KeyError` between the template placeholders and `substitute()`, and three bugs in `hybrid_retriever.py` (`mode` field, `.chunks`, `task=` argument).
4. **Planning and checklist.** AI compared the repository with the lecturer's requirements and helped me plan the remaining work. It generated `CHECKLIST_SUA_DO_AN.md`, which I used to track the fixes.
5. **README.** AI helped write `README.md` so that it follows the required format (setup, single run command, `make eval`).
6. **Testset**: AI support me to draft the 18 questions and expected answers; I checked them against the lecture notes.

| Part | AI role | Written by |
|---|---|---|
| Tool selection | Suggested the tools | Decision by me |
| `draft_template_submission`, `read_draft`, `search_lesson_content` | Explained how they work; found and explained bugs | Code by me, fixes suggested by AI |
| `hybrid_retriever.py` | Found three bugs and explained the fixes | Fixes typed by me |
| Plan and `CHECKLIST_SUA_DO_AN.md` | Compared the project with the requirements | Checklist generated by AI |
| `README.md` | Rewrote it to match the requirements | Written by AI, checked by me |
| `tests/testset.jsonl` | Drafted questions, expected answers and expected documents | Drafted by AI, edited and checked by myself |

**Main prompts (examples):** 
- *"Hãy hỗ trợ tôi trong việc tìm ra các tools cần thiết cho đồ án của tôi"*
- *"Tại sao khúc này code tôi lại không hoạt động?"*
- *"Kiểm tra xem phần testset của tôi sửa lại đã đúng chưa?"*
- *"Hãy tìm giúp tôi các lỗi còn ẩn đọng trong code"* 
- *"Hãy sửa README cho tôi theo đúng yêu cầu đề bài"* 
- *"Kiểm tra và đối chiếu đồ án của tôi với checklist của tôi xem còn điều gì chưa được khắc phục hay không?"*

I can explain every line of the code listed above.

