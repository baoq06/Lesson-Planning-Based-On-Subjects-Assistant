## Mai Quốc Bảo (24110080, git: baoq06)

**Tool:** Google Antigravity (Claude 3.7 Sonnet / Gemini 2.0 Flash), used in VS Code, October 2026.

**How I used AI**
1. **RAG Triad Evaluation & Session-Isolated Logging (`src/rag/eval_metrics.py`, `logs/`):**
   - AI guided the design of continuous LLM-as-a-judge scoring for the three RAG Triad metrics (`faithfulness`, `answer_relevance`, `context_precision`), replacing the discrete step scoring (1.0/0.5/0.0).
   - I implemented the session-isolated logging structure (`CURRENT_SESSION_DIR`, `run_*.jsonl`, and automated generation of `.png` chart and `.csv` summary upon exiting the demo loop). AI reviewed the output format and identified the JSONL compliance bug (removing `indent=2` to ensure exactly one JSON record per line).
2. **Quantitative Retrieval Configuration Comparison (`tests/compare_configs.py` - Rubric Bonus +1):**
   - In response to the instructor's feedback regarding "pre-written conclusions", AI guided me in replacing all hardcoded qualitative statements with dynamic empirical logic computed directly from the benchmark runs across 8 test cases (Hit Rate, average Cosine Relevance, and Jina API Latency).
   - AI also assisted in implementing the `--resume` flag logic in `tests/eval.py`, allowing incomplete test runs to seamlessly resume without re-running completed cases.
3. **Independent Second Host Interoperability Testing (`scripts/test_second_host.py` - Rubric Bonus +1):**
   - AI provided the architectural pattern for a standalone second Python client running outside the primary ReAct agent loop, connecting to the MCP Server over standard I/O (stdio) to dynamically discover tools and test executions (`calculator`, `get_curriculum_week`, and `check_late_submission_penalty`) with automated success/failure checks.

| Part / Module | AI role | Written by |
|---|---|---|
| `tests/compare_configs.py` (+1 bonus) | Guided dynamic evaluation formulas based on empirical benchmark metrics (removed pre-written conclusions) | Implemented and benchmarked by me |
| `src/rag/eval_metrics.py` & `tests/eval.py` | Assisted in prompting for continuous RAG Triad Judge and the `--resume` flag logic | Integrated into pipeline with CSV/chart generation by me |
| `scripts/test_second_host.py` (+1 bonus) | Provided independent MCP client template via `StdioServerParameters` | Implemented and verified by me |
**Main prompts (examples):**
- *"The instructor commented that compare_configs.py had pre-written conclusions in section 1.2; guide me on how to rewrite it with dynamic evaluation based on actual benchmark metrics."*
- *"My log records token usage, but the instructor flagged that it is not valid JSONL. What is the root cause and how do I fix it?"*
- *"Write an independent test_second_host.py script to demonstrate MCP Server interoperability from a second Python client for rubric bonus points."*
- *"How do I modify eval.py so it defaults to running from scratch and only resumes when the --resume flag is explicitly passed?"*

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

## Lý Trần Gia Khang (24110098, git: kangkzxw)

**Tool:** Claude Code (Anthropic) and Antigravity (Google), used in VS Code, October 2026.

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

