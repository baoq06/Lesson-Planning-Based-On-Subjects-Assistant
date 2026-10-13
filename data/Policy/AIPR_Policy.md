# Course Policy: AI Programming with Python (AIPR Course Policy)

## 1. Course Information & General Rules
- **Course name:** AI Programming with Python (AIPR).
- **Course code:** AIPR_24110FIE
- **Duration:** 7.5 weeks (2 sessions per week, 4 hours per session) -> 60 contact hours.
- **Announcement & support channels:** All announcements about assignments, deadlines and learning materials are posted on the university LMS portal and on the AIPR Learning Assistant (AI assistant system).

## 2. Assessment Structure & Grading Scheme
The final course grade is calculated on a 10-point scale with the following weights:

- **1. Attendance (5%):**
  - Assesses full and punctual attendance at sessions and direct participation in interactive activities (Recall Sprints, Q&A).
  - A student who is absent from more than 20% of all sessions is barred from the exam and is not eligible to defend the final project.

- **2. Lab Exercises (15%):**
  - Aggregates the results of in-class lab exercises from Week 1 to Week 7 (`ex1_1`, `ex1_2`, `ex2_1`, `lab_sheet`).
  - Students must complete their source code and pass all automated tests (unit tests) for each week.

- **3. Periodic Tests (30%):**
  - Consists of 2 major tests (15% each):
    - *Test 1 (Midterm - Week 4):* Assesses the skills of building a RAG pipeline, Chunking, Embeddings and Hybrid Search (BM25 + Vector).
    - *Test 2 (Week 7):* Assesses the skills of programming the Agentic Loop, Function Calling and the Model Context Protocol (MCP).

- **4. Final Project & Oral Defense (50%):**
  - Build a complete Agentic AI system that combines: a RAG Engine, MCP Tools (with verifiable Side Effects), and a Multi-Agent model (Producer-Critic Verifier).
  - The score includes: evaluation of the submission package, the unit test / quantitative benchmark pass rate, and an **individual oral defense** held directly with the lecturer.

## 3. Final Project Rules & Format
- **Format:**
  - Students must work in groups of **2 to 3 students per group**.
  - The project must satisfy all technical and domain criteria set by the lecturer.
  - Each group member must have a clearly defined portion of work, recorded in the Workload Breakdown table.
- **Individual Oral Defense:**
  - The lecturer will examine each student directly and independently.
  - Each member's final score is based on depth of understanding of the source code, ability to justify the architecture design, and actual contribution (group scores are not averaged across members).
- **Standard Submission Package (Deliverables Bundle):**
  When submitting the final project, each group must package all of the following:
  1. **Source Code:** The complete source code with a GitHub Repository link or a `.zip` file.
  2. **Project report (`report.docx`):** A document describing the problem, system architecture, data flow diagram, experiments and results.
  3. **Execution logs (`logs/`):** A folder with the traces of real system runs (Agent loop trace, MCP calls, retriever logs).
  4. **Standard test set (`testset.jsonl`):** A benchmark dataset of at least more than 12 test cases used for quantitative measurement (Faithfulness, Hit Rate, Refusal Rate).
  5. **AI usage report (`AI_USAGE.md`):** A document that transparently declares the use of GenAI tools during the project.
  6. **Deployment guide (`README.md`):** A document with environment setup instructions, environment variable configuration and detailed demo run commands.

## 4. Late Submission & Deadline Extension Policy
- **Standard late submission policy:**
  - Late by 1 to 24 hours (Day 1): **10% deduction** from the total score of the submission.
  - Late by 25 to 48 hours (Day 2): **20% deduction** from the total score of the submission.
  - Late by 49 to 72 hours (Day 3): **30% deduction** from the total score of the submission.
  - Late by more than 72 hours (after 3 days): **score of 0** for that assignment. The submission portal closes automatically.
- **Deadline extension policy:**
  - **The only condition:** The university and the course **ONLY consider extensions for force majeure medical cases** (illness or hospitalization with a valid medical certificate / medical record from a doctor or medical facility).
  - **Maximum extension period:** No more than **5 days** after the original deadline.
  - **Request procedure:** Students must submit a request with a photo of the medical evidence through the Learning Assistant system or by email to the lecturer before the submission deadline expires.

## 5. Academic Integrity & GenAI Policy
The course encourages students to make use of large language models (LLMs), but strict academic honesty is mandatory:
- **Permitted use of GenAI:**
  - Students **ARE ALLOWED** to use LLM tools (ChatGPT, Claude, Groq, Gemini, Copilot) to: look up theoretical concepts, explain programming errors, optimize algorithms, or act as a critic (Code Review).
  - Every use of GenAI must be recorded in the `AI_USAGE.md` file (including: tool name, link/model version, the main prompts used, and a description of the code that was generated/optimized).
  - Students must fully own and understand 100% of every line of the source code they submit.
- **Violations & penalties:**
  - Copying the source code of another group, or having an outside person write the work, is strictly prohibited.
  - During the Individual Oral Defense, if a student **cannot explain the source code** generated by AI, or plagiarism is detected:
    - The submission will be **DEDUCTED 100% OF ITS SCORE (receives 0 points)** immediately.
    - The violation file will be forwarded to the Faculty Disciplinary Council for course-level disciplinary action.

## 6. Grade Appeal & Complaint Procedure
- After the lecturer publishes the score for each assignment/test, students have **03 working days** to submit an appeal request if they find an error in the grading process.
- The appeal must clearly state the question, the part of the code or the grading criterion to be reviewed, together with a specific reason.
