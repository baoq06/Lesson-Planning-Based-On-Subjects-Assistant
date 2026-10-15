import os
import sys
import json
import logging
import re
import ast
import operator 
from datetime import datetime, date
from dataclasses import asdict
from typing import Optional, Literal
from pathlib import Path
from mcp.server.mcpserver import MCPServer
from string import Template
from src.schemas import DraftResult, SearchHit, SearchResult, TimelineItem
from src.rag.vector_store import VectorStore
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.load_documents import load_all_course_documents


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
TEMPLATE_PATH = REPO_ROOT / "data" / "templates" / "extension_request.txt"
DRAFTS_DIR = REPO_ROOT / "data" / "drafts"
PLANS_DIR = REPO_ROOT / "data" / "plans"
POLICY_PATH = REPO_ROOT / "data" / "Policy" / "AIPR_Policy.md"
MAX_EXTENSION_DAYS = 5
STUDENT_ID_PATTERN = re.compile(r"^\d{8}$")

MEMORY_PATH = REPO_ROOT / "data" / "mcp_memory.json"
DATA_DIR = REPO_ROOT / "data"
AIPR_DIR = DATA_DIR / "lecture-notes"
CACHE_PATH = REPO_ROOT / "data" / "vector_store_cache.json"

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

logging.basicConfig(
    stream=sys.stderr, 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("mcp_server")

_retriever: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    """Dựng retriever ở lần gọi đầu tiên: đọc cache nếu có, không thì embed tài liệu rồi lưu cache."""
    global _retriever
    if _retriever is not None:
        return _retriever

    store = VectorStore()
    if CACHE_PATH.exists():
        store.load(str(CACHE_PATH))
        logger.info("Loaded %d chunks from cache.", len(store._chunks))
    else:
        logger.info("Cache not found. Indexing course documents...")
        store.add_documents_batch(load_all_course_documents(DATA_DIR))
        store.save(str(CACHE_PATH))
        logger.info("Indexed and cached %d chunks.", len(store._chunks))

    _retriever = HybridRetriever(vector_store=store)
    return _retriever

MIN_RELEVANCE = 0.35
def _is_lesson_chunk(chunk, week: int) -> bool:
    """Chỉ giữ đoạn bài giảng; nếu week > 0 thì chỉ giữ đúng tuần đó."""
    if chunk.metadata.get("type") != "curriculum":
        return False
    return week == 0 or chunk.metadata.get("week") == week

mcp = MCPServer("LessonPlanningAssistantServer")

def _eval_ast(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        op_type = type(node.op)
        if op_type in OPERATORS:
            return OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        op_type = type(node.op)
        if op_type in OPERATORS:
            return OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported operator: {op_type.__name__}")
    else:
        raise TypeError(f"Unsupported syntax: {type(node).__name__}")

@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the precise calculated result (READ-ONLY, changes nothing).

    Use when exact numerical computations, grade weighting, percentages, or timeline additions
    are required to eliminate LLM arithmetic calculation errors.
    Do NOT call this tool for non-mathematical text or non-trivial Python logic.
    Required information: a valid mathematical string expression containing numbers and standard operators (+, -, *, /, **, etc.).
    If the expression is missing or incomplete, ASK the user to provide the exact numbers or formulas to compute.
    After a successful call, use the returned numerical value directly in your response or subsequent calculations.
    On failure returns a string starting with "Error:".

    Args:
        expression: A valid mathematical string to evaluate (e.g. '0.05 * 9 + 0.15 * 8.5', '24 * 3 + 12').
    """
    logger.info(f"Executing 'calculator' tool with expression: '{expression}'")
    try:
        cleaned_expr = expression.strip().replace("^", "**")
        tree = ast.parse(cleaned_expr, mode='eval')
        result = _eval_ast(tree.body)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        logger.error(f"Error: Failed to evaluate expression '{expression}': {e}")
        return f"Error: Failed to evaluate expression '{expression}': {e}"

@mcp.tool()
def search_lesson_content(query: str, mode: Literal["dense", "hybrid"] = "hybrid", top_k: int = 4, week: int = 0,) -> str:
    """Search the AIPR lecture notes (Weeks 1-6) for passages relevant to a question (READ-ONLY, changes nothing).

    Use this FIRST for any question about course content: concepts, code examples, techniques,
    common mistakes. Answer ONLY from the returned passages and ALWAYS cite their doc_id (and week).
    If "found" is false, the topic is NOT covered in the course: refuse politely and do NOT answer
    from your own knowledge. Lecture notes for Weeks 7-8 are not available.
    Do NOT use this tool for grading, late penalties or extension requests (use the dedicated tools).
    Returns JSON with query, mode, found, best_relevance and hits (doc_id, chunk_id, week, score, text).
    On failure returns a string starting with "Error:".

    Args:
        query: the student's question or key terms.
        mode: "hybrid" (keywords + meaning, best for exact names like 'safe_substitute')
            or "dense" (meaning only, best for naturally phrased questions).
        top_k: number of passages to return (1-8).
        week: 0 to search all weeks, or 1-6 to search only that week.
    """
    logger.info(f"Executing 'search_lesson_content' query='{query}' mode={mode} top_k={top_k} week={week}")
    try:
        query = query.strip()
        if not query:
            logger.info(f"empty query")
            return "Error: query must not be empty."
        if mode not in ("dense", "hybrid"):
            logger.info(f"invalid mode")
            return "Error: mode must be 'dense' or 'hybrid'."
        if not 0 <= week <= 6:
            logger.info(f"invalid week")
            return "Error: week must be 0 (all weeks) or between 1 and 6."
        top_k = max(1, min(top_k, 8))

        retriever = get_retriever()
        pool = len(retriever.vector_store._chunks)

        ranked = retriever.search(query, top_k=pool, mode=mode)
        hits = [(c, s) for c, s in ranked if _is_lesson_chunk(c, week)][:top_k]

        if mode == "dense":
            best = hits[0][1] if hits else 0.0
        else:
            dense = [(c, s) for c, s in retriever.search(query, top_k=pool, mode="dense")
                     if _is_lesson_chunk(c, week)]
            best = dense[0][1] if dense else 0.0

        result = SearchResult(
            query=query,
            mode=mode,
            found=bool(hits) and best >= MIN_RELEVANCE,
            best_relevance=round(best, 4),
            hits=[
                SearchHit(
                    doc_id=c.doc_id,
                    chunk_id=c.chunk_id,
                    week=c.metadata.get("week", 0),
                    score=round(s, 4),
                    text=c.content,
                )
                for c, s in hits
            ],
        )
        logger.info(f"Successfully searched for query '{query}' and found {len(hits)} hits.")
        return json.dumps(asdict(result), ensure_ascii=False)
    except Exception as e:
        logger.exception("search_lesson_content failed")
        return f"Error: search_lesson_content failed: {e}"

@mcp.tool()
def get_curriculum_week(week: int, full_content: bool = False) -> str:
    """Read curriculum lecture notes, session structures, and learning objectives for a course week (READ-ONLY, changes nothing).

    Use when planning a lesson, designing exercises, checking topics covered, or drafting slides for a given week.
    Currently, lecture notes are available ONLY for Weeks 1 to 6.
    Do NOT call this tool for weeks outside the 1 to 6 range (e.g. Week 7, Week 8, or Week 0).
    Required information: the week number (1-6). The 'full_content' flag is optional (defaults to False).
    If the user does not specify the week number, ASK the user which week (1 to 6) they want to inspect.
    Keep 'full_content=False' for structured summaries; set 'full_content=True' ONLY when full instructor scripts and code walkthroughs are needed.
    After a successful call, synthesize the learning objectives and session structure to answer the user's request.
    On failure returns a string starting with "Error:".
    Args:
        week: The curriculum week number (must be an integer from 1 to 6).
        full_content: False to retrieve a structured overview; True to read the entire raw markdown file.
    """
    logger.info(f"Executing 'get_curriculum_week' for week {week}, full_content={full_content}")
    if week < 1 or week > 6:
        logger.warning(f"Invalid week requested: {week}")
        return f"Error: Available lecture notes are currently only for Weeks 1 to 6 (requested week {week})."
    
    file_path = AIPR_DIR / f"w{week}_lecture_notes.md"
    if not file_path.exists():
        logger.error(f"Lecture note file not found: {file_path}")
        return f"Error: Lecture note file not found for Week {week} at {file_path.name}."

    try: 
        content = file_path.read_text(encoding="utf-8")
        logger.info(f"Successfully read lecture notes for Week {week} ({len(content)} characters).")

        summary_sections = []
        
        header_match = re.search(r"(# Week \d+.*?)(?=## Session 1|\Z)", content, re.DOTALL)
        if header_match:
            summary_sections.append(header_match.group(1).strip())

        s1_match = re.search(r"(## Session 1.*?### Session Structure.*?\|[\s\S]*?)(?=### 1\.|\Z)", content)
        if s1_match:
            summary_sections.append(s1_match.group(1).strip())

        s2_match = re.search(r"(## Session 2.*?### Session Structure.*?\|[\s\S]*?)(?=### 1\.|\Z)", content)
        if s2_match:
            summary_sections.append(s2_match.group(1).strip())

        if summary_sections:
                result = "\n\n---\n\n".join(summary_sections)
                if len(result) > 2500:
                    result = result[:2500] + "\n\n...(truncated. Use 'search_lesson_content' for specific sections)."
                return result

        return content[:1500] + "\n\n...(truncated. Use 'search_lesson_content' to query specific topics)."
    
    except Exception as e:
        logger.error(f"Error occurred while reading curriculum for Week {week}: {e}")
        return f"Error: Failed to read curriculum file for Week {week}: {e}"

@mcp.tool()
def get_gotchas_and_best_practices(week: int) -> str:
    """Extract common student mistakes, pitfalls, and pedagogical interventions for a course week (READ-ONLY, changes nothing).
    
    Use when preparing interactive Q&A, designing formative assessments, anticipating student bugs,
    or advising instructors on tricky concepts for a given week.
    Currently available ONLY for Weeks 1 to 6.
    Do NOT call this tool for general curriculum overviews or for weeks outside the 1 to 6 range.
    Required information: the week number (1-6).
    If the week number is not specified, ASK the user which week they need the gotchas and common mistakes for.
    After a successful call, use the extracted common mistakes and interventions to give actionable teaching recommendations.
    On failure returns a string starting with "Error:".
    Args:
        week: The curriculum week number (must be an integer from 1 to 6).
    """
    logger.info(f"Executing 'get_gotchas_and_best_practices' for week {week}")
    if week < 1 or week > 6:
        logger.warning(f"Invalid week requested: {week}")
        return f"Error: Available lecture notes are currently only for Weeks 1 to 6 (requested week {week})."
    
    file_path = AIPR_DIR / f"w{week}_lecture_notes.md"
    if not file_path.exists():
        logger.error(f"Lecture note file not found: {file_path}")
        return f"Error: Lecture note file not found for Week {week} at {file_path.name}."

    try: 
        content = file_path.read_text(encoding="utf-8")
        pattern = r"(## Common Student Mistakes & Interventions.*?)(?=## KC Answer Key|## \S|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

        if match:
            logger.info(f"Successfully extracted 'Common Student Mistakes & Interventions' for Week {week}.")
            return f"### Gotchas & Common Student Mistakes (Week {week}):\n\n" + match.group(1).strip()

        logger.warning(f"Common Student Mistakes section not found for Week {week}.")
        return f"Error: 'Common Student Mistakes & Interventions' section not found in Week {week} lecture notes."
    
    except Exception as e:
        logger.error(f"Error occurred while extracting gotchas for Week {week}: {e}")
        return f"Error: Failed to read lecture notes for Week {week}: {e}"

@mcp.tool()
def calculate_course_grade(
    attendance: float,
    labs: float,
    test1: float,
    test2: float,   
    final_project: float,
    absence_rate: float = 0.0,
) -> str:
    """Calculate the final AIPR course grade per course policy [POLICY_AIPR] (READ-ONLY, changes nothing).
    Use when computing a student's final weighted course score or verifying policy compliance.
    Per AIPR Course Policy Section 2 [POLICY_AIPR]: 
    - Attendance (5%), Practical Labs (15%), Test 1 Midterm (15%), Test 2 (15%), Final Project & Oral Defense (50%).
    - Note on passing threshold: The course policy does NOT specify a pass/fail threshold (e.g. >= 5.0).
    - ATTENDANCE POLICY RULE [POLICY_AIPR]: Students with absence_rate > 20% are barred from the final exam / not eligible to defend the final project. The policy does NOT specify setting final grade to 0.0.
    
    CRITICAL INSTRUCTION FOR MISSING INPUTS:
    - Required information: attendance, labs, test1, test2, final_project (all on a 0-10 scale), and absence_rate (0-100%).
    - absence_rate does NOT have a default value. If the user does not provide absence_rate or any component grade, DO NOT call this tool and NEVER assume/invent values (such as assuming absence_rate is 0%). ASK the user to clarify first.
    - On failure returns a string starting with "Error:".
    Args:
        attendance: Attendance component score on a 0-10 scale.
        labs: Practical lab assignment score on a 0-10 scale.
        test1: Midterm Test 1 score on a 0-10 scale.
        test2: Test 2 score on a 0-10 scale.
        final_project: Final Project & Oral Defense score on a 0-10 scale.
        absence_rate: Student absence percentage (0.0 to 100.0). Required, no default.
    """
    logger.info(
        f"Executing 'calculate_course_grade' with attendance={attendance}, labs={labs}, "
        f"test1={test1}, test2={test2}, final_project={final_project}, absence_rate={absence_rate}%"
    )
    try:
        grade_components = {
            "Attendacne": attendance,
            "Labs": labs,
            "Test 1": test1,
            "Test 2": test2,
            "Final project": final_project,
        }
        for name, score in grade_components.items():
            if not (0.0 <= score <= 10.0):
                return (
                    f"Error: Score {name} is invalid ({score}). "
                    f"All scores must be from 0.0 to 10.0."
                )

        if not (0.0 <= absence_rate <= 100.0):
            return (
                f"Error: Invalid absence rate ({absence_rate}%). "
                f"Absence rate must be form 0.0% to 100.0% [POLICY_AIPR]."
            )

        if absence_rate > 20.0:
            logger.info(f"Student exceeded absence limit: absence_rate={absence_rate}% > 20%")
            return (
                f" Attendance warning:\n"
                f"- Absence rate: {absence_rate:.1f}% (exceeds the limit 20%).\n"
                f"- Current status: Banned from taking the final test / protecting the final project.\n"
            )

        final_grade = (
            (attendance * 0.05)
            + (labs * 0.15)
            + (test1 * 0.15)
            + (test2 * 0.15)
            + (final_project * 0.50)
        )
        final_grade = round(final_grade, 2)
        
        logger.info(f"Calculated final grade: {final_grade}")
        return (
            f"COURSE EVALUATION RESULT (AIPR):\n"
            f"- Attendance (5%): {attendance}\n"
            f"- Practical Labs (15%): {labs}\n"
            f"- Test 1 - Midterm (15%): {test1}\n"
            f"- Test 2 (15%): {test2}\n"
            f"- Final Project & Oral Defense (50%): {final_project}\n"
            f"- Absence Rate: {absence_rate:.1f}% (Compliant)\n"
            f"----------------------------------------\n"
            f"FINAL GRADE: {final_grade} / 10\n"
            #f"Status: {'PASSED' if passed else 'FAILED'}"
        )
    except Exception as e:
        logger.error(f"Error: Failed to calculate course grade: {e}")
        return f"Error: Failed to calculate course grade: {e}"
    
@mcp.tool()
def check_late_submission_penalty(hours_late: int) -> str:
    """Calculate the penalty deduction percentage for an assignment submitted past the deadline (READ-ONLY, changes nothing).

    Use when a student or instructor asks about late penalties, deduction rates, or portal closing rules.
    Per AIPR Course Policy Section 4:
    - <= 0 hours late: 0% penalty (On time).
    - 1 to 24 hours late (Day 1): Deduct 10% of total score.
    - 25 to 48 hours late (Day 2): Deduct 20% of total score.
    - 49 to 72 hours late (Day 3): Deduct 30% of total score.
    - > 72 hours late: Submission rejected (0 points awarded, portal closed).
    Do NOT call this tool when the student is asking for a medical deadline extension (use 'draft_template_submission' instead).
    Required information: the number of hours late as an integer.
    If the late duration is missing, ASK the user how many hours past the deadline the submission was made.
    After a successful call, explain the exact penalty deduction percentage and the policy status to the student or instructor.
    On failure returns a string starting with "Error:".
    Args:
        hours_late: Number of hours elapsed past the official deadline (e.g. 5, 36, 80).
    """
    logger.info(f"Executing 'check_late_submission_penalty' for hours_late={hours_late}")
    try:
        if hours_late <= 0:
            return "Submission is on time. No penalty applied (Penalty: 0%)."
        
        if hours_late <= 24:
            penalty = 10
            desc = "Late by 1 to 24 hours (Day 1)"
        elif hours_late <= 48:
            penalty = 20
            desc = "Late by 25 to 48 hours (Day 2)"
        elif hours_late <= 72:
            penalty = 30
            desc = "Late by 49 to 72 hours (Day 3)"
        else:
            logger.info(f"Late submission rejected (>72h): {hours_late} hours")
            return (
                f"SUBMISSION REJECTED: Over 72 hours late ({hours_late} hours late).\n"
                f"- Policy: AIPR Course Policy Section 4.\n"
                f"- Status: Submission portal is closed.\n"
                f"- Grade: 0 points awarded for this assignment."
            )

        logger.info(f"Late penalty applied: -{penalty}% ({desc})")
        return (
            f"LATE SUBMISSION PENALTY APPLIED:\n"
            f"- Late Duration: {hours_late} hours ({desc}).\n"
            f"- Policy: AIPR Course Policy Section 4.\n"
            f"- Penalty: DEDUCT {penalty}% of the assignment's total score."
        )
    
    except Exception as e:
        logger.error(f"Error occurred while checking late submission penalty: {e}")
        return f"Error: Failed to check late submission penalty: {e}"

@mcp.tool()
def draft_template_submission(student_id: str, student_name: str, assignment: str, days_requested: int, reason: str, has_medical_proof: bool) -> str:
    """Draft a deadline-extension request letter and SAVE it as a .txt file (this is only a DRAFT, not submitted).

    Use when a student cannot submit an assignment on time because of a medical reason
    (illness, accident, hospitalization...).
    Per the course policy, the request must be sent BEFORE the deadline; if the student is already
    past the deadline, warn them that the request may not be accepted.
    Do NOT call this tool when the reason is not medical (e.g. overslept, forgot, did not notice...).
    Required information: student ID, student name, assignment name, number of extension days,
    reason, and a valid medical certificate (mandatory).
    If any information is missing, ASK the student until everything is provided. NEVER invent information.
    After a successful call, you MUST call read_draft with the same student_id to verify the draft,
    and only then tell the student that this is a DRAFT and has NOT been submitted.
    Calling this tool again with the same student_id OVERWRITES the previous draft.
    On failure returns a string starting with "Error:".

    Args:
        student_id: student ID with exactly 8 digits, e.g. "24110001".
        student_name: full name of the student writing the request.
        assignment: name of the assignment to extend.
        days_requested: number of extension days (1-5).
        reason: short description of the medical reason.
        has_medical_proof: True ONLY if the student confirms having a valid medical certificate.
            NEVER assume True.
    """
    logger.info(f"Executing 'draft_template_submission' for student {student_id}")
    try:
        student_id, student_name = student_id.strip(), student_name.strip()
        assignment, reason = assignment.strip(), reason.strip()

        if not (student_id and student_name and assignment and reason):
            logger.warning("Missing required arguments in extension draft submission.")
            return "Error: Student ID, Student Name, Assignment name, and Reason are required."
        
        if not STUDENT_ID_PATTERN.fullmatch(student_id):
            logger.warning(f"Invalid Student ID format: '{student_id}'")
            return f"Error: Student ID '{student_id}' is invalid. It must contain exactly 8 digits."

        if not has_medical_proof:
            logger.warning("Extension request rejected due to missing medical proof.")
            return "Error: A valid medical certificate is strictly required per Course Policy Section 4."
        
        if not 1 <= days_requested <= MAX_EXTENSION_DAYS:
            logger.warning(f"Requested days out of range: {days_requested}")
            return f"Error: Requested days must be between 1 and {MAX_EXTENSION_DAYS} days."

        if not TEMPLATE_PATH.exists():
            logger.error(f"Template file not found at: {TEMPLATE_PATH}")
            return "Error: Template file not found."
        content = Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(
            student_name=student_name,
            student_id=student_id,
            assignment=assignment,
            days_requested=days_requested,
            reason=reason,
            today=date.today().strftime("%Y-%m-%d"),
        )

        DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
        path = DRAFTS_DIR / f"draft_submission_{student_id}.txt"
        overwritten = path.exists()
        path.write_text(content, encoding="utf-8")
        logger.info(f"Draft saved successfully to: {path} (overwritten={overwritten})")

        return json.dumps(
            asdict(DraftResult(
                path=str(path),
                overwritten=overwritten,
                chars=len(content),
                note="This is an unsubmitted draft. Call 'read_draft' with this student ID to verify the content.",

            )),
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("Failed to create draft extension request.")
        return f"Error: Failed to create draft extension request: {e}"

@mcp.tool()
def read_draft(student_id: str) -> str:
    """Read back the saved extension-request draft of a student (READ-ONLY, changes nothing).

    Use this RIGHT AFTER calling draft_template_submission to verify that the draft file
    was actually created, and to show the letter content to the student.
    Compare the returned "chars" with the "chars" reported by draft_template_submission:
    if they are equal, this is the draft that was just written.
    A draft does NOT mean the request has been submitted.
    Returns JSON with path, chars, content. 
    On failure returns a string starting with "Error:" (for example when no draft exists for this student yet).

    Args:
        student_id: the 8-digit student ID used when drafting the letter, e.g. "24110001".
    """
    logger.info(f"Executing 'read_draft' for student ID {student_id}")
    try:
        student_id = student_id.strip()
        if not STUDENT_ID_PATTERN.fullmatch(student_id):
            return "Error: Student ID must contain exactly 8 digits (e.g. '24110xxx')."

        path = DRAFTS_DIR / f"draft_submission_{student_id}.txt"
        if not path.exists():
            return f"Error: No draft found for Student ID {student_id}. Please call 'draft_template_submission' first."

        content = path.read_text(encoding="utf-8")
        logger.info(f"Draft read successfully: {path} ({len(content)} characters)")
        return json.dumps(
            {"path": str(path), "chars": len(content), "content": content},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("Failed to read draft file.")
        return f"Error: Failed to read draft: {e}"

@mcp.tool()
def export_lesson_plan(
    week: int,
    title: str,
    duration_minutes: int,
    objectives: list[str],
    timeline: list[TimelineItem],
    homework: str
) -> str:
    """Export and SAVE a completed lesson plan to disk (WRITE / SAVE to disk).

    Use when a lesson plan has been fully generated, structured, and is ready to be saved permanently for the instructor. Valid course weeks are 1 through 8.
    Do NOT call this tool with incomplete or empty lesson plan sections, or while still brainstorming without the user's explicit consent.
    Required information: week (1 to 8), non-empty title, duration_minutes, objectives list, timeline list (each item having 'time', 'activity', and 'detail'), and homework string.
    If any information or section is missing, ASK the user to provide or approve the missing details before exporting. NEVER invent details.
    After a successful call, inform the user of the saved file paths (data/plans/) and present a brief summary of the exported plan.
    On failure returns a string starting with "Error:".
    
    Args:
        week: The curriculum week number (must be an integer from 1 to 8).
        title: Title of the lesson plan (cannot be empty).
        duration_minutes: Total session duration in minutes (e.g. 90, 180).
        objectives: List of learning objective strings (e.g. ["Understand CNN architecture", "Train ResNet"]).
        timeline: List of timeline items, each containing 'time', 'activity', and 'detail'.
        homework: Description of homework assignments, readings, or preparation for the next session.
    """
    logger.info(f"Executing 'export_lesson_plan' for week={week}, title='{title}'")
    try:
        if week < 1 or week > 6:
            logger.warning(f"Invalid week requested: {week}")
            return f"Error: Week {week} is invalid. The course covers weeks 1 to 6."
    
        title = title.strip()
        if not title:
            logger.warning("Empty lesson plan title provided.")
            return "Error: Lesson plan title cannot be empty."

        PLANS_DIR.mkdir(parents=True, exist_ok=True)

        plan_data = {
            "week": week,
            "title": title,
            "duration_minutes": duration_minutes,
            "created_at": datetime.now().isoformat(),
            "objectives": objectives,
            "timeline": timeline,
            "homework": homework.strip() if homework else ""
        }
        json_path = PLANS_DIR / f"lesson_plan_week_{week}.json"
        json_path.write_text(json.dumps(plan_data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved lesson plan JSON to {json_path}")

        md_lines = [
            f"# LESSON PLAN: WEEK {week} - {title.upper()}",
            f"- **Duration:** {duration_minutes} minutes",
            f"- **Created At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. Learning Objectives",
        ]
        for obj in objectives:
            md_lines.append(f"- {obj}")

        md_lines.append("\n## 2. Session Timeline")
        for item in timeline:
            t = item.get("time", "")
            act = item.get("activity", "")
            detail = item.get("detail", "")
            md_lines.append(f"- **[{t}] {act}:** {detail}")

        md_lines.append("\n## 3. Homework & Independent Study")
        md_lines.append(homework.strip() if homework.strip() else "No homework assigned.")

        md_path = PLANS_DIR / f"lesson_plan_week_{week}.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        logger.info(f"Saved lesson plan Markdown to {md_path}")

        return json.dumps({
            "status": "success",
            "week": week,
            "title": title,
            "json_path": str(json_path),
            "md_path": str(md_path),
            "note": "Lesson plan successfully exported to disk."
        }, ensure_ascii=False)

    except Exception as e:
        logger.exception(f"Error occurred while exporting lesson plan for Week {week}: {e}")
        return f"Error: Failed to export lesson plan for Week {week}: {e}"

if __name__ == "__main__":
    logger.info("Starting LessonPlanningAssistantServer on stdio transport...")
    mcp.run(transport="stdio")