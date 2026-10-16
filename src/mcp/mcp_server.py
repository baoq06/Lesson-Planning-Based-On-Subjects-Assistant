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
from src.schemas import DraftResult, SearchHit, SearchResult, ReadDraftResult, ExportPlanResult, TimelineItem
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
MAX_EXPONENT = 100

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
        
        if op_type is ast.Pow:
            if abs(right) > MAX_EXPONENT:
                raise ValueError(
                     f"Exponent too large ({right}). Exponent must not exceed {MAX_EXPONENT}."
                )
            
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
    """Calculate exact math expressions (+, -, *, /, **). Use for grade weighting or percentages to avoid arithmetic errors. Exponent limited to 100.

    Args:
        expression: Math string to evaluate (e.g. '0.05 * 9 + 0.15 * 8.5').
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
    """Search AIPR lecture notes (Weeks 1-6) for course concepts, code, and techniques. Answer ONLY from returned passages and ALWAYS cite [LP_W0{week}]. If found=false, refuse politely.

    Args:
        query: Search keywords or student question.
        mode: 'hybrid' (keywords + meaning) or 'dense' (meaning only). Default 'hybrid'.
        top_k: Number of passages (1-8, default 4).
        week: 0 for all weeks, or 1-6 to filter.
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
    """Get syllabus overview, topics, and learning objectives for a week (Weeks 1-6 only). Always cite [LP_W0{week}]. If week is not specified, ask the user first.

    Args:
        week: Week number (integer 1 to 6).
        full_content: False for structured summary; True for complete lecture notes.
    """
    logger.info(f"Executing 'get_curriculum_week' for week {week}, full_content={full_content}")
    if week < 1 or week > 6:
        logger.warning(f"Invalid week requested: {week}")
        return f"Error: Available lecture notes are currently only for Weeks 1 to 6 (requested week {week})."
    
    file_path = AIPR_DIR / f"w{week}_lecture_notes.md"
    if not file_path.exists():
        logger.error(f"Lecture note file not found: {file_path}")
        return f"Error: Lecture note file not found for Week {week} at {file_path.name}."

    doc_tag = f"[LP_W0{week}]"
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
            result = f"Source: {doc_tag}\n\n" + "\n\n---\n\n".join(summary_sections)
            if len(result) > 2500:
                result = result[:2500] + f"\n\n...(truncated. Use 'search_lesson_content' for specific sections) {doc_tag}."
            return result
        return f"Source: {doc_tag}\n\n" + content[:1500] + f"\n\n...(truncated. Use 'search_lesson_content' to query specific topics) {doc_tag}."
    
    except Exception as e:
        logger.error(f"Error occurred while reading curriculum for Week {week}: {e}")
        return f"Error: Failed to read curriculum file for Week {week}: {e}"

@mcp.tool()
def get_gotchas_and_best_practices(week: int) -> str:
    """Extract common student mistakes, pitfalls, and best practices for Week 1-6. Always cite [LP_W0{week}]. If week is not specified, ask the user first.

    Args:
        week: Week number (integer 1 to 6).
    """
    logger.info(f"Executing 'get_gotchas_and_best_practices' for week {week}")
    if week < 1 or week > 6:
        logger.warning(f"Invalid week requested: {week}")
        return f"Error: Available lecture notes are currently only for Weeks 1 to 6 (requested week {week})."
    
    file_path = AIPR_DIR / f"w{week}_lecture_notes.md"
    if not file_path.exists():
        logger.error(f"Lecture note file not found: {file_path}")
        return f"Error: Lecture note file not found for Week {week} at {file_path.name}."

    doc_tag = f"[LP_W0{week}]"
    try: 
        content = file_path.read_text(encoding="utf-8")
        pattern = r"(## Common Student Mistakes & Interventions.*?)(?=## KC Answer Key|## \S|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

        if match:
            logger.info(f"Successfully extracted 'Common Student Mistakes & Interventions' for Week {week}.")
            return f"### Gotchas & Common Mistakes (Week {week}) {doc_tag}:\n\nSource: {doc_tag}\n" + match.group(1).strip()

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
    absence_rate: float,
) -> str:
    """Compute AIPR weighted grade [POLICY_AIPR]: Attendance (5%), Labs (15%), Test 1 (15%), Test 2 (15%), Final Project (50%).
    Rule: absence_rate > 20% bars student from final exam/defense. Policy does NOT specify passing threshold (no >=5.0 threshold).
    All component scores (0-10) and absence_rate (0-100) are REQUIRED. If any is missing, ask the user first.

    Args:
        attendance: Attendance score (0-10).
        labs: Practical labs score (0-10).
        test1: Midterm Test 1 score (0-10).
        test2: Test 2 score (0-10).
        final_project: Final Project score (0-10).
        absence_rate: Absence percentage (0.0-100.0). No default.
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
                f" Attendance warning [POLICY_AIPR]:\n"
                f"- Absence rate: {absence_rate:.1f}% (exceeds the limit 20%).\n"
                f"- Current status: Banned from taking the final test / protecting the final project based on section 2 [POLICY_AIPR].\n"
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
            f"COURSE EVALUATION RESULT [POLICY_AIPR]:\n"
            f"- Attendance (5%): {attendance}\n"
            f"- Practical Labs (15%): {labs}\n"
            f"- Test 1 - Midterm (15%): {test1}\n"
            f"- Test 2 (15%): {test2}\n"
            f"- Final Project & Oral Defense (50%): {final_project}\n"
            f"- Absence Rate: {absence_rate:.1f}% (Compliant)\n"
            f"----------------------------------------\n"
            f"FINAL GRADE: {final_grade} / 10 [POLICY_AIPR]\n"
            #f"Status: {'PASSED' if passed else 'FAILED'}"
        )
    except Exception as e:
        logger.error(f"Error: Failed to calculate course grade: {e}")
        return f"Error: Failed to calculate course grade: {e}"
    
@mcp.tool()
def check_late_submission_penalty(hours_late: int) -> str:
    """Check assignment late penalty per Section 4 [POLICY_AIPR]: <=24h (-10%), 25-48h (-20%), 49-72h (-30%), >72h (rejected / 0 pts). If hours_late is unknown, ask user first.

    Args:
        hours_late: Number of hours submitted past deadline (integer >= 0).
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
                f"SUBMISSION REJECTED: Over 72 hours late ({hours_late} hours late) [POLICY_AIPR].\n"
                f"- Policy: AIPR Course Policy Section 4 [POLICY_AIPR].\n"
                f"- Status: Submission portal is closed.\n"
                f"- Grade: 0 points awarded for this assignment [POLICY_AIPR]."
            )

        logger.info(f"Late penalty applied: -{penalty}% ({desc})")
        return (
            f"LATE SUBMISSION PENALTY APPLIED [POLICY_AIPR]:\n"
            f"- Late Duration: {hours_late} hours ({desc}).\n"
            f"- Policy: AIPR Course Policy Section 4 [POLICY_AIPR].\n"
            f"- Penalty: DEDUCT {penalty}% of the assignment's total score [POLICY_AIPR]."
        )
    
    except Exception as e:
        logger.error(f"Error occurred while checking late submission penalty: {e}")
        return f"Error: Failed to check late submission penalty: {e}"

@mcp.tool()
def draft_template_submission(student_id: str, student_name: str, assignment: str, days_requested: int, reason: str, has_medical_proof: bool) -> str:
    """Draft and save a medical deadline-extension request (.txt file). Requires medical proof (mandatory).
    IMPORTANT: After calling this tool, MUST immediately call 'read_draft' with student_id to verify the file before responding.

    Args:
        student_id: Exactly 8 digits (e.g. '24110001').
        student_name: Full student name.
        assignment: Assignment name (e.g. 'Lab 1').
        days_requested: Number of days (1-5).
        reason: Medical reason description.
        has_medical_proof: True if student confirms having valid medical proof; False otherwise.
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
    """Read back a saved extension-request draft file to verify content (Write-then-Read pattern).

    Args:
        student_id: The 8-digit student ID (e.g. '24110001').
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
        result = ReadDraftResult(
            path=str(path),
            chars=len(content),
            content=content
        )
        return json.dumps(asdict(result), ensure_ascii=False)
    except Exception as e:
        logger.exception("Failed to read draft file.")
        return f"Error: Failed to read draft: {e}"

if __name__ == "__main__":
    logger.info("Starting LessonPlanningAssistantServer on stdio transport...")
    mcp.run(transport="stdio")  