import os
import sys
import json
import logging
import re
from datetime import datetime, date
from dataclasses import asdict
from typing import Optional
from pathlib import Path
from mcp.server.mcpserver import MCPServer
from string import Template
from src.schemas import DraftResult

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
TEMPLATE_PATH = REPO_ROOT / "data" / "templates" / "extension_request.txt"
DRAFTS_DIR = REPO_ROOT / "data" / "drafts"
PLANS_DIR = REPO_ROOT / "data" / "plans"
MAX_EXTENSION_DAYS = 5
STUDENT_ID_PATTERN = re.compile(r"^\d{8}$")

MEMORY_PATH = REPO_ROOT / "data" / "mcp_memory.json"
DATA_DIR = REPO_ROOT / "data"
AIPR_DIR = DATA_DIR / "lecture-notes"

logging.basicConfig(
    stream=sys.stderr, 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("mcp_server")

mcp = MCPServer("LessonPlanningAssistantServer")

@mcp.tool()
def remember(key: str, content: str) -> str:
    logger.info(f"Executing 'remember' tool for key: '{key}'")
    try:
        if not key.strip():
            return "Error: A non-empty key is required."
        
        memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8")) if MEMORY_PATH.exists() else {}
        memory[key] = content

        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_PATH.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(f"Successfully stored key '{key}' in memory.")
        return f"Stored key '{key}' in memory."
    except Exception as e:
        logger.error(f"Error occurred while storing key '{key}': {e}")
        return f"Error occurred while storing key '{key}': {e}"

@mcp.tool()
def recall(query: str, top_k: int = 3) -> str:
    logger.info(f"Executing 'recall' tool with query: '{query}', top_k={top_k}")
    try:
        top_k = max(1, min(top_k, 8))
        if not MEMORY_PATH.exists():
            return "Nothing remembered matches that query."

        memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        hits = [f"{k}: {v}" for k, v in memory.items() if query.lower() in (k + v).lower()]
        
        logger.info(f"Found {len(hits)} matching items in memory.")
        if not hits:
            return "Nothing remembered matches that query."
        return "\n".join(hits[:top_k])
    except Exception as e:
        logger.error(f"Error occurred while recalling query '{query}': {e}")
        return f"Error occurred while recalling query '{query}': {e}"

@mcp.tool()
def calculator(expression: str) -> str:
    logger.info(f"Executing 'calculator' tool with expression: '{expression}'")
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        logger.error(f"Error occurred while evaluating expression '{expression}': {e}")
        return f"Error occurred while evaluating expression '{expression}': {e}"

@mcp.tool()
def get_curriculum_week(week: int, full_content: bool = False) -> str:
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

        if full_content:
            return content

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
            return (
                "\n\n---\n\n".join(summary_sections)
                + "\n\n*(Note: This is a structured summary of Learning Objectives & Session Structures. Pass 'full_content=True' if you need the full instructor script and code alongs).*"
            )

        return content[:3000] + "\n...(content truncated)"
    except Exception as e:
        logger.error(f"Error occurred while reading curriculum for Week {week}: {e}")
        return f"Error occurred while reading curriculum file for Week {week}: {e}"

@mcp.tool()
def get_gotchas_and_best_practices(week: int) -> str:
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
        return f"Error occurred while reading lecture notes for Week {week}: {e}"

@mcp.tool()
def calculate_course_grade(
    attendance: float,
    labs: float,
    test1: float,
    test2: float,
    final_project: float,
    absence_rate: float = 0.0,
) -> str:
    logger.info(
        f"Executing 'calculate_course_grade' with attendance={attendance}, labs={labs}, "
        f"test1={test1}, test2={test2}, final_project={final_project}, absence_rate={absence_rate}%"
    )
    try:
        if absence_rate > 20.0:
            logger.info(f"Student failed attendance requirement: absence_rate={absence_rate}% > 20%")
            return (
                f"WARNING: Absence rate is {absence_rate:.1f}% (exceeds the 20% policy limit).\n"
                f"- Status: Student is DISQUALIFIED from the final exam/defense per Course Policy Section 2.\n"
                f"- Final Grade: 0.0 (Failed)."
            )

        final_grade = (
            (attendance * 0.05)
            + (labs * 0.15)
            + (test1 * 0.15)
            + (test2 * 0.15)
            + (final_project * 0.50)
        )
        final_grade = round(final_grade, 2)
        passed = final_grade >= 5.0

        logger.info(f"Calculated final grade: {final_grade}, passed={passed}")
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
            f"Status: {'PASSED' if passed else 'FAILED'}"
        )
    except Exception as e:
        logger.error(f"Error occurred while calculating course grade: {e}")
        return f"Error occurred while calculating course grade: {e}"

@mcp.tool()
def check_late_submission_penalty(hours_late: int) -> str:
    logger.info(f"Executing 'check_late_submission_penalty' for hours_late={hours_late}")
    try:
        if hours_late <= 0:
            return "✅ Submission is on time. No penalty applied (Penalty: 0%)."
        
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
        return f"Error occurred while checking late submission penalty: {e}"

@mcp.tool()
def draft_template_submission(student_id: str, student_name: str, assg: str, days_requested: int, reason: str, proof: bool) -> str:
    logger.info(f"Executing 'draft_template_submission' for student {student_id}")
    try:
        student_id, student_name = student_id.strip(), student_name.strip()
        assignment, reason = assignment.strip(), reason.strip()

        if not (student_id and student_name and assg and reason):
            logger.warning("Missing required arguments in extension draft submission.")
            return "Error: Student ID, Student Name, Assignment name, and Reason are required."
        
        if not STUDENT_ID_PATTERN.fullmatch(student_id):
            logger.warning(f"Invalid Student ID format: '{student_id}'")
            return f"Error: Student ID '{student_id}' is invalid. It must contain exactly 8 digits."

        if not proof:
            logger.warning("Extension request rejected due to missing medical proof.")
            return "Error: A valid medical certificate is strictly required per Course Policy Section 4."
        
        if not 1 <= days_requested <= MAX_EXTENSION_DAYS:
            logger.warning(f"Requested days out of range: {days_requested}")
            return f"Error: Requested days must be between 1 and {MAX_EXTENSION_DAYS} days."

        if not TEMPLATE_PATH.exists():
            logger.error(f"Template file not found at: {TEMPLATE_PATH}")
            return "System Error: Extension request template file not found."
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
    timeline: list[dict],
    homework: str
) -> str:
    logger.info(f"Executing 'export_lesson_plan' for week={week}, title='{title}'")
    try:
        if week < 1 or week > 8:
            logger.warning(f"Invalid week requested: {week}")
            return f"Error: Week {week} is invalid. The course covers weeks 1 to 8."
    
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
        return f"Error occurred while exporting lesson plan for Week {week}: {e}"

if __name__ == "__main__":
    logger.info("Starting LessonPlanningAssistantServer on stdio transport...")
    mcp.run(transport="stdio")