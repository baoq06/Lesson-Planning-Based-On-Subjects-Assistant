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
TEMPLATE_PATH = REPO_ROOT / "data" / "prompts" / "extension_request.txt"
DRAFTS_DIR = REPO_ROOT / "data" / "drafts"
PLANS_DIR = REPO_ROOT / "plans"
MAX_EXTENSION_DAYS = 5
STUDENT_ID_PATTERN = re.compile(r"\d{8}")

MEMORY_PATH = REPO_ROOT / "data" / "mcp_memory.json"
DATA_DIR = REPO_ROOT / "data"
AIPR_DIR = DATA_DIR / "AI_Programming_with_Python"

# setup logger để in ra stderr (in ra vào file log để kiểm tra lỗi)
logging.basicConfig(
    stream=sys.stderr, 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("mcp_server")

mcp = MCPServer("LessonPlanningAssistantServer")

@mcp.tool()
def remember(key: str, content: str) -> str:
    #Thay đổi file (MEMORY_PATH.write_text): tạo mới hoặc ghi đè cặp key-value.
    logger.info(f"Executing 'remember' for key: '{key}'")
    try:
        if not key.strip():
            return "A non-empty key is required."
        
        memory = json.loads(MEMORY_PATH.read_text()) if MEMORY_PATH.exists() else {}
        memory[key] = content

        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_PATH.write_text(json.dumps(memory, indent=2))

        logger.info(f"Successfully remembered key '{key}'")
        return f"Stored key '{key}' in memory." 
    except Exception as e:
        logger.error(f"Error occurred while remembering key '{key}': {e}")
        return f"Error occurred while remembering key '{key}': {e}"

@mcp.tool()
def recall(query: str, top_k: int = 3) -> str:
    #Chỉ đọc (Read-only): không làm thay đổi nội dung file.
    logger.info(f"Executing 'recall' for query: '{query}' with top_k={top_k}")
    try:
        top_k = max(1, min(top_k, 8))
        memory = json.loads(MEMORY_PATH.read_text()) if MEMORY_PATH.exists() else {}
        hits = [f"{k}: {v}" for k, v in memory.items() if query.lower() in (k + v).lower()]
        return "\n".join(hits[:top_k]) or "Nothing remembered matches that query."

        logger.info(f"Recall found {len(hits)} matching items.")
        return "\n".join(hits[:top_k]) or "Nothing remembered matches that query."
    except Exception as e:
        logger.error(f"Error occurred while recalling query '{query}': {e}")
        return f"Error occurred while recalling query '{query}': {e}"

@mcp.tool()
def calculator(expression: str) -> str:
    logger.info(f"Executing 'calculator' for expression: '{expression}'")
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        logger.error(f"Error occurred while evaluating expression '{expression}': {e}")
        return f"Error occurred while evaluating expression '{expression}': {e}"

@mcp.tool()
def get_curriculum_week(week: int) -> str:
    logger.info(f"Executing 'get_curriculum_week' for week: {week}")
    if week < 1 or week > 8:
        logger.warning(f"Invalid week requested: {week}")
        return f"Error: Week {week} is invalid. The course only includes weeks 1 to 8."
    
    file_path = AIPR_DIR / f"AIPR_w{week}.md"
    if not file_path.exists():
        logger.error(f"File not found for week {week}: {file_path}")
        return f"Error: File not found for week {week} at {file_path.name}."

    try: 
        content = file_path.read_text(encoding="utf-8")
        logger.info(f"Successfully read curriculum for week {week}")
        return content
    except Exception as e:
        logger.error(f"Error occurred while reading curriculum file for week {week}: {e}")
        return f"Error occurred while reading curriculum file for week {week}: {e}"

@mcp.tool()
def get_gotchas_and_best_practices(week: int) -> str:
    logger.info(f"Executing 'get_gotchas_and_best_practices' for week: {week}")
    if week < 1 or week > 8:
        logger.warning(f"Invalid week requested: {week}")
        return f"Error: Week {week} is invalid. The course only includes weeks 1 to 8."
    
    file_path = AIPR_DIR / f"AIPR_w{week}.md"
    if not file_path.exists():
        logger.error(f"File not found for week {week}: {file_path}")
        return f"Error: File not found for week {week} at {file_path.name}."

    try: 
        content = file_path.read_text(encoding="utf-8")
        pattern = r"(## 4\.\s*Các bẫy thường gặp.*?)(?=## 5\.|\Z)"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            logger.info(f"Successfully extracted 'Common Pitfalls' section for week {week}")
            return f"### Gotchas & Best Practices (Week {week}):\n\n" + match.group(1).strip()

        logger.warning(f"Gotchas section not found in the document for week {week}")
        return f"Error: Gotchas section not found in the document for week {week}."
    except Exception as e:
        logger.error(f"Error occurred while reading curriculum file for week {week}: {e}")
        return f"Error occurred while reading curriculum file for week {week}: {e}"

@mcp.tool()
def calculate_course_grade(
    attendance: float,
    labs: float,
    test1: float,
    test2: float,
    final_project: float,
    absence_rate: float = 0.0,
) -> str:
    logger.info(f"Executing 'calculate_course_grade' with attendance={attendance}, labs={labs}, test1={test1}, test2={test2}, final_project={final_project}, absence_rate={absence_rate}")
    try:
        if absence_rate > 20.0:
            logger.info(f"Student failed attendance: absence_rate={absence_rate} exceeds 20%")
            return (
                f"❌ CẢNH BÁO: Tỷ lệ vắng học là {absence_rate:.1f}% (vượt quá mức cho phép 20%).\n"
                f"- Trạng thái: Sinh viên BỊ CẤM THI / Không đủ điều kiện bảo vệ đồ án cuối kỳ theo Quy chế mục 2.\n"
                f"- Điểm tổng kết: 0.0 (Không đạt)."
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
            f"📊 KẾT QUẢ ĐÁNH GIÁ MÔN HỌC (AIPR):\n"
            f"- Chuyên cần (5%): {attendance}\n"
            f"- Bài tập Lab (15%): {labs}\n"
            f"- Kiểm tra 1 - Giữa kỳ (15%): {test1}\n"
            f"- Kiểm tra 2 (15%): {test2}\n"
            f"- Đồ án & Vấn đáp (50%): {final_project}\n"
            f"- Tỷ lệ vắng mặt: {absence_rate:.1f}% (Hợp lệ)\n"
            f"----------------------------------------\n"
            f"🎯 ĐIỂM TỔNG KẾT: {final_grade} / 10\n"
            f"📌 Trạng thái: {'ĐẠT (Passed)' if passed else 'KHÔNG ĐẠT (Failed)'}"
        )
    except Exception as e:
        logger.error(f"Error occurred while calculating course grade: {e}")
        return f"Error occurred while calculating course grade: {e}"

@mcp.tool()
def check_late_submission_penalty(hours_late: int) -> str:
    logger.info(f"Executing 'check_late_submission_penalty' for hours_late={hours_late}")
    try:
        if hours_late <= 0:
            return "✅ Bài nộp đúng hạn. Không bị trừ điểm (Mức phạt: 0%)."
        
        if hours_late <= 24:
            penalty = 10
            desc = "Trễ từ 1 đến 24 giờ (Ngày thứ 1)"
        elif hours_late <= 48:
            penalty = 20
            desc = "Trễ từ 25 đến 48 giờ (Ngày thứ 2)"
        elif hours_late <= 72:
            penalty = 30
            desc = "Trễ từ 49 đến 72 giờ (Ngày thứ 3)"
        else:
            logger.info(f"Late submisssion rejected (>72h): {hours_late} hours")
            return (
                f"❌ NỘP TRỄ QUÁ 72 GIỜ (Cụ thể: {hours_late} giờ trễ).\n"
                f"- Căn cứ: Mục 4 Quy chế môn học AIPR.\n"
                f"- Trạng thái: Cổng nộp bài đã đóng.\n"
                f"- Mức điểm: NHẬN 0 ĐIỂM cho bài tập này."
            )
        logger.info(f"Late penalty applied: -{penalty}% ({desc})")
        return (
            f"⚠️ CẢNH BÁO NỘP BÀI TRỄ HẠN:\n"
            f"- Thời gian trễ: {hours_late} giờ ({desc}).\n"
            f"- Căn cứ: Mục 4 Quy chế môn học AIPR.\n"
            f"- Mức phạt: TRỪ {penalty}% tổng số điểm của bài nộp."
        )
    except Exception as e:
        logger.error(f"Error occurred while checking late submission penalty: {e}")
        return f"Error occurred while checking late submission penalty: {e}"

@mcp.tool()
def draft_template_submission(student_id: str, student_name: str, assg: str, days_requested: int, reason: str, proof: bool) -> str:
    logger.info(f"Executing draft_template_submission for student {student_id}")
    try:
        student_id, student_name = student_id.strip(), student_name.strip()
        assg, reason = assg.strip(), reason.strip()

        if not (student_id and student_name and assg and reason):
            logger.info(f"[Error] student_id, student_name, assg and reason are required.")
            return "Error: student_id, student_name, assg and reason are required."
        if not proof:
            logger.info(f"[Error] a valid medical certificate is required (policy section 4).")
            return "Error: a valid medical certificate is required (policy section 4)."
        if not 1 <= days_requested <= MAX_EXTENSION_DAYS:
            logger.info(f"[Error] days_requested must be between 1 and {MAX_EXTENSION_DAYS}.")
            return f"Error: days_requested must be between 1 and {MAX_EXTENSION_DAYS}."

        # Điền tham số vào trong mẫu txt
        content = Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(
            student_name=student_name,
            student_id=student_id,
            assignment=assg,
            days_requested=days_requested,
            reason=reason,
            today=date.today().strftime("%Y-%m-%d"),
        )

        # Ghi file
        DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
        path = DRAFTS_DIR / f"draft_submission_{student_id}.txt"
        overwritten = path.exists()
        path.write_text(content, encoding="utf-8")
        logger.info("draft saved: %s (overwritten=%s)", path, overwritten)

        return json.dumps(
            asdict(DraftResult(
                path=str(path),
                overwritten=overwritten,
                chars=len(content),
                note="This is a draft and has NOT been submitted. Call read_draft with this path to verify.",
            )),
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("draft_template_submission failed")
        return f"Error: draft_template_submission failed: {e}"

@mcp.tool()
def export_lesson_plan(
    week: int,
    title: str,
    duration_minutes: int,
    objectives: list[str],
    timeline: list[dict],
    homework: str
) -> str:
    logger.info(f"Executing 'export_lesson_plan' for week={week}, title={title}")
    try:
        if week < 1 or week > 8:
            logger.warning(f"Invalid week provided: {week}")
            return f"Error: Week {week} is invalid. The course only includes weeks 1 to 8."
    
        title = title.strip()
        if not title:
            logger.warning("Empty title provided for lesson plan.")
            return "Error: Lesson plan title cannot be empty."

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

        ]
        for obj in objectives:
            md_lines.append(f"- {obj}")

        md_lines.append()

    except Exception as e:
        logger.exception(f"Error occurred while exporting lesson plan for week {week}: {e}")
        return f"Error: Failed to export lesson plan for week {week}: {e}"

if __name__ == "__main__":
    mcp.run(transport="stdio")