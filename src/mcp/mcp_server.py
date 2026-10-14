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
STUDENT_ID_PATTERN = re.compile(r"\d{8}")

MEMORY_PATH = REPO_ROOT / "data" / "mcp_memory.json"
DATA_DIR = REPO_ROOT / "data"
AIPR_DIR = DATA_DIR / "lecture-notes"

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
    logger.info(f"Chạy tool 'remember' với key: '{key}'")
    try:
        if not key.strip():
            return "Phải có một key không rỗng."
        
        memory = json.loads(MEMORY_PATH.read_text()) if MEMORY_PATH.exists() else {}
        memory[key] = content

        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_PATH.write_text(json.dumps(memory, indent=2))

        logger.info(f"Nhớ key thành công: '{key}'")
        return f"Lưu key '{key}' vào memory." 
    except Exception as e:
        logger.error(f"Có lỗi xảy ra khi nhớ key '{key}': {e}")
        return f"Có lỗi xảy ra khi nhớ key '{key}': {e}"

@mcp.tool()
def recall(query: str, top_k: int = 3) -> str:
    #Chỉ đọc (Read-only): không làm thay đổi nội dung file.
    logger.info(f"Chạy tool 'recall' với query: '{query}' và top_k={top_k}")
    try:
        top_k = max(1, min(top_k, 8))
        memory = json.loads(MEMORY_PATH.read_text()) if MEMORY_PATH.exists() else {}
        hits = [f"{k}: {v}" for k, v in memory.items() if query.lower() in (k + v).lower()]
        
        logger.info(f"Đã tìm thấy {len(hits)} mục phù hợp trong bộ nhớ.")
        return "\n".join(hits[:top_k]) or "Không tìm thấy thông tin nào trong bộ nhớ khớp với truy vấn."
    except Exception as e:
        logger.error(f"Có lỗi xảy ra khi truy xuất query '{query}': {e}")
        return f"Có lỗi xảy ra khi truy xuất query '{query}': {e}"

@mcp.tool()
def calculator(expression: str) -> str:
    logger.info(f"Chạy tool 'calculator' với biểu thức: '{expression}'")
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        logger.error(f"Có lỗi xảy ra với biểu thức '{expression}': {e}")
        return f"Có lỗi xảy ra với biểu thức '{expression}': {e}"

@mcp.tool()
def get_curriculum_week(week: int) -> str:
    logger.info(f"Chạy tool 'get_curriculum_week' với tuần: {week}")
    if week < 1 or week > 8:
        logger.warning(f"Tuần không hợp lệ: {week}")
        return f"Lỗi: Tuần {week} không hợp lệ. Khóa học chỉ có từ tuần 1 đến 8."
    
    file_path = AIPR_DIR / f"AIPR_w{week}.md"
    if not file_path.exists():
        logger.error(f"Không tìm thấy file cho tuần {week}: {file_path}")
        return f"Lỗi: Không tìm thấy file cho tuần {week} ở {file_path.name}."

    try: 
        content = file_path.read_text(encoding="utf-8")
        logger.info(f"Thành công đọc chương trình học của tuần {week}")
        return content
    except Exception as e:
        logger.error(f"Có lỗi xảy ra khi đọc file chương trình học của tuần {week}: {e}")
        return f"Có lỗi xảy ra khi đọc file chương trình học của tuần {week}: {e}"

@mcp.tool()
def get_gotchas_and_best_practices(week: int) -> str:
    logger.info(f"Chạy tool 'get_gotchas_and_best_practices' cho tuần: {week}")
    if week < 1 or week > 8:
        logger.warning(f"Tuần không hợp lệ: {week}")
        return f"Lỗi: Tuần {week} không hợp lệ. Khóa học chỉ có từ tuần 1 đến 8."
    
    file_path = AIPR_DIR / f"AIPR_w{week}.md"
    if not file_path.exists():
        logger.error(f"Không tìm thấy file cho tuần {week}: {file_path}")
        return f"Lỗi: Không tìm thấy file cho tuần {week} ở {file_path.name}."

    try: 
        content = file_path.read_text(encoding="utf-8")
        pattern = r"(## 4\.\s*Các bẫy thường gặp.*?)(?=## 5\.|\Z)"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            logger.info(f"Trích xuất thành công mục 'Common Pitfalls' cho tuần {week}")
            return f"### Gotchas & Best Practices (Week {week}):\n\n" + match.group(1).strip()

        logger.warning(f"Không tìm thấy mục Gotchas trong tài liệu của tuần {week}")
        return f"Lỗi: Không tìm thấy mục Gotchas trong tài liệu của tuần {week}."
    except Exception as e:
        logger.error(f"Có lỗi xảy ra khi đọc file chương trình học của tuần {week}: {e}")
        return f"Có lỗi xảy ra khi đọc file chương trình học của tuần {week}: {e}"

@mcp.tool()
def calculate_course_grade(
    attendance: float,
    labs: float,
    test1: float,
    test2: float,
    final_project: float,
    absence_rate: float = 0.0,
) -> str:
    logger.info(f"Chạy tool 'calculate_course_grade' với Chuyên cần={attendance}, Bài tập lab={labs}, Kiểm tra 1={test1}, Kiểm tra 2={test2}, Đồ án & Vấn đáp={final_project}, Tỷ lệ vắng mặt={absence_rate}")
    try:
        if absence_rate > 20.0:
            logger.info(f"Điểm chuyên cần của sinh viên không đạt: Tỷ lệ vắng mặt={absence_rate} quá 20%")
            return (
                f"CẢNH BÁO: Tỷ lệ vắng học là {absence_rate:.1f}% (vượt quá mức cho phép 20%).\n"
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

        logger.info(f"Điểm tổng kết: {final_grade}, Trạng thái={passed}")
        return (
            f"KẾT QUẢ ĐÁNH GIÁ MÔN HỌC (AIPR):\n"
            f"- Chuyên cần (5%): {attendance}\n"
            f"- Bài tập Lab (15%): {labs}\n"
            f"- Kiểm tra 1 - Giữa kỳ (15%): {test1}\n"
            f"- Kiểm tra 2 (15%): {test2}\n"
            f"- Đồ án & Vấn đáp (50%): {final_project}\n"
            f"- Tỷ lệ vắng mặt: {absence_rate:.1f}% (Hợp lệ)\n"
            f"----------------------------------------\n"
            f"ĐIỂM TỔNG KẾT: {final_grade} / 10\n"
            f"Trạng thái: {'ĐẠT (Passed)' if passed else 'KHÔNG ĐẠT (Failed)'}"
        )
    except Exception as e:
        logger.error(f"Có lỗi xảy ra khi tính điểm: {e}")
        return f"Có lỗi xảy ra khi tính điểm: {e}"

@mcp.tool()
def check_late_submission_penalty(hours_late: int) -> str:
    logger.info(f"Chạy tool 'check_late_submission_penalty' với số giờ trễ={hours_late}")
    try:
        if hours_late <= 0:
            return "Bài nộp đúng hạn. Không bị trừ điểm (Mức phạt: 0%)."
        
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
            logger.info(f"Bài nộp trễ bị từ chối (>72h): {hours_late} giờ")
            return (
                f"NỘP TRỄ QUÁ 72 GIỜ (Cụ thể: {hours_late} giờ trễ).\n"
                f"- Căn cứ: Mục 4 Quy chế môn học AIPR.\n"
                f"- Trạng thái: Cổng nộp bài đã đóng.\n"
                f"- Mức điểm: NHẬN 0 ĐIỂM cho bài tập này."
            )
        logger.info(f"Mức phạt nộp trễ được áp dụng: -{penalty}% ({desc})")
        return (
            f"CẢNH BÁO NỘP BÀI TRỄ HẠN:\n"
            f"- Thời gian trễ: {hours_late} giờ ({desc}).\n"
            f"- Căn cứ: Mục 4 Quy chế môn học AIPR.\n"
            f"- Mức phạt: TRỪ {penalty}% tổng số điểm của bài nộp."
        )
    except Exception as e:
        logger.error(f"Có lỗi xảy ra khi kiểm tra mức phạt nộp trễ: {e}")
        return f"Có lỗi xảy ra khi kiểm tra mức phạt nộp trễ: {e}"

@mcp.tool()
def draft_template_submission(student_id: str, student_name: str, assignment: str, days_requested: int, reason: str, has_medical_proof: bool) -> str:
    logger.info(f"Chạy tool 'draft_template_submission' với sinh viên {student_id}")
    try:
        student_id, student_name = student_id.strip(), student_name.strip()
        assignment, reason = assignment.strip(), reason.strip()

        if not (student_id and student_name and assignment and reason):
            logger.info(f"Mã số sinh viên, Tên sinh viên, tên bài tập và lý do không được bỏ trống.")
            return "Error: Mã số sinh viên, Tên sinh viên, tên bài tập và lý do không được bỏ trống."
        if not STUDENT_ID_PATTERN.fullmatch(student_id):
            logger.info(f"Mã số sinh viên phải đúng 8 chữ số, ví dụ: '24110xxx'.")
            return "Error: Mã số sinh viên phải có đúng 8 chữ số, ví dụ: '24110xxx'."
        if not has_medical_proof:
            logger.info(f"Chứng nhận y tế không được bỏ trống (policy mục 4).")
            return "Error: Chứng nhận y tế không được bỏ trống (policy mục 4)."
        if not 1 <= days_requested <= MAX_EXTENSION_DAYS:
            logger.info(f"Ngày yêu cầu phải từ 1 đến {MAX_EXTENSION_DAYS}.")
            return f"Error: Ngày yêu cầu phải từ 1 đến {MAX_EXTENSION_DAYS}."

        # Điền tham số vào trong mẫu txt
        content = Template(TEMPLATE_PATH.read_text(encoding="utf-8")).substitute(
            student_name=student_name,
            student_id=student_id,
            assignment=assignment,
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
                note="Đây là bản nháp, CHƯA được nộp. Gọi tool 'read_draft' với student_id này để xác thực.",
            )),
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("Nộp bản nháp không thành công")
        return f"Error: Nộp bản nháp không thành công: {e}"

@mcp.tool()
def read_draft(student_id: str) -> str:
    logger.info(f"Chạy tool 'read_draft' với sinh viên {student_id}")
    try:
        student_id = student_id.strip()
        if not STUDENT_ID_PATTERN.fullmatch(student_id):
            logger.info(f"Mã số sinh viên phải có đúng 8 chữ số, ví dụ: '24110xxx'.")
            return "Lỗi: Mã số sinh viên phải có đúng 8 chữ số, ví dụ: '24110xxx'."

        path = DRAFTS_DIR / f"draft_submission_{student_id}.txt"
        if not path.exists():
            logger.info(f"Không tìm thấy bản nháp trùng khớp với mã số sinh viên {student_id}. Vui lòng gọi tool 'draft_template_submission trước.")
            return f"Lỗi: Không tìm thấy bản nháp trùng khớp với mã số sinh viên {student_id}. Vui lòng gọi tool 'draft_template_submission trước."

        content = path.read_text(encoding="utf-8")
        logger.info("draft read: %s (%d chars)", path, len(content))
        return json.dumps(
            {"path": str(path), "chars": len(content), "content": content},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("Đọc bản nháp không thành công")
        return f"Lỗi: Đọc bản nháp không thành công: {e}"

@mcp.tool()
def export_lesson_plan(
    week: int,
    title: str,
    duration_minutes: int,
    objectives: list[str],
    timeline: list[dict],
    homework: str
) -> str:
    logger.info(f"Chạy tool 'export_lesson_plan' với tuần={week}, tiêu đề={title}")
    try:
        if week < 1 or week > 8:
            logger.warning(f"Tuần không hợp lệ: {week}")
            return f"Lỗi: Tuần {week} không hợp lệ. Khóa học chỉ có từ tuần 1 đến 8."
    
        title = title.strip()
        if not title:
            logger.warning("Tiêu đề không được bỏ trống")
            return "Lỗi: Tiêu đề không được bỏ trống"

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
        logger.info(f"Lưu JSON kế hoạch bài học vào {json_path}")

        md_lines = [
            f"# GIÁO ÁN BÀI GIẢNG: TUẦN {week} - {title.upper()}",
            f"- **Thời lượng:** {duration_minutes} phút",
            f"- **Ngày khởi tạo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. Chuẩn đầu ra & Mục tiêu (Learning Objectives)",
        ]
        for obj in objectives:
            md_lines.append(f"- {obj}")

        md_lines.append("\n## 2. Tiến trình bài giảng (Timeline)")
        for item in timeline:
            t = item.get("time", "")
            act = item.get("activity", "")
            detail = item.get("detail", "")
            md_lines.append(f"- **[{t}] {act}:** {detail}")

        md_lines.append("\n## 3. Bài tập về nhà & Nhiệm vụ tự học (Homework)")
        md_lines.append(homework.strip() if homework.strip() else "Không có bài tập về nhà.")

        md_path = PLANS_DIR / f"lesson_plan_week_{week}.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        logger.info(f"Saved lesson plan Markdown to {md_path}")

        return json.dumps({
            "status": "success",
            "week": week,
            "title": title,
            "json_path": str(json_path),
            "md_path": str(md_path),
            "note": "Xuất kế hoạch bài học vào ổ đĩa thành công."
        }, ensure_ascii=False)

    except Exception as e:
        logger.exception(f"Có lỗi xảy ra khi xuất kế hoạch bài học cho tuần {week}: {e}")
        return f"Có lỗi xảy ra khi xuất kế hoạch bài học cho tuần {week}: {e}"

if __name__ == "__main__":
    mcp.run(transport="stdio")