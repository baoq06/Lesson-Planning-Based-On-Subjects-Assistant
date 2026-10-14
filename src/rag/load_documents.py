from pathlib import Path
import re
from src.schemas import Document

# Khối metadata ở đầu file, kẹp giữa hai dòng '---'
FRONT_MATTER = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Tách khối metadata '---' ở đầu file. Trả về (metadata, nội dung còn lại)."""
    match = FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, text[match.end():]


def load_all_course_documents(data_dir: Path | None = None) -> list[Document]:
    """
    Nạp tài liệu môn học trong data/:
    - Bài giảng (bản cho sinh viên): data/lecture-notes/w*_lecture_notes.md
    - Quy chế: data/Policy/AIPR_Policy.md
    Metadata (doc_id, title, version, date, owner) lấy từ khối '---' đầu mỗi file.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data"

    docs: list[Document] = []

    lecture_dir = data_dir / "lecture-notes"
    if lecture_dir.exists():
        for file_path in sorted(lecture_dir.glob("w*.md")):
            match = re.search(r"w(\d+)", file_path.name, re.IGNORECASE)
            week_num = int(match.group(1)) if match else 0

            meta, body = parse_front_matter(file_path.read_text(encoding="utf-8"))
            docs.append(Document(
                content=body,
                doc_id=meta.get("doc_id", f"LP_W{week_num:02d}"),
                title=meta.get("title", f"Week {week_num}"),
                source=file_path.name,
                metadata={
                    "week": week_num,
                    "type": "curriculum",
                    "version": meta.get("version", ""),
                    "date": meta.get("date", ""),
                    "owner": meta.get("owner", ""),
                    "source": file_path.name,
                },
            ))

    policy_file = data_dir / "Policy" / "AIPR_Policy.md"
    if policy_file.exists():
        meta, body = parse_front_matter(policy_file.read_text(encoding="utf-8"))
        docs.append(Document(
            content=body,
            doc_id=meta.get("doc_id", "POLICY_AIPR"),
            title=meta.get("title", "Quy chế môn học AIPR"),
            source=policy_file.name,
            metadata={
                "type": "policy",
                "version": meta.get("version", ""),
                "date": meta.get("date", ""),
                "owner": meta.get("owner", ""),
                "source": policy_file.name,
            },
        ))

    return docs
