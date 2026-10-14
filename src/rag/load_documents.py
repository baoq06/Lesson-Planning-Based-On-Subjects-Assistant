from pathlib import Path
import re
from src.schemas import Document

def load_all_course_documents(data_dir: Path | None = None) -> list[Document]:
    """
    Quét toàn bộ tài liệu bài giảng và quy chế trong thư mục data/ của dự án.
    - Bài giảng: data/lecture-notes/w*_lecture_notes.md
    - Quy chế: data/Policy/AIPR_Policy.md
    """
    if data_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        data_dir = repo_root / "data"

    docs: list[Document] = []
    
    lecture_dir = data_dir / "lecture-notes"
    if lecture_dir.exists():
        for file_path in sorted(lecture_dir.glob("w*.md")):
            match = re.search(r"w(\d+)", file_path.name, re.IGNORECASE)
            week_num = int(match.group(1)) if match else 0
            
            content = file_path.read_text(encoding="utf-8")
            docs.append(Document(
                content=content,
                doc_id=f"LP_W{week_num:02d}",        
                title=f"Bài giảng Tuần {week_num}",
                source=str(file_path.name),
                metadata={
                    "week": week_num, 
                    "type": "curriculum",
                    "source": str(file_path.name)
                }
            ))

    policy_file = data_dir / "Policy" / "AIPR_Policy.md"
    if policy_file.exists():
        docs.append(Document(
            content=policy_file.read_text(encoding="utf-8"),
            doc_id="POLICY_AIPR",
            title="Quy chế môn học AIPR",
            source=str(policy_file.name),
            metadata={
                "type": "policy",
                "source": str(policy_file.name)
            }
        ))

    return docs
