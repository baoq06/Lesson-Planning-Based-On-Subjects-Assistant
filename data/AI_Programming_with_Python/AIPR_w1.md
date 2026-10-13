# Tuần 1: Cấu trúc dữ liệu và Typing trong Python (Data Layer)

## 1. Thông tin chung & Chuẩn đầu ra (Learning Objectives - A1.1 đến A1.4)
- **A1.1**: Lựa chọn kiểu dữ liệu phù hợp (`list`, `tuple`, `dict`, `set`) cho bài toán xử lý dữ liệu AI và biện luận lý do lựa chọn.
- **A1.2**: Đọc và ghi dữ liệu JSON bằng module `json` của Python, bao gồm tuần tự hóa đối tượng tùy chỉnh (Custom Object Serialization).
- **A1.3**: Định nghĩa `@dataclass` với đầy đủ type hints và giá trị mặc định (`field(default_factory=...)`) để mô hình hóa dữ liệu AI có cấu trúc chặt chẽ.
- **A1.4**: Chuyển đổi danh sách cá thể `@dataclass` thành dictionary để tuần tự hóa hai chiều (JSON serialization với `asdict`, `to_dict`, `from_dict`).

## 2. Lý thuyết nền tảng & Kiến trúc hệ thống
### 2.1. So sánh các cấu trúc dữ liệu cốt lõi trong AI Pipeline
- **List vs Set**:
  - `list` duy trì thứ tự chèn, cho phép truy cập theo chỉ số $O(1)$ và lưu trữ các phần tử trùng lặp; thích hợp để lưu chuỗi token hoặc mảng tin nhắn trong ngữ cảnh hội thoại.
  - `set` không duy trì thứ tự nhưng tối ưu hóa tốc độ tìm kiếm và kiểm tra thành viên `x in s` với độ phức tạp $O(1)$; thường được sử dụng để lọc stop-words, lưu từ vựng (vocabulary) và khử trùng lặp (deduplication).
- **Dict vs Dataclass**:
  - `dict` thô rất linh hoạt nhưng không kiểm soát cấu trúc (schema-less), dễ phát sinh lỗi `KeyError` và không có tự động hoàn thiện (autocomplete) trong IDE.
  - `@dataclass` cung cấp Type Safety, tự động sinh các phương thức đặc biệt (`__init__`, `__repr__`, `__eq__`), hỗ trợ kiểm tra kiểu tĩnh (Static Type Checking) và có phương thức xác thực `__post_init__`.
- **Tuple**:
  - Cấu trúc dữ liệu bất biến (immutable), an toàn khi dùng làm khóa cho dictionary (ví dụ: `(user_id, session_id)`), tối ưu bộ nhớ và bảo đảm tính toàn vẹn khi lưu trữ vector tọa độ cố định.

### 2.2. Kỹ thuật Phòng vệ dữ liệu (Defensive Data Validation)
- Hàm `__post_init__` được tự động gọi ngay sau hàm khởi tạo của `@dataclass`. Đây là chốt chặn quan trọng để:
  - Kiểm tra tính rỗng của văn bản (`content.strip()`).
  - Kiểm định miền giá trị hợp lệ (ví dụ: số tuần học `1 <= week <= 8`, điểm số `1 <= rating <= 5`).
  - Ném ngoại lệ `ValueError` ngay lập tức để ngăn chặn dữ liệu rác đi sâu vào RAG Pipeline.

## 3. Các bài tập Lab & Mã nguồn mẫu từng bước (Step-by-step Exercises)
### Exercise 1.1 — Code Along 1: Load, Filter & Sort Reviews (ex1_1_filter_sort.py)
```python
import json

with open("data/reviews.json", "r", encoding="utf-8") as f:
    reviews = json.load(f)

# Lọc các bài đánh giá có nội dung văn bản
with_text = [r for r in reviews if r.get("text")]
print(f"Loaded: {len(reviews)}, With text: {len(with_text)}")

# Sắp xếp giảm dần theo rating
top_rated = sorted(with_text, key=lambda r: r["rating"], reverse=True)
for r in top_rated[:3]:
    print(f"  {r['product']}: {r['rating']} - {r['text'][:50]}...")
```

### Exercise 1.2 — Code Along 2: Parse Mock API Response (ex1_2_parse_api.py)
```python
def extract_assistant_content(api_response: dict) -> str:
    choices = api_response.get("choices", [])
    if not choices:
        return "No response"
    return choices[0].get("message", {}).get("content", "")

def safe_extract_nested(data: dict, path: str, default=None):
    keys = path.split(".")
    curr = data
    for k in keys:
        if isinstance(curr, dict):
            curr = curr.get(k)
        else:
            return default
    return curr if curr is not None else default
```

### Exercise 2.2 — Custom JSON Encoder cho Dataclass (ex2_2_dataclass_json.py)
```python
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json

@dataclass
class EmbeddingResult:
    vector: list[float]
    model: str = "text-embedding-3-small"
    token_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    text: str = ""

class DataclassEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return super().default(obj)
```

## 4. Các bẫy thường gặp & Nguyên tắc thiết kế (Gotchas & Best Practices)
- **Bẫy Default Mutable Arguments**: Tuyệt đối không dùng `topics: list = []` trong `@dataclass`, phải dùng `field(default_factory=list)` để tránh hiện tượng mọi đối tượng chia sẻ chung một danh sách bộ nhớ.
- **Bẫy Datetime Serialization**: `json.dumps()` mặc định không thể serialize đối tượng `datetime`, cần chuyển thành chuỗi ISO (`isoformat()`) trước khi dump ra đĩa.
- **Bẫy Khuyết khóa Dictionary**: Khi truy cập dữ liệu JSON thô, luôn ưu tiên sử dụng `dict.get(key, default)` thay vì `dict[key]` để tránh `KeyError` làm sập chương trình.

## 5. Nhiệm vụ tự học & Đánh giá năng lực (Independent Labs)
- Viết pipeline `ex1_3_data_pipeline.py` đọc danh sách reviews, khử trùng lặp theo `review_id` bằng `set()`, gom nhóm tính rating trung bình theo từng sản phẩm và lưu ra `data/top_5_products.json`.
