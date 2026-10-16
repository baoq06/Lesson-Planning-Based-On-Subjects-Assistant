.PHONY: setup demo eval

# Cài đặt toàn bộ thư viện cần thiết
setup:
	python -m pip install --upgrade pip
	pip install groq python-dotenv rich mcp rank_bm25 numpy requests matplotlib

# Chạy chương trình tương tác trực tiếp qua terminal
demo:
	python -m src.agent.loop

# Đánh giá benchmark định lượng hệ thống (kèm so sánh 2 cấu hình Hybrid vs Dense)
eval:
	python -m tests.eval
	python -m tests.compare_configs
