.PHONY: setup demo run eval eval-quick clean

# Cài đặt toàn bộ thư viện cần thiết
setup:
	python -m pip install --upgrade pip
	pip install groq python-dotenv rich mcp rank_bm25 numpy requests

# Chạy chương trình tương tác trực tiếp qua terminal
demo:
	python -m src.agent.loop

run: demo

# Đánh giá benchmark định lượng hệ thống (Rubric 10 điểm Đánh giá & bằng chứng)
eval:
	python -m tests.eval

# Đánh giá nhanh (2 câu smoke test)
eval-quick:
	python -m tests.eval --limit 2

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
