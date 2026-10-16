.PHONY: setup demo eval

# Cài đặt toàn bộ thư viện cần thiết
setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

# Chạy chương trình tương tác trực tiếp qua terminal
demo:
	python -m src.agent.loop

# Đánh giá benchmark định lượng hệ thống (kèm so sánh 2 cấu hình Hybrid vs Dense)
# Mặc định 'make eval' chạy từ đầu. Để chạy tiếp từ câu bị ngắt: make eval ARGS="--resume"
eval:
	python -m tests.eval $(ARGS)
	python -m tests.compare_configs
