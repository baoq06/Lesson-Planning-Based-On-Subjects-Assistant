.PHONY: setup demo eval multi second_host

# Cài đặt toàn bộ thư viện cần thiết
setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

# Chạy chương trình tương tác trực tiếp qua terminal (Single Agent ReAct)
demo:
	python -m src.agent.loop

# Chạy hệ thống Multi-Agent (Producer + Critic CLI)
multi:
	python -m src.agent.team_critic

# Kiểm thử MCP Server từ Host/Client thứ hai độc lập (+1 điểm Interop)
second_host:
	python scripts/test_second_host.py

# Đánh giá benchmark định lượng hệ thống (kèm so sánh 2 cấu hình Hybrid vs Dense)
# Mặc định 'make eval' chạy từ đầu. Để chạy tiếp từ câu bị ngắt: make eval ARGS="--resume"
eval:
	python -m tests.eval $(ARGS)
	python -m tests.compare_configs
