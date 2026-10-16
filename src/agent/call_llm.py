import sys
import re
from typing import Any, Optional
from dotenv import load_dotenv
from rich import print as rprint
from pathlib import Path
import time
import random
import logging
from groq import (
    Groq,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
)

logger = logging.getLogger("agent.llm")

MODEL = "qwen/qwen3.8-27b"

def call_llm(
    client: Groq,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    model: str = MODEL,
    temperature: float = 0.1,
    max_retries: int = 5,
    base_delay: float = 1.0,
    timeout: float = 30.0,
) -> Any:
    """
    Gọi LLM API có kiểm soát timeout và retry với Exponential Backoff:
    - Retry với: 429 (Rate limit), 5xx (Server error), Timeout, Connection error.
    - KHÔNG retry với: 400 (Bad request), 401 (Auth error), 404 (Not found).
    - Toàn bộ log lỗi HTTP và số lần thử được đẩy ra sys.stderr.
    """
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
    }
    if tools:
        kwargs["tools"] = tools

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)

        # 1. Nhóm lỗi KHÔNG RETRY (400, 401, 404)
        except (AuthenticationError, BadRequestError, NotFoundError) as e:
            status_code = getattr(e, "status_code", "4xx")
            err_msg = f"[NON-RETRYABLE] [HTTP {status_code}] Lỗi không thể khắc phục bằng retry: {e}"
            print(err_msg, file=sys.stderr)
            logger.error(err_msg)
            raise e  

        # 2. Nhóm lỗi RETRYABLE (429, 5xx, Mất kết nối, Timeout)
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError, APIStatusError) as e:
            status_code = getattr(e, "status_code", "Network/Timeout")

            # Nếu là lỗi 4xx khác (ngoại trừ 429) -> Không retry
            if isinstance(e, APIStatusError) and isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 429:
                err_msg = f"[NON-RETRYABLE] [HTTP {status_code}] Client error: {e}"
                print(err_msg, file=sys.stderr)
                logger.error(err_msg)
                raise e

            if attempt < max_retries:
                # Công thức chuẩn: delay = base * 2^attempt + jitter ngẫu nhiên
                delay = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)

                # Nếu là 429 RateLimitError, ưu tiên đọc thời gian chờ do Groq đề xuất
                if isinstance(e, RateLimitError) or status_code == 429:
                    match = re.search(r"try again in ([0-9.]+)s", str(e))
                    if match:
                        groq_wait = float(match.group(1)) + 0.5
                        delay = max(delay, groq_wait)
                    else:
                        delay = max(delay, 4.0)

                warn_msg = (
                    f"[RETRYABLE] [HTTP {status_code}] Gọi LLM thất bại (thử lần {attempt + 1}/{max_retries}). "
                    f"Đợi {delay:.2f}s trước khi thử lại... Lỗi: {e}"
                )
                print(warn_msg, file=sys.stderr)
                logger.warning(warn_msg)
                time.sleep(delay)
            else:
                fail_msg = f"[FAILED] [HTTP {status_code}] Đã hết {max_retries} lần retry mà vẫn thất bại: {e}"
                print(fail_msg, file=sys.stderr)
                logger.error(fail_msg)
                raise e

        except Exception as e:
            print(f"[UNKNOWN ERROR] Lỗi không xác định khi gọi LLM: {e}", file=sys.stderr)
            raise e
