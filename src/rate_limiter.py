"""
Module kiểm soát tốc độ gọi API sử dụng thuật toán Token Bucket.
Đảm bảo số lượng yêu cầu không vượt quá giới hạn cho phép
và xử lý tự động thử lại khi nhận lỗi vượt định mức.
"""

import time
import logging
import threading
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class RateLimitExceededError(Exception):
    """Ngoại lệ khi vượt quá giới hạn tốc độ và đã dùng hết số lần thử lại."""
    pass


class RateLimiter:
    """
    Kiểm soát tốc độ gọi API bằng thuật toán Token Bucket.

    Thuật toán Token Bucket cho phép xử lý đột biến ngắn hạn
    trong khi vẫn duy trì giới hạn trung bình dài hạn.

    Đặc điểm:
        - An toàn đa luồng (thread-safe).
        - Hỗ trợ chiến lược thử lại với backoff lũy thừa.
        - Ghi log chi tiết bằng tiếng Việt.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_multiplier: float = 1.5,
        max_retries: int = 3,
        initial_retry_delay: float = 2.0,
        backoff_factor: float = 2.0,
        max_retry_delay: float = 60.0,
    ):
        """
        Khởi tạo RateLimiter.

        Tham số:
            requests_per_minute: Số yêu cầu tối đa mỗi phút.
            burst_multiplier: Hệ số cho phép đột biến (ví dụ: 1.5 = 150% RPM).
            max_retries: Số lần thử lại tối đa khi gặp lỗi.
            initial_retry_delay: Thời gian chờ ban đầu (giây) trước khi thử lại.
            backoff_factor: Hệ số nhân thời gian chờ sau mỗi lần thất bại.
            max_retry_delay: Thời gian chờ tối đa giữa các lần thử lại (giây).
        """
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.backoff_factor = backoff_factor
        self.max_retry_delay = max_retry_delay

        # Tham số Token Bucket
        self._lock = threading.Lock()
        if requests_per_minute > 0:
            self._tokens_per_second = requests_per_minute / 60.0
            self._max_tokens = requests_per_minute * burst_multiplier
            self._current_tokens = self._max_tokens
        else:
            # Không giới hạn tốc độ
            self._tokens_per_second = float("inf")
            self._max_tokens = float("inf")
            self._current_tokens = float("inf")

        self._last_refill_time = time.monotonic()

        # Thống kê
        self._total_waits = 0
        self._total_wait_time = 0.0
        self._total_retries = 0

        logger.info(
            "Khởi tạo RateLimiter: %d yeu cau/phut, toi da %d lan thu lai.",
            requests_per_minute,
            max_retries,
        )

    def _refill_tokens(self) -> None:
        """Nạp lại token dựa trên thời gian đã trôi qua."""
        now = time.monotonic()
        elapsed = now - self._last_refill_time
        self._last_refill_time = now

        if self._tokens_per_second == float("inf"):
            return

        new_tokens = elapsed * self._tokens_per_second
        self._current_tokens = min(self._current_tokens + new_tokens, self._max_tokens)

    def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Yêu cầu token để thực hiện một lần gọi API.
        Nếu không đủ token, hàm sẽ chặn (block) cho đến khi có đủ.

        Tham số:
            tokens: Số token cần dùng (mặc định là 1).
            timeout: Thời gian tối đa chờ đợi (giây). None = chờ vô hạn.

        Trả về:
            True nếu thành công, False nếu hết thời gian chờ.
        """
        if self._tokens_per_second == float("inf"):
            return True

        deadline = (time.monotonic() + timeout) if timeout else None
        wait_start = time.monotonic()

        while True:
            with self._lock:
                self._refill_tokens()
                if self._current_tokens >= tokens:
                    self._current_tokens -= tokens
                    return True

                # Tính thời gian cần chờ để có đủ token
                tokens_needed = tokens - self._current_tokens
                wait_time = tokens_needed / self._tokens_per_second

            if deadline and time.monotonic() + wait_time > deadline:
                logger.warning("Hết thời gian chờ token (timeout).")
                return False

            logger.debug("Token bucket cạn, chờ %.2f giay...", wait_time)
            time.sleep(min(wait_time, 0.1))

            # Cập nhật thống kê
            self._total_waits += 1
            self._total_wait_time += (time.monotonic() - wait_start)

    def execute_with_retry(
        self,
        func: Callable,
        *args,
        is_rate_limit_error: Optional[Callable[[Exception], bool]] = None,
        **kwargs,
    ) -> Any:
        """
        Thực thi một hàm với cơ chế thử lại tự động.

        Chiến lược thử lại sử dụng backoff lũy thừa: thời gian chờ
        tăng dần theo hệ số sau mỗi lần thất bại.

        Tham số:
            func: Hàm cần thực thi.
            *args: Tham số vị trí cho hàm.
            is_rate_limit_error: Hàm kiểm tra xem lỗi có phải do vượt định mức không.
            **kwargs: Tham số từ khóa cho hàm.

        Trả về:
            Kết quả của hàm nếu thành công.

        Ném ra:
            RateLimitExceededError: Nếu đã dùng hết số lần thử lại.
        """
        if is_rate_limit_error is None:
            is_rate_limit_error = self._default_rate_limit_check

        last_exception = None
        retry_delay = self.initial_retry_delay

        for attempt in range(self.max_retries + 1):
            # Kiểm soát tốc độ trước mỗi lần gọi
            self.acquire()

            try:
                return func(*args, **kwargs)
            except Exception as loi:
                last_exception = loi

                if attempt >= self.max_retries:
                    logger.error(
                        "Da thu lai %d lan, van that bai. Loi cuoi: %s",
                        self.max_retries,
                        loi,
                    )
                    break

                if is_rate_limit_error(loi):
                    # Lỗi do vượt định mức - chờ lâu hơn
                    actual_delay = min(retry_delay * 2, self.max_retry_delay)
                    logger.warning(
                        "Vuot dinh muc API (lan thu %d/%d). Cho %.1f giay truoc khi thu lai...",
                        attempt + 1,
                        self.max_retries,
                        actual_delay,
                    )
                    self._total_retries += 1
                    time.sleep(actual_delay)
                    retry_delay = min(retry_delay * self.backoff_factor, self.max_retry_delay)
                else:
                    # Lỗi khác - thử lại với thời gian chờ ngắn hơn
                    logger.warning(
                        "Loi API (lan thu %d/%d): %s. Cho %.1f giay...",
                        attempt + 1,
                        self.max_retries,
                        loi,
                        retry_delay,
                    )
                    self._total_retries += 1
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * self.backoff_factor, self.max_retry_delay)

        raise RateLimitExceededError(
            f"Da thu lai {self.max_retries} lan nhung van that bai. "
            f"Loi cuoi cung: {last_exception}"
        ) from last_exception

    def _default_rate_limit_check(self, error: Exception) -> bool:
        """
        Kiểm tra mặc định xem một ngoại lệ có phải do vượt định mức API không.

        Tham số:
            error: Ngoại lệ cần kiểm tra.

        Trả về:
            True nếu đây là lỗi vượt định mức.
        """
        error_str = str(error).lower()
        rate_limit_keywords = [
            "rate limit",
            "too many requests",
            "quota exceeded",
            "resource exhausted",
            "429",
            "rateerror",
        ]
        return any(keyword in error_str for keyword in rate_limit_keywords)

    def get_statistics(self) -> dict:
        """
        Lấy thống kê hoạt động của rate limiter.

        Trả về:
            Từ điển chứa các chỉ số thống kê.
        """
        return {
            "tong_so_lan_cho": self._total_waits,
            "tong_thoi_gian_cho_giay": round(self._total_wait_time, 2),
            "tong_so_lan_thu_lai": self._total_retries,
            "token_hien_tai": round(self._current_tokens, 2),
            "token_toi_da": self._max_tokens,
        }

    def reset_statistics(self) -> None:
        """Đặt lại tất cả thống kê về 0."""
        self._total_waits = 0
        self._total_wait_time = 0.0
        self._total_retries = 0
        logger.debug("Da dat lai thong ke RateLimiter.")
