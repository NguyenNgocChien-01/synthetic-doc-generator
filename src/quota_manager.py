"""
Module quản lý hạn mức tài nguyên API (Quota Manager).
Theo dõi lượng token tiêu thụ, ước tính chi phí và cảnh báo
khi tiến gần đến giới hạn cho phép.
"""

import json
import logging
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Số ký tự trung bình ước tính tương đương 1 token
CHARS_PER_TOKEN_ESTIMATE = 4

# Số token ước tính cho mỗi yêu cầu sinh ảnh đại diện
AVATAR_REQUEST_TOKEN_ESTIMATE = 500


class QuotaStatus:
    """Trạng thái sử dụng hạn mức."""
    BINH_THUONG = "binh_thuong"
    CANH_BAO = "canh_bao"
    NGUY_HIEM = "nguy_hiem"
    VUOT_GIOI_HAN = "vuot_gioi_han"


class QuotaExceededError(Exception):
    """Ngoại lệ khi đã vượt quá giới hạn hạn mức."""
    pass


class QuotaWarningError(Exception):
    """Ngoại lệ cảnh báo khi gần đến giới hạn hạn mức."""
    pass


class QuotaManager:
    """
    Quản lý hạn mức (quota) và lượng token tiêu thụ khi gọi API.

    Chức năng:
        - Theo dõi số token đã dùng trong ngày.
        - Ước tính số token cần dùng cho một yêu cầu.
        - Cảnh báo khi tiêu thụ vượt ngưỡng cho phép.
        - Lưu trạng thái để khôi phục khi khởi động lại.
        - An toàn đa luồng.
    """

    def __init__(
        self,
        daily_token_limit: int = 1_000_000,
        warning_threshold_percent: float = 80.0,
        critical_threshold_percent: float = 95.0,
        state_file: Optional[str] = None,
    ):
        """
        Khởi tạo QuotaManager.

        Tham số:
            daily_token_limit: Giới hạn token tối đa mỗi ngày.
            warning_threshold_percent: Ngưỡng cảnh báo (%).
            critical_threshold_percent: Ngưỡng nguy hiểm (%).
            state_file: Đường dẫn tệp lưu trạng thái. None = không lưu.
        """
        self.daily_token_limit = daily_token_limit
        self.warning_threshold = daily_token_limit * (warning_threshold_percent / 100)
        self.critical_threshold = daily_token_limit * (critical_threshold_percent / 100)
        self.state_file = Path(state_file) if state_file else None

        self._lock = threading.Lock()
        self._tokens_used_today = 0
        self._requests_today = 0
        self._current_date = date.today()
        self._session_tokens = 0
        self._session_requests = 0
        self._failed_requests = 0

        # Tải trạng thái từ tệp nếu có
        self._load_state()

        logger.info(
            "Khoi tao QuotaManager: gioi han %d token/ngay, canh bao tai %.0f%%.",
            daily_token_limit,
            warning_threshold_percent,
        )

    def estimate_tokens(self, text_content: str = "", image_request: bool = False) -> int:
        """
        Ước tính số token cần dùng cho một yêu cầu.

        Tham số:
            text_content: Nội dung văn bản gửi đi (nếu có).
            image_request: Có phải yêu cầu sinh ảnh không.

        Trả về:
            Số token ước tính.
        """
        text_tokens = len(text_content) // CHARS_PER_TOKEN_ESTIMATE
        image_tokens = AVATAR_REQUEST_TOKEN_ESTIMATE if image_request else 0
        # Thêm overhead cho system prompt và định dạng
        overhead = 50
        return text_tokens + image_tokens + overhead

    def check_availability(self, estimated_tokens: int = 0) -> Tuple[bool, str]:
        """
        Kiểm tra xem có đủ hạn mức để thực hiện yêu cầu không.

        Tham số:
            estimated_tokens: Số token ước tính sẽ dùng.

        Trả về:
            Tuple (co_the_tiep_tuc, thong_bao).
        """
        with self._lock:
            self._refresh_if_new_day()

            projected = self._tokens_used_today + estimated_tokens

            if projected >= self.daily_token_limit:
                msg = (
                    f"Vuot gioi han token: da dung {self._tokens_used_today:,}, "
                    f"can them {estimated_tokens:,}, gioi han {self.daily_token_limit:,}."
                )
                return False, msg

            if projected >= self.critical_threshold:
                msg = (
                    f"NGUY HIEM: Gan het han muc. "
                    f"Da dung {self._tokens_used_today:,}/{self.daily_token_limit:,} "
                    f"({self._get_usage_percent():.1f}%)."
                )
                return True, msg

            if projected >= self.warning_threshold:
                msg = (
                    f"CANH BAO: Da dung {self._get_usage_percent():.1f}% han muc ngay."
                )
                return True, msg

            return True, ""

    def record_usage(self, tokens_used: int, success: bool = True) -> None:
        """
        Ghi nhận lượng token đã sử dụng sau khi hoàn thành yêu cầu.

        Tham số:
            tokens_used: Số token thực tế đã dùng.
            success: Yêu cầu có thành công không.
        """
        with self._lock:
            self._refresh_if_new_day()

            self._tokens_used_today += tokens_used
            self._session_tokens += tokens_used
            self._requests_today += 1
            self._session_requests += 1

            if not success:
                self._failed_requests += 1

            logger.debug(
                "Ghi nhan su dung: +%d token, tong hom nay: %d/%d.",
                tokens_used,
                self._tokens_used_today,
                self.daily_token_limit,
            )

            # Ghi cảnh báo nếu cần
            percent = self._get_usage_percent()
            if percent >= 95:
                logger.critical(
                    "NGUY HIEM: Da su dung %.1f%% han muc token trong ngay!", percent
                )
            elif percent >= 80:
                logger.warning(
                    "CANH BAO: Da su dung %.1f%% han muc token trong ngay.", percent
                )

            # Lưu trạng thái sau mỗi lần ghi nhận
            self._save_state()

    def get_status(self) -> Dict:
        """
        Lấy thông tin trạng thái đầy đủ của quota.

        Trả về:
            Từ điển chứa các thông số trạng thái.
        """
        with self._lock:
            self._refresh_if_new_day()
            percent = self._get_usage_percent()
            remaining = max(0, self.daily_token_limit - self._tokens_used_today)

            if percent >= 100:
                trang_thai = QuotaStatus.VUOT_GIOI_HAN
            elif percent >= 95:
                trang_thai = QuotaStatus.NGUY_HIEM
            elif percent >= 80:
                trang_thai = QuotaStatus.CANH_BAO
            else:
                trang_thai = QuotaStatus.BINH_THUONG

            return {
                "trang_thai": trang_thai,
                "token_da_dung_hom_nay": self._tokens_used_today,
                "token_con_lai_hom_nay": remaining,
                "gioi_han_hang_ngay": self.daily_token_limit,
                "phan_tram_su_dung": round(percent, 2),
                "yeu_cau_hom_nay": self._requests_today,
                "token_phien_lam_viec": self._session_tokens,
                "yeu_cau_phien_lam_viec": self._session_requests,
                "yeu_cau_that_bai": self._failed_requests,
                "ngay_hien_tai": self._current_date.isoformat(),
                "thoi_gian_cap_nhat": datetime.now().isoformat(),
            }

    def get_remaining_capacity(self) -> int:
        """
        Tính số yêu cầu bình thường còn có thể thực hiện.

        Trả về:
            Số yêu cầu ước tính còn lại.
        """
        with self._lock:
            remaining_tokens = max(0, self.daily_token_limit - self._tokens_used_today)
            avg_tokens_per_request = AVATAR_REQUEST_TOKEN_ESTIMATE + 200
            if avg_tokens_per_request == 0:
                return 0
            return remaining_tokens // avg_tokens_per_request

    def estimate_batch_feasibility(self, count: int) -> Dict:
        """
        Đánh giá khả năng thực hiện một lô yêu cầu.

        Tham số:
            count: Số yêu cầu trong lô.

        Trả về:
            Báo cáo đánh giá khả năng thực thi.
        """
        avg_tokens = AVATAR_REQUEST_TOKEN_ESTIMATE + 200
        total_estimated = count * avg_tokens
        remaining = max(0, self.daily_token_limit - self._tokens_used_today)
        feasible_count = min(count, remaining // avg_tokens) if avg_tokens > 0 else count

        return {
            "so_luong_yeu_cau": count,
            "token_uoc_tinh_tong": total_estimated,
            "token_con_lai": remaining,
            "co_the_thuc_hien": feasible_count,
            "kha_thi": feasible_count >= count,
            "phan_tram_co_the_thuc_hien": round((feasible_count / count * 100) if count > 0 else 0, 1),
        }

    def _get_usage_percent(self) -> float:
        """Tính phần trăm sử dụng (nội bộ, không khóa)."""
        if self.daily_token_limit == 0:
            return 0.0
        return (self._tokens_used_today / self.daily_token_limit) * 100

    def _refresh_if_new_day(self) -> None:
        """Đặt lại bộ đếm nếu bước sang ngày mới (nội bộ, không khóa)."""
        today = date.today()
        if today != self._current_date:
            logger.info(
                "Ngay moi (%s): dat lai bo dem token. Hom qua da dung %d token.",
                today.isoformat(),
                self._tokens_used_today,
            )
            self._tokens_used_today = 0
            self._requests_today = 0
            self._current_date = today
            self._save_state()

    def _save_state(self) -> None:
        """Lưu trạng thái hiện tại ra tệp JSON."""
        if not self.state_file:
            return
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "ngay": self._current_date.isoformat(),
                "token_da_dung": self._tokens_used_today,
                "yeu_cau_da_thuc_hien": self._requests_today,
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except OSError as loi:
            logger.warning("Khong the luu trang thai quota: %s", loi)

    def _load_state(self) -> None:
        """Tải trạng thái từ tệp JSON nếu tồn tại."""
        if not self.state_file or not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            saved_date = date.fromisoformat(state.get("ngay", "1970-01-01"))
            if saved_date == date.today():
                self._tokens_used_today = state.get("token_da_dung", 0)
                self._requests_today = state.get("yeu_cau_da_thuc_hien", 0)
                logger.info(
                    "Tai trang thai quota: da dung %d token hom nay.",
                    self._tokens_used_today,
                )
        except (OSError, json.JSONDecodeError, ValueError) as loi:
            logger.warning("Khong the tai trang thai quota: %s", loi)
