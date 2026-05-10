"""
Module giám sát và hiển thị tiến trình theo thời gian thực.
Sử dụng thư viện rich để hiển thị bảng điều khiển chuyên nghiệp.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)


class ProgressMonitor:
    """
    Theo dõi và hiển thị tiến trình sinh dữ liệu theo thời gian thực.

    Hiển thị:
        - Thanh tiến trình với ETA.
        - Bảng thống kê: số lượng thành công/thất bại, tốc độ, thời gian.
        - Trạng thái hạn mức API.
        - Nhật ký hoạt động gần nhất.
    """

    MAX_LOG_LINES = 8

    def __init__(self, total_count: int, doc_type: str, console: Optional[Console] = None):
        """
        Khởi tạo ProgressMonitor.

        Tham số:
            total_count: Tổng số mẫu cần sinh.
            doc_type: Loại tài liệu đang sinh.
            console: Đối tượng Console của rich (tùy chọn).
        """
        self.total_count = total_count
        self.doc_type = doc_type
        self.console = console or Console()

        # Bộ đếm tiến trình
        self._success_count = 0
        self._failure_count = 0
        self._current_id = ""
        self._start_time = time.monotonic()

        # Trạng thái API và quota
        self._api_status = "Chua kiem tra"
        self._quota_percent = 0.0
        self._tokens_used = 0

        # Nhật ký hoạt động gần nhất
        self._recent_logs: list = []

        # Đối tượng rich
        self._progress = None
        self._task_id: Optional[TaskID] = None
        self._live: Optional[Live] = None

        logger.debug(
            "Khoi tao ProgressMonitor: %d mau, loai tai lieu: %s.",
            total_count,
            doc_type,
        )

    def start(self) -> None:
        """Bắt đầu hiển thị bảng điều khiển tiến trình."""
        self._start_time = time.monotonic()

        self._progress = Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="bright_green"),
            MofNCompleteColumn(),
            TextColumn("[cyan]{task.percentage:>5.1f}%"),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            console=self.console,
            expand=False,
        )

        self._task_id = self._progress.add_task(
            f"Sinh {self.doc_type}", total=self.total_count
        )

        self._live = Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()
        self._log_event(f"Bat dau sinh {self.total_count} mau [{self.doc_type}]")

    def update(
        self,
        success: bool,
        sample_id: str = "",
        error_message: str = "",
        tokens_used: int = 0,
        quota_percent: float = 0.0,
    ) -> None:
        """
        Cập nhật tiến trình sau khi sinh xong một mẫu.

        Tham số:
            success: Mẫu có được sinh thành công không.
            sample_id: ID của mẫu vừa sinh.
            error_message: Thông báo lỗi nếu thất bại.
            tokens_used: Số token đã tiêu thụ.
            quota_percent: Phần trăm quota đã dùng.
        """
        if success:
            self._success_count += 1
            self._current_id = sample_id
            self._log_event(f"[OK] {sample_id}")
        else:
            self._failure_count += 1
            short_err = error_message[:60] if error_message else "Loi khong xac dinh"
            self._log_event(f"[THAT BAI] {short_err}")

        self._tokens_used += tokens_used
        self._quota_percent = quota_percent

        # Cập nhật thanh tiến trình
        if self._progress and self._task_id is not None:
            self._progress.advance(self._task_id, 1)

        # Cập nhật giao diện
        if self._live:
            self._live.update(self._build_layout())

    def set_api_status(self, status: str) -> None:
        """
        Cập nhật trạng thái kết nối API.

        Tham số:
            status: Chuỗi mô tả trạng thái ('kha_dung', 'loi', ...).
        """
        self._api_status = status
        if self._live:
            self._live.update(self._build_layout())

    def finish(self, summary_message: str = "") -> None:
        """
        Kết thúc phiên giám sát và hiển thị tóm tắt cuối cùng.

        Tham số:
            summary_message: Thông báo tóm tắt bổ sung.
        """
        if self._live:
            self._live.stop()

        elapsed = time.monotonic() - self._start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        # In bảng tóm tắt cuối cùng
        summary = Table(title="Ket qua sinh du lieu", show_header=False, box=None)
        summary.add_column("Nhan", style="bold cyan", width=30)
        summary.add_column("Gia tri", style="white")

        total_done = self._success_count + self._failure_count
        ty_le = (
            (self._success_count / total_done * 100) if total_done > 0 else 0.0
        )

        summary.add_row("Loai tai lieu", self.doc_type.upper())
        summary.add_row("Tong mau yeu cau", str(self.total_count))
        summary.add_row("Thanh cong", f"[green]{self._success_count}[/green]")
        summary.add_row("That bai", f"[red]{self._failure_count}[/red]")
        summary.add_row("Ti le thanh cong", f"[bold]{ty_le:.1f}%[/bold]")
        summary.add_row("Thoi gian thuc hien", elapsed_str)
        if elapsed > 0 and self._success_count > 0:
            toc_do = self._success_count / elapsed
            summary.add_row("Toc do trung binh", f"{toc_do:.2f} mau/giay")
        summary.add_row("Token da su dung", f"{self._tokens_used:,}")

        if summary_message:
            summary.add_row("Ghi chu", summary_message)

        self.console.print()
        self.console.print(Panel(summary, border_style="green", padding=(1, 2)))

    def _build_layout(self) -> Layout:
        """Xây dựng layout giao diện rich."""
        layout = Layout()
        layout.split_column(
            Layout(self._build_header_panel(), name="header", size=3),
            Layout(self._build_progress_panel(), name="progress", size=5),
            Layout(name="body", ratio=1),
        )
        layout["body"].split_row(
            Layout(self._build_stats_panel(), name="stats"),
            Layout(self._build_log_panel(), name="logs"),
        )
        return layout

    def _build_header_panel(self) -> Panel:
        """Xây dựng thanh tiêu đề."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title_text = Text()
        title_text.append("Cong cu Sinh Du lieu OCR Tong hop  |  ", style="bold white")
        title_text.append(f"Loai: {self.doc_type.upper()}", style="bold yellow")
        title_text.append(f"  |  {now}", style="dim white")
        return Panel(title_text, style="blue", padding=(0, 1))

    def _build_progress_panel(self) -> Panel:
        """Xây dựng thanh tiến trình."""
        if self._progress:
            return Panel(self._progress, title="Tien trinh", border_style="cyan", padding=(0, 1))
        return Panel("Dang khoi dong...", border_style="dim")

    def _build_stats_panel(self) -> Panel:
        """Xây dựng bảng thống kê."""
        stats = Table(show_header=False, box=None, padding=(0, 1))
        stats.add_column("Nhan", style="dim", width=22)
        stats.add_column("Gia tri", style="bold white")

        total_done = self._success_count + self._failure_count
        elapsed = time.monotonic() - self._start_time
        toc_do = (total_done / elapsed) if elapsed > 0 and total_done > 0 else 0.0
        ty_le = (
            (self._success_count / total_done * 100) if total_done > 0 else 0.0
        )
        remaining = self.total_count - total_done
        eta_str = (
            str(timedelta(seconds=int(remaining / toc_do)))
            if toc_do > 0 and remaining > 0
            else "--:--:--"
        )

        stats.add_row("Thanh cong", f"[green]{self._success_count:,}[/green]")
        stats.add_row("That bai", f"[red]{self._failure_count:,}[/red]")
        stats.add_row("Ti le thanh cong", f"[bold]{ty_le:.1f}%[/bold]")
        stats.add_row("Toc do", f"{toc_do:.2f} mau/giay")
        stats.add_row("Thoi gian da qua", str(timedelta(seconds=int(elapsed))))
        stats.add_row("ETA", f"[cyan]{eta_str}[/cyan]")
        stats.add_row("Token da dung", f"{self._tokens_used:,}")

        # Thanh quota
        quota_color = "green"
        if self._quota_percent >= 95:
            quota_color = "red"
        elif self._quota_percent >= 80:
            quota_color = "yellow"
        stats.add_row(
            "Quota API",
            f"[{quota_color}]{self._quota_percent:.1f}%[/{quota_color}]",
        )
        stats.add_row("Trang thai API", self._api_status)

        return Panel(stats, title="Thong ke", border_style="green", padding=(0, 1))

    def _build_log_panel(self) -> Panel:
        """Xây dựng ô nhật ký hoạt động gần nhất."""
        log_text = Text()
        for line in self._recent_logs[-self.MAX_LOG_LINES:]:
            timestamp, message = line
            log_text.append(f"[{timestamp}] ", style="dim")
            if "[OK]" in message:
                log_text.append(message + "\n", style="green")
            elif "[THAT BAI]" in message:
                log_text.append(message + "\n", style="red")
            else:
                log_text.append(message + "\n", style="white")

        return Panel(log_text, title="Nhat ky hoat dong", border_style="yellow", padding=(0, 1))

    def _log_event(self, message: str) -> None:
        """Thêm một sự kiện vào nhật ký nội bộ."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._recent_logs.append((timestamp, message))
        # Giữ tối đa 50 dòng trong bộ nhớ
        if len(self._recent_logs) > 50:
            self._recent_logs = self._recent_logs[-50:]


class SimpleProgressMonitor:
    """
    Phiên bản đơn giản dùng tqdm khi rich không khả dụng.
    Dự phòng khi môi trường không hỗ trợ terminal đầy đủ.
    """

    def __init__(self, total_count: int, doc_type: str, **kwargs):
        self.total_count = total_count
        self.doc_type = doc_type
        self._success_count = 0
        self._failure_count = 0
        self._start_time = time.monotonic()
        self._pbar = None

        try:
            from tqdm import tqdm
            self._pbar = tqdm(
                total=total_count,
                desc=f"Sinh {doc_type}",
                unit="mau",
                ncols=100,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
            )
        except ImportError:
            print(f"[THONG TIN] Bat dau sinh {total_count} mau loai '{doc_type}'...")

    def start(self) -> None:
        self._start_time = time.monotonic()

    def update(self, success: bool, sample_id: str = "", error_message: str = "", **kwargs) -> None:
        if success:
            self._success_count += 1
        else:
            self._failure_count += 1
        if self._pbar:
            self._pbar.update(1)
            self._pbar.set_postfix(
                thanh_cong=self._success_count,
                that_bai=self._failure_count,
            )
        else:
            total = self._success_count + self._failure_count
            if total % 10 == 0:
                print(
                    f"[TIEN TRINH] {total}/{self.total_count} "
                    f"(Thanh cong: {self._success_count}, That bai: {self._failure_count})"
                )

    def set_api_status(self, status: str) -> None:
        pass

    def finish(self, summary_message: str = "") -> None:
        if self._pbar:
            self._pbar.close()
        elapsed = time.monotonic() - self._start_time
        ty_le = (
            (self._success_count / (self._success_count + self._failure_count) * 100)
            if (self._success_count + self._failure_count) > 0
            else 0
        )
        print(
            f"\n[HOAN THANH] Thanh cong: {self._success_count}, "
            f"That bai: {self._failure_count}, "
            f"Ti le: {ty_le:.1f}%, "
            f"Thoi gian: {timedelta(seconds=int(elapsed))}"
        )
        if summary_message:
            print(f"[GHI CHU] {summary_message}")


def create_progress_monitor(
    total_count: int,
    doc_type: str,
    prefer_rich: bool = True,
) -> "ProgressMonitor | SimpleProgressMonitor":
    """
    Factory: Tạo monitor phù hợp với môi trường hiện tại.

    Tham số:
        total_count: Tổng số mẫu cần sinh.
        doc_type: Loại tài liệu.
        prefer_rich: Ưu tiên dùng rich nếu khả dụng.

    Trả về:
        Đối tượng ProgressMonitor hoặc SimpleProgressMonitor.
    """
    if prefer_rich:
        try:
            import rich  # noqa: F401
            return ProgressMonitor(total_count, doc_type)
        except ImportError:
            logger.warning("Thu vien 'rich' khong kha dung, dung SimpleProgressMonitor.")
    return SimpleProgressMonitor(total_count, doc_type)

