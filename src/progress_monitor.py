"""
Module for monitoring and displaying real-time progress.
Uses the rich library for professional control panel display.
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
    For monitoring and displaying real-time progress.

    Displays:
        - A progress bar with ETA.
        - A summary table: number of successes/failures, speed, time.
        - API rate limit status.
        - Recent activity log.
    """

    MAX_LOG_LINES = 8

    def __init__(self, total_count: int, doc_type: str, console: Optional[Console] = None):
        """
        Initialize ProgressMonitor.

        Parameters:
            total_count: Total number of samples to generate.
            doc_type: Type of document being generated.
            console: Rich console object (optional).
        """
        self.total_count = total_count
        self.doc_type = doc_type
        self.console = console or Console()

        # Progress counter
        self._success_count = 0
        self._failure_count = 0
        self._current_id = ""
        self._start_time = time.monotonic()

        # API status and quota
        self._api_status = "Not yet checked"
        self._quota_percent = 0.0
        self._tokens_used = 0

        # Recent activity log
        self._recent_logs: list = []

        # Rich object
        self._progress = None
        self._task_id: Optional[TaskID] = None
        self._live: Optional[Live] = None

        logger.debug(
            "Initialized ProgressMonitor: %d samples, type: %s.",
            total_count,
            doc_type,
        )

    def start(self) -> None:
        """Start displaying the control panel."""
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
            f"Generating {self.doc_type}", total=self.total_count
        )

        self._live = Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()
        self._log_event(f"Started generating {self.total_count} samples [{self.doc_type}]")

    def update(
        self,
        success: bool,
        sample_id: str = "",
        error_message: str = "",
        tokens_used: int = 0,
        quota_percent: float = 0.0,
    ) -> None:
        """
        Update progress after generating a sample.

        Parameters:
            success: Whether the sample was generated successfully.
            sample_id: ID of the sample just generated.
            error_message: Error message if failed.
            tokens_used: Number of tokens consumed.
            quota_percent: Percentage of quota used.
        """
        if success:
            self._success_count += 1
            self._current_id = sample_id
            self._log_event(f"[OK] {sample_id}")
        else:
            self._failure_count += 1
            short_err = error_message[:60] if error_message else "Unknown error"
            self._log_event(f"[THAT BAI] {short_err}")

        self._tokens_used += tokens_used
        self._quota_percent = quota_percent

        # Update progress bar
        if self._progress and self._task_id is not None:
            self._progress.advance(self._task_id, 1)

        # Update interface
        if self._live:
            self._live.update(self._build_layout())

    def set_api_status(self, status: str) -> None:
        """
        Update API connection status.

        Parameters:
            status: String describing the status ('good', 'bad', ...).
        """
        self._api_status = status
        if self._live:
            self._live.update(self._build_layout())

    def finish(self, summary_message: str = "") -> None:
        """
        Finish monitoring and display a final summary.

        Parameters:
            summary_message: Additional summary information.
        """
        if self._live:
            self._live.stop()

        elapsed = time.monotonic() - self._start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        # In final summary table
        summary = Table(title="Final summary", show_header=False, box=None)
        summary.add_column("Nhan", style="bold cyan", width=30)
        summary.add_column("Gia tri", style="white")

        total_done = self._success_count + self._failure_count
        ty_le = (
            (self._success_count / total_done * 100) if total_done > 0 else 0.0
        )

        summary.add_row("Document type", self.doc_type.upper())
        summary.add_row("Total requested samples", str(self.total_count))
        summary.add_row("Success", f"[green]{self._success_count}[/green]")
        summary.add_row("Failure", f"[red]{self._failure_count}[/red]")
        summary.add_row("Success rate", f"[bold]{ty_le:.1f}%[/bold]")
        summary.add_row("Execution time", elapsed_str)
        if elapsed > 0 and self._success_count > 0:
            toc_do = self._success_count / elapsed
            summary.add_row("Average speed", f"{toc_do:.2f} samples/second")
        # summary.add_row("Tokens used", f"{self._tokens_used:,}")

        if summary_message:
            summary.add_row("Notes", summary_message)

        self.console.print()
        self.console.print(Panel(summary, border_style="green", padding=(1, 2)))

    def _build_layout(self) -> Layout:
        """Build rich layout."""
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
        """Build header bar."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title_text = Text()
        title_text.append("Cong cu Sinh Du lieu OCR Tong hop  |  ", style="bold white")
        title_text.append(f"Loai: {self.doc_type.upper()}", style="bold yellow")
        title_text.append(f"  |  {now}", style="dim white")
        return Panel(title_text, style="blue", padding=(0, 1))

    def _build_progress_panel(self) -> Panel:
        """Build progress bar."""
        if self._progress:
            return Panel(self._progress, title="Tien trinh", border_style="cyan", padding=(0, 1))
        return Panel("Dang khoi dong...", border_style="dim")

    def _build_stats_panel(self) -> Panel:
        """Build statistics table."""
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

        # Quota bar
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
        """Build recent activity log."""
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
        """Add an event to the internal log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._recent_logs.append((timestamp, message))
        # Keep at most 50 lines in memory
        if len(self._recent_logs) > 50:
            self._recent_logs = self._recent_logs[-50:]


class SimpleProgressMonitor:
    """
    A simpler version using tqdm when rich is not available.
    For environments without full terminal support.
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
                desc=f"Generating {doc_type}",
                unit="samples",
                ncols=100,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
            )
        except ImportError:
            print(f"[INFO] Starting generation of {total_count} samples of type '{doc_type}'...")

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
                success=self._success_count,
                failure=self._failure_count,
            )
        else:
            total = self._success_count + self._failure_count
            if total % 10 == 0:
                print(
                    f"[PROGRESS] {total}/{self.total_count} "
                    f"(Success: {self._success_count}, Failure: {self._failure_count})"
                )

    def set_api_status(self, status: str) -> None:
        pass

    def finish(self, summary_message: str = "") -> None:
        if self._pbar:
            self._pbar.close()
        elapsed = time.monotonic() - self._start_time
        success_rate = (
            (self._success_count / (self._success_count + self._failure_count) * 100)
            if (self._success_count + self._failure_count) > 0
            else 0
        )
        print(
            f"\n[COMPLETED] Success: {self._success_count}, "
            f"Failure: {self._failure_count}, "
            f"Success rate: {success_rate:.1f}%, "
            f"Elapsed time: {timedelta(seconds=int(elapsed))}"
        )
        if summary_message:
            print(f"[NOTE] {summary_message}")


def create_progress_monitor(
    total_count: int,
    doc_type: str,
    prefer_rich: bool = True,
) -> "ProgressMonitor | SimpleProgressMonitor":
    """
    Factory: Create a monitor suitable for the current environment.

    Parameters:
        total_count: Total number of samples to generate.
        doc_type: Type of document.
        prefer_rich: Prefer using rich if available.

    Returns:
        A ProgressMonitor or SimpleProgressMonitor.
    """
    if prefer_rich:
        try:
            import rich  # noqa: F401
            return ProgressMonitor(total_count, doc_type)
        except ImportError:
            logger.warning("The 'rich' library is not available, using SimpleProgressMonitor.")
    return SimpleProgressMonitor(total_count, doc_type)

