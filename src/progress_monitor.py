import sys
import time
import threading

class ProgressMonitor:
    def __init__(self, total_count: int, doc_type: str, **kwargs):
        self.total_count = total_count
        self._current_count = 0
        self._spinning = False
        self._spinner_thread = None
        self._lock = threading.Lock()

    def start(self):
        self._set_spinning(True)

    def _set_spinning(self, state: bool):
        self._spinning = state
        if state:
            self._spinner_thread = threading.Thread(target=self._spin, daemon=True)
            self._spinner_thread.start()
        elif self._spinner_thread:
            self._spinner_thread.join()

    def _spin(self):
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while self._spinning:
            with self._lock:
                sys.stdout.write(f"\r{frames[i % len(frames)]} {self._current_count} / {self.total_count}")
                sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def set_api_status(self, status: str): pass

    def update(self, success: bool, **kwargs) -> None:
        # Dừng spinner, in kết quả, rồi chạy lại spinner
        self._spinning = False
        if self._spinner_thread:
            self._spinner_thread.join()

        self._current_count += 1
        icon = "OK" if success else "Failed"
        sys.stdout.write(f"\r{icon} {self._current_count} / {self.total_count}\n")
        sys.stdout.flush()

        self._set_spinning(True)

    def finish(self, summary_message: str = ""):
        self._spinning = False
        if self._spinner_thread:
            self._spinner_thread.join()
        sys.stdout.write(f"\r Done! {self._current_count} / {self.total_count}\n")
        sys.stdout.flush()


def create_progress_monitor(total_count: int, doc_type: str, prefer_rich: bool = True):
    return ProgressMonitor(total_count, doc_type)