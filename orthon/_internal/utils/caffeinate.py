import subprocess
import sys
import atexit
import platform


_caffeinate_process = None


def prevent_sleep():
    """
    Prevent macOS from sleeping while the current process runs.
    No-op on non-macOS systems.
    """
    global _caffeinate_process

    if platform.system() != "Darwin":
        return

    try:
        _caffeinate_process = subprocess.Popen(
            ["caffeinate", "-dims"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        atexit.register(_cleanup)

    except Exception:
        # Never fail the job because of this
        pass


def _cleanup():
    global _caffeinate_process
    if _caffeinate_process:
        try:
            _caffeinate_process.terminate()
        except Exception:
            pass
        _caffeinate_process = None
