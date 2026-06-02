import os
import time
import signal
import socket
import subprocess
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

try:
    import snap7
    from snap7.util import get_bool
except ImportError:
    print("Missing python-snap7. Please install it in .venv first.")
    raise


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(BASE_DIR, "main.py")
PYTHON_EXE = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LAUNCHER_DEBUG_LOG = os.path.join(LOG_DIR, "launcher_debug.log")
MAIN_LOG = os.path.join(LOG_DIR, "main_from_launcher.log")

PLC_IP = os.getenv("PLC_IP", "192.168.150.103")
PLC_RACK = int(os.getenv("PLC_RACK", "0"))

# S7-300 thường là slot 2. Nếu PLC của anh dùng slot 1 thì sửa trong .env:
# PLC_SLOT=1
PLC_SLOT = int(os.getenv("PLC_SLOT", "2"))

# START BIT: DB17.DBX16.4
START_DB_NUMBER = int(os.getenv("START_DB_NUMBER", "17"))
START_BYTE = int(os.getenv("START_BYTE", "16"))
START_BIT = int(os.getenv("START_BIT", "4"))

POLL_SEC = float(os.getenv("PLC_POLL_SEC", "0.5"))
RECONNECT_SEC = float(os.getenv("PLC_RECONNECT_SEC", "2.0"))

LOCK_HOST = "127.0.0.1"
LOCK_PORT = int(os.getenv("LAUNCHER_LOCK_PORT", "58741"))

MOCK_PLC = os.getenv("MOCK_PLC", "false").strip().lower() in ("1", "true", "yes", "on")
HEADLESS = os.getenv("HEADLESS", "0")

client: Any = None
main_proc: subprocess.Popen[Any] | None = None
launcher_lock: socket.socket | None = None

last_state: bool | None = None
last_debug_print_ts = 0.0


def log(msg: str) -> None:
    text = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(text, flush=True)

    try:
        with open(LAUNCHER_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


def acquire_single_instance_lock() -> bool:
    global launcher_lock

    try:
        launcher_lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        launcher_lock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        launcher_lock.bind((LOCK_HOST, LOCK_PORT))
        launcher_lock.listen(1)
        return True
    except OSError:
        log("[LAUNCHER] Another launcher is already running.")
        return False


def cleanup_client() -> None:
    global client

    try:
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass

            try:
                client.destroy()
            except Exception:
                pass
    finally:
        client = None


def ensure_client() -> bool:
    global client

    if MOCK_PLC:
        return True

    if client is not None:
        try:
            if client.get_connected():
                return True
        except Exception:
            pass

    cleanup_client()

    try:
        client = snap7.client.Client()
        log(
            "[LAUNCHER] Connecting PLC "
            f"ip={PLC_IP} rack={PLC_RACK} slot={PLC_SLOT} "
            f"start=DB{START_DB_NUMBER}.DBX{START_BYTE}.{START_BIT}"
        )
        client.connect(PLC_IP, PLC_RACK, PLC_SLOT)
        ok = bool(client.get_connected())
        log(f"[LAUNCHER] PLC connected={ok}")
        return ok
    except Exception as e:
        log(f"[LAUNCHER] PLC connect error: {repr(e)}")
        cleanup_client()
        stop_main()
        return False


def read_start_bit() -> bool:
    global last_debug_print_ts

    if MOCK_PLC:
        now = time.time()
        if now - last_debug_print_ts >= 5.0:
            log("[LAUNCHER] MOCK_PLC=true -> start_enable=True")
            last_debug_print_ts = now
        return True

    if client is None:
        raise RuntimeError("PLC client is None")

    data = client.db_read(START_DB_NUMBER, START_BYTE, 1)
    raw_value = int(data[0])
    start_enable = bool(get_bool(data, 0, START_BIT))

    now = time.time()
    if now - last_debug_print_ts >= 2.0:
        log(
            "[LAUNCHER] PLC READ "
            f"DB{START_DB_NUMBER}.DBB{START_BYTE}=0x{raw_value:02X} "
            f"bits={raw_value:08b} "
            f"DBX{START_BYTE}.{START_BIT}={start_enable}"
        )
        last_debug_print_ts = now

    return start_enable


def is_main_running() -> bool:
    global main_proc

    if main_proc is None:
        return False

    return main_proc.poll() is None


def start_main() -> None:
    global main_proc

    if is_main_running():
        return

    python_exe = PYTHON_EXE if os.path.exists(PYTHON_EXE) else "python"

    if not os.path.exists(MAIN_SCRIPT):
        log(f"[LAUNCHER] ERROR: main.py not found: {MAIN_SCRIPT}")
        return

    log("[LAUNCHER] Starting main.py")
    log(f"[LAUNCHER] MAIN_SCRIPT={MAIN_SCRIPT}")
    log(f"[LAUNCHER] PYTHON_EXE={python_exe}")
    log(f"[LAUNCHER] MAIN_LOG={MAIN_LOG}")

    env = os.environ.copy()
    env["HEADLESS"] = HEADLESS
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    kwargs: dict[str, Any] = {
        "cwd": BASE_DIR,
        "env": env,
        "stdout": open(MAIN_LOG, "a", encoding="utf-8"),
        "stderr": subprocess.STDOUT,
    }

    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        main_proc = subprocess.Popen(
            [python_exe, MAIN_SCRIPT],
            **kwargs,
        )
        log(f"[LAUNCHER] main.py pid={main_proc.pid}")
    except Exception as e:
        log(f"[LAUNCHER] Failed to start main.py: {repr(e)}")
        main_proc = None


def stop_main() -> None:
    global main_proc

    if not is_main_running():
        main_proc = None
        return

    log("[LAUNCHER] Stopping main.py")

    try:
        if main_proc is not None:
            if os.name == "nt":
                main_proc.terminate()
            else:
                main_proc.send_signal(signal.SIGTERM)

            main_proc.wait(timeout=5)
    except Exception as e:
        log(f"[LAUNCHER] Graceful stop failed: {repr(e)}")
        try:
            if main_proc is not None:
                main_proc.kill()
        except Exception:
            pass
    finally:
        main_proc = None


def main() -> None:
    global last_state

    log("[LAUNCHER] Started")
    log(f"[LAUNCHER] BASE_DIR={BASE_DIR}")
    log(f"[LAUNCHER] PLC_IP={PLC_IP}")
    log(f"[LAUNCHER] PLC_RACK={PLC_RACK}")
    log(f"[LAUNCHER] PLC_SLOT={PLC_SLOT}")
    log(f"[LAUNCHER] START_BIT=DB{START_DB_NUMBER}.DBX{START_BYTE}.{START_BIT}")
    log(f"[LAUNCHER] MOCK_PLC={MOCK_PLC}")

    consecutive_failures = 0

    while True:
        try:
            if not ensure_client():
                backoff = min(RECONNECT_SEC * (2 ** consecutive_failures), 60.0)
                consecutive_failures += 1
                log(f"[LAUNCHER] Reconnect backoff={backoff:.1f}s")
                time.sleep(backoff)
                continue

            consecutive_failures = 0
            start_enable = read_start_bit()

            if start_enable != last_state:
                log(f"[LAUNCHER] start_enable changed -> {start_enable}")
                last_state = start_enable

            # FIX CHÍNH:
            # Nếu bit đang ON thì đảm bảo main.py đang chạy.
            # Nếu main.py crash mà bit vẫn ON, launcher sẽ tự start lại.
            if start_enable:
                if not is_main_running():
                    start_main()
            else:
                stop_main()

        except Exception as e:
            log(f"[LAUNCHER] Loop error: {repr(e)}")
            stop_main()
            cleanup_client()
            time.sleep(RECONNECT_SEC)

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    if not acquire_single_instance_lock():
        raise SystemExit(0)

    try:
        main()
    except KeyboardInterrupt:
        log("[LAUNCHER] Exit by user")
    finally:
        stop_main()
        cleanup_client()