import os
import time
import signal
import socket
import subprocess
from dotenv import load_dotenv

try:
    import snap7
    from snap7.util import get_bool
except ImportError:
    print("Missing python-snap7. Please install it in .venv first.")
    raise

load_dotenv()

PLC_IP = os.getenv("PLC_IP", "192.168.150.103")
PLC_RACK = int(os.getenv("PLC_RACK", "0"))
PLC_SLOT = int(os.getenv("PLC_SLOT", "2"))

START_DB_NUMBER = 17
START_BYTE = 16
START_BIT = 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(BASE_DIR, "main.py")
PYTHON_EXE = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")

POLL_SEC = 0.5
RECONNECT_SEC = 2.0
LOCK_HOST = "127.0.0.1"
LOCK_PORT = 58741

client = None
main_proc = None
launcher_lock = None

last_state = None
launch_armed = True


def acquire_single_instance_lock() -> bool:
    global launcher_lock

    try:
        launcher_lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        launcher_lock.bind((LOCK_HOST, LOCK_PORT))
        launcher_lock.listen(1)
        return True
    except OSError:
        print("[LAUNCHER] Another launcher is already running.")
        return False


def cleanup_client():
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

    if client is not None:
        try:
            if client.get_connected():
                return True
        except Exception:
            pass

    cleanup_client()

    try:
        client = snap7.client.Client()
        print(f"[LAUNCHER] Connecting PLC ip={PLC_IP} rack={PLC_RACK} slot={PLC_SLOT}")
        client.connect(PLC_IP, PLC_RACK, PLC_SLOT)
        ok = client.get_connected()
        print(f"[LAUNCHER] PLC connected={ok}")
        return ok
    except Exception as e:
        print(f"[LAUNCHER] PLC connect error: {e}")
        cleanup_client()
        stop_main()
        return False


def read_start_bit() -> bool:
    data = client.db_read(START_DB_NUMBER, START_BYTE, 1)
    return get_bool(data, 0, START_BIT)


def is_main_running() -> bool:
    global main_proc
    return main_proc is not None and main_proc.poll() is None


def start_main():
    global main_proc

    if is_main_running():
        return

    python_exe = PYTHON_EXE if os.path.exists(PYTHON_EXE) else "python"

    print("[LAUNCHER] Starting main.py")
    print(f"[LAUNCHER] MAIN_SCRIPT={MAIN_SCRIPT}")
    print(f"[LAUNCHER] PYTHON_EXE={python_exe}")

    kwargs = {"cwd": BASE_DIR}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    main_proc = subprocess.Popen(
        [python_exe, MAIN_SCRIPT],
        **kwargs
    )


def stop_main():
    global main_proc

    if not is_main_running():
        main_proc = None
        return

    print("[LAUNCHER] Stopping main.py")
    try:
        if os.name == "nt":
            main_proc.terminate()
        else:
            main_proc.send_signal(signal.SIGTERM)

        main_proc.wait(timeout=5)
    except Exception as e:
        print(f"[LAUNCHER] Graceful stop failed: {e}")
        try:
            main_proc.kill()
        except Exception:
            pass
    finally:
        main_proc = None


def main():
    global last_state, launch_armed

    print("[LAUNCHER] Started")

    consecutive_failures = 0

    while True:
        try:
            if not ensure_client():
                backoff = min(RECONNECT_SEC * (2 ** consecutive_failures), 60.0)
                consecutive_failures += 1
                time.sleep(backoff)
                continue

            consecutive_failures = 0
            start_enable = read_start_bit()

            if start_enable != last_state:
                print(f"[LAUNCHER] start_enable changed -> {start_enable}")
                last_state = start_enable

            # Chỉ start 1 lần khi bit vừa chuyển từ 0 -> 1
            if start_enable and launch_armed:
                start_main()
                launch_armed = False

            # Khi bit về 0 thì stop main và arm lại cho lần sau
            elif not start_enable:
                stop_main()
                launch_armed = True

        except Exception as e:
            print(f"[LAUNCHER] Loop error: {e}")
            stop_main()
            cleanup_client()

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    if not acquire_single_instance_lock():
        raise SystemExit(0)

    try:
        main()
    except KeyboardInterrupt:
        print("[LAUNCHER] Exit by user")
    finally:
        stop_main()
        cleanup_client()