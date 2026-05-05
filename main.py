# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

import os
import time
from datetime import datetime
import threading
import queue
import copy
import math
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2  # type: ignore[import]
import numpy as np  # type: ignore[import]

from pipelines.vision_pipeline import VisionPipeline
from pipelines.driver_state_pipeline import DriverStatePipeline
from pipelines.crane_pipeline import CranePipeline
from pipelines.hands_pipeline import HandsPipeline
from pipelines.phone_usage_pipeline import PhoneUsagePipeline
from pipelines.phone_context_pipeline import PhoneContextPipeline
from policy.policy_engine import PolicyEngine
from alerts.voice_alert import VoiceAlert
from utils.api_logger import APILogger
from utils.box_utils import safe_box, box_center, box_area, point_distance, expand_box, point_in_box
from utils.threading_utils import LatestResult
from utils.types import DEFAULT_HANDS_OUT, DEFAULT_VISION_OUT, DEFAULT_DRIVER_OUT, HandsOut, VisionOut, DriverStateOut, PhoneContextOut, PhoneUsageOut


def load_env_file(path: str):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        print(f"[ENV] load failed: {path} -> {e}")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_env_file(os.path.join(BASE_DIR, ".env"))


COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)

HEADLESS = os.environ.get("HEADLESS", "0") == "1"
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# ===== DEBUG / PERF SWITCHES =====
SHOW_DEBUG_BOXES = os.getenv("SHOW_DEBUG_BOXES", "0") == "1"
SHOW_DEBUG_POINTS = os.getenv("SHOW_DEBUG_POINTS", "0") == "1"
ENABLE_DEBUG_LOGS = os.getenv("ENABLE_DEBUG_LOGS", "1") == "1"
ENABLE_EVIDENCE_SAVE = os.getenv("ENABLE_EVIDENCE_SAVE", "1") == "1"

# ===== INIT PIPELINES =====
vision = VisionPipeline()
driver_state = DriverStatePipeline()

crane_line = CranePipeline()
hands_pipeline = HandsPipeline(no_hand_time=8.0)
phone_usage_pipeline = PhoneUsagePipeline()
phone_context_pipeline = PhoneContextPipeline()
policy = PolicyEngine()
voice_alert = VoiceAlert(cooldown_sec=5)
api_logger = APILogger()

# ===== EVIDENCE CONTROL =====
EVIDENCE_WINDOW_SEC = 90
EVIDENCE_MAX_PER_WINDOW = 1
evi_window_start = {}
evi_window_count = {}
os.makedirs("output/evidence", exist_ok=True)

# ===== PIPELINE WORKER STOP SIGNAL =====
stop_event = threading.Event()

# ===== PIPELINE RESULT STORES (thread-safe) =====
hands_result: LatestResult[HandsOut] = LatestResult(cast(HandsOut, dict(DEFAULT_HANDS_OUT)))  # type: ignore
vision_result: LatestResult[VisionOut] = LatestResult(cast(VisionOut, dict(DEFAULT_VISION_OUT)))  # type: ignore
driver_result: LatestResult[DriverStateOut] = LatestResult(cast(DriverStateOut, dict(DEFAULT_DRIVER_OUT)))  # type: ignore

# ===== CAMERA STATE =====
cap: Any = None
camera_thread: Optional[threading.Thread] = None
camera_stop_event = threading.Event()
camera_lock = threading.Lock()
latest_frame: Optional[Any] = None
latest_frame_time = 0.0
camera_opened = False

# ===== PLC SHARED STATE =====
latest_crane_out = {}
crane_lock = threading.Lock()

# ===== AUDIO STATE =====
ai_audio_q: queue.Queue[Any] = queue.Queue()
plc_audio_q: queue.Queue[Any] = queue.Queue()
pending_plc: List[str] = []
pending_plc_lock = threading.Lock()

last_ai_audio = None
last_ai_audio_time = 0.0

last_plc_audio = None
last_plc_audio_time = 0.0

AI_REPEAT_SEC = 8.0
PLC_REPEAT_SEC = 8.0

# ===== DEBUG RATE LIMIT =====
last_dets_log_time = 0.0
last_drv_log_time = 0.0
last_dbg_log_time = 0.0
DEBUG_LOG_EVERY_SEC = 1.5

# ===== MAIN LOOP TUNING =====
FRAME_STALE_SEC = 0.6
MAIN_IDLE_SLEEP_SEC = 0.005

# ===== MULTI-RATE TUNING =====
HANDS_RUN_INTERVAL_SEC = 0.10
DRIVER_RUN_INTERVAL_SEC = 0.12
PHONE_RUN_INTERVAL_SEC = 0.10

# ===== CACHED RESULTS (phone context/usage stay in main loop) =====
cached_context_out: PhoneContextOut = {
    "phone_contexts": [],
    "lap_zone_box": None,
    "mid_leg_zone_box": None,
    "left_leg_zone_box": None,
    "right_leg_zone_box": None,
    "lap_attention": False,
    "left_side_attention": False,
    "right_side_attention": False,
    "lap_attention_strength": 0.0,
    "left_side_attention_strength": 0.0,
    "right_side_attention_strength": 0.0,
    "head_down": False,
    "body_box": None,
    "face_center": None,
}
cached_phone_usage_out: PhoneUsageOut = {
    "phone_using": False,
    "best_phone_box": None,
    "score": -999,
    "source": None,
    "temporal_track_len": 0,
}

last_phone_run_time = 0.0


def open_camera_device():
    global cap, camera_opened

    print(f"[SYS] open_camera_device called, CAMERA_INDEX={CAMERA_INDEX}")

    if cap is not None:
        return True

    # Try multiple camera indices if the default fails
    indices_to_try = [CAMERA_INDEX]
    if CAMERA_INDEX != 0:
        indices_to_try.insert(0, 0)  # Try 0 first if not already
    indices_to_try.extend([i for i in range(5) if i not in indices_to_try])

    for idx in indices_to_try:
        print(f"[SYS] Trying camera index {idx}")
        cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cam is None or not cam.isOpened():
            print(f"[SYS] Camera open failed, index={idx}")
            try:
                if cam is not None:
                    cam.release()
            except Exception:
                pass
            continue

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        cap = cam
        camera_opened = True
        print(f"[SYS] Camera opened, index={idx}")
        return True

    print("[SYS] All camera indices failed")
    cap = None
    camera_opened = False
    return False


def close_camera_device():
    global cap, camera_opened

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
        cap = None

    camera_opened = False
    print("[SYS] Camera device closed")


def camera_reader_worker():
    global latest_frame, latest_frame_time

    while not camera_stop_event.is_set():
        try:
            if cap is None:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            with camera_lock:
                latest_frame = frame
                latest_frame_time = time.time()

        except Exception as e:
            print("[CAMERA WORKER ERROR]", e)
            time.sleep(0.05)


def start_camera_system() -> bool:
    global camera_thread, latest_frame, latest_frame_time

    if not open_camera_device():
        return False

    with camera_lock:
        latest_frame = None
        latest_frame_time = 0.0

    camera_stop_event.clear()

    camera_thread = threading.Thread(target=camera_reader_worker, daemon=True)
    camera_thread.start()
    print("[SYS] Camera reader thread started")
    return True


def stop_camera_system():
    global camera_thread, latest_frame, latest_frame_time

    camera_stop_event.set()

    if camera_thread is not None:
        try:
            camera_thread.join(timeout=1.0)
        except Exception:
            pass
        camera_thread = None

    with camera_lock:
        latest_frame = None
        latest_frame_time = 0.0

    close_camera_device()
    print("[SYS] Camera system stopped")


def get_latest_frame():
    with camera_lock:
        if latest_frame is None:
            return None, 0.0
        return latest_frame.copy(), latest_frame_time


def hands_worker_fn():
    while not stop_event.is_set():
        frame, _ = get_latest_frame()
        if frame is None:
            time.sleep(0.010)
            continue
        out = hands_pipeline.run(frame)
        hands_result.put(out)
        time.sleep(HANDS_RUN_INTERVAL_SEC * 0.80)


def vision_worker_fn():
    while not stop_event.is_set():
        frame, _ = get_latest_frame()
        if frame is None:
            time.sleep(0.010)
            continue
        hands = hands_result.get()
        out = vision.run(frame, hand_centers=hands.get("hand_centers", []))
        vision_result.put(out)
        # VisionPipeline self-throttles with CALL_EVERY_SEC; sleep avoids spin-waiting
        time.sleep(0.010)


def driver_state_worker_fn():
    while not stop_event.is_set():
        frame, _ = get_latest_frame()
        if frame is None:
            time.sleep(0.010)
            continue
        hands = hands_result.get()
        vis = vision_result.get()

        person_box = None
        if vis.get("persons"):
            person_box = vis.get("persons", [{}])[0].get("xyxy")  # type: ignore

        out = driver_state.run(
            frame,
            person_box=person_box,
            left_shoulder=hands.get("left_shoulder"),
            right_shoulder=hands.get("right_shoulder"),
            left_hip=hands.get("left_hip"),
            right_hip=hands.get("right_hip"),
        )
        driver_result.put(out)
        time.sleep(DRIVER_RUN_INTERVAL_SEC * 0.80)


def make_driver_body_box(person_box: Optional[List[float]]) -> Optional[List[float]]:
    """
    Nới vùng người lái rộng hơn:
    - rộng hơn sang hai bên để không bó upper-body
    - kéo xuống dưới để giữ tay/lap tốt hơn
    """
    b = safe_box(person_box)
    if b is None:
        return None

    x1, y1, x2, y2 = b
    w = x2 - x1
    h = y2 - y1

    bx1 = x1 - w * 0.22
    bx2 = x2 + w * 0.22
    by1 = y1 - h * 0.10
    by2 = y2 + h * 0.75

    return [bx1, by1, bx2, by2]


def filter_phone_candidates_for_usage(
    vision_out: VisionOut,
    hands_out: HandsOut,
    driver_out: DriverStateOut,
    driver_body_box: Optional[List[float]],
    frame_shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """
    Giảm nhầm chuột/đồ vật linh tinh thành phone bằng ngữ cảnh:
    - ưu tiên box gần tay
    - ưu tiên box gần mặt
    - ưu tiên box nằm trong driver body box đã nới
    - loại bớt box nhỏ, xa tay, xa mặt, nằm ngoài ngữ cảnh lái
    """
    phones = vision_out.get("phones", [])
    if not phones:
        return []

    fh, fw = frame_shape[:2]
    frame_diag = math.hypot(fw, fh)

    fh, fw = frame_shape[:2]
    frame_diag = math.hypot(fw, fh)

    face_center = driver_out.get("face_center")
    hand_points = []
    for key in ["left_wrist", "right_wrist"]:
        pt = hands_out.get(key)
        if pt is not None:
            hand_points.append(pt)

    for hc in hands_out.get("hand_centers", []):
        if hc is not None:
            hand_points.append(hc)

    body_box_big = expand_box(driver_body_box, sx=0.10, sy=0.08)

    filtered = []
    for ph in phones:
        box = ph.get("xyxy")
        conf = float(ph.get("conf", 0.0))
        center = box_center(box)
        area = box_area(box)
        area_ratio = area / float(max(1, fw * fh))

        score = 0

        if conf >= 0.70:
            score += 3
        elif conf >= 0.55:
            score += 2
        elif conf >= 0.45:
            score += 1

        inside_body = point_in_box(center, body_box_big)
        if inside_body:
            score += 2
        else:
            score -= 2

        near_hand = False
        for hp in hand_points:
            if point_distance(center, hp) <= frame_diag * 0.16:
                near_hand = True
                break
        if near_hand:
            score += 4

        near_face = point_distance(center, face_center) <= frame_diag * 0.14
        if near_face:
            score += 2

        if area_ratio < 0.00020:
            score -= 2

        if (not near_hand) and (not near_face) and (not inside_body):
            score -= 3

        if score >= 2:
            filtered.append({
                "xyxy": box,
                "conf": conf,
            })

    if not filtered:
        return []

    return [p["xyxy"] for p in filtered if p.get("xyxy") is not None]


def is_box_kept(box: Optional[List[float]], kept_boxes: List[Any], tol: float = 10.0) -> bool:
    c1 = box_center(box)
    if c1 is None:
        return False
    for kb in kept_boxes:
        c2 = box_center(kb)
        if c2 is None:
            continue
        if point_distance(c1, c2) <= tol:
            return True
    return False


def draw_yaw_vector(frame: Any, nose_point: Optional[Tuple[float, float]], yaw_deg: Optional[float], threshold: float = 15.0) -> None:
    if nose_point is None or yaw_deg is None:
        return

    length = 80
    angle_rad = -yaw_deg * np.pi / 180.0
    end_x = int(nose_point[0] + length * np.sin(angle_rad))
    end_y = int(nose_point[1])

    color = COLOR_GREEN if abs(yaw_deg) < threshold else COLOR_RED
    cv2.arrowedLine(frame, (int(nose_point[0]), int(nose_point[1])), (end_x, end_y), color, 2, tipLength=0.3)


def draw_eye_indicator(frame: Any, left_eye_center: Optional[Tuple[float, float]], right_eye_center: Optional[Tuple[float, float]], ear: Optional[float], threshold: float = 0.25) -> None:
    if ear is None or left_eye_center is None or right_eye_center is None:
        return

    if ear > threshold:
        cv2.circle(frame, (int(left_eye_center[0]), int(left_eye_center[1])), 4, COLOR_WHITE, -1)
        cv2.circle(frame, (int(right_eye_center[0]), int(right_eye_center[1])), 4, COLOR_WHITE, -1)


def draw_overlay(frame: Any, vision_out: VisionOut, driver_out: DriverStateOut, crane_out: Dict[str, Any], alert: Any, phone_usage_out: PhoneUsageOut) -> None:
    y = 30
    cv2.putText(
        frame, f"ALERT: {alert}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        COLOR_YELLOW if alert != "NONE" else COLOR_GREEN,
        2
    )

    y += 30
    has_plc_signal = bool(crane_out)
    status_text = "PLC ACTIVE" if has_plc_signal else "FREE"
    status_color = COLOR_RED if has_plc_signal else COLOR_GREEN
    cv2.putText(frame, f"CRANE: {status_text}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    y += 30
    if driver_out.get("yaw") is not None:
        cv2.putText(
            frame, f"Yaw: {driver_out.get('yaw')} deg",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            COLOR_WHITE, 1
        )

    y += 25
    if driver_out.get("ear") is not None:
        cv2.putText(
            frame, f"EAR: {driver_out.get('ear')}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            COLOR_WHITE, 1
        )

    y += 25
    cv2.putText(
        frame,
        f"PHONE_USE: {phone_usage_out.get('phone_using', False)} score={phone_usage_out.get('score', -1)}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        COLOR_CYAN if phone_usage_out.get("phone_using") else COLOR_WHITE,
        1
    )

    y += 25
    cv2.putText(
        frame,
        f"DROWSY: {driver_out.get('drowsy', False)}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        COLOR_CYAN if driver_out.get("drowsy") else COLOR_WHITE,
        1
    )

    y += 25
    cv2.putText(
        frame,
        f"YAWNING: {driver_out.get('yawning', False)}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        COLOR_CYAN if driver_out.get("yawning", False) else COLOR_WHITE,
        1
    )

    if driver_out.get("face_ok") is not None:
        y += 25
        cv2.putText(
            frame,
            f"FACE_OK: {driver_out.get('face_ok', False)}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            COLOR_CYAN if driver_out.get("face_ok", False) else COLOR_WHITE,
            1
        )


def plc_worker():
    global latest_crane_out

    while True:
        try:
            data = crane_line.run()
        except Exception as e:
            data = {"signals": {}, "error": str(e)}

        with crane_lock:
            latest_crane_out = copy.deepcopy(data)

        if "error" in data:
            time.sleep(0.5)
        else:
            time.sleep(0.2)


def audio_worker():
    global pending_plc

    while True:
        try:
            try:
                item = ai_audio_q.get_nowait()
            except queue.Empty:
                item = None

            if item is not None:
                print("AUDIO WORKER GOT AI:", item["alert"].name)

                try:
                    if hasattr(crane_line, "plc_channel"):
                        if hasattr(crane_line, "plc_channel") and crane_line.plc_channel is not None:
                            crane_line.plc_channel.stop()
                except Exception:
                    pass

                try:
                    voice_alert.stop()
                except Exception:
                    pass

                voice_alert.speak(
                    item["alert"],
                    item["driver_out"],
                    item["vision_out"]
                )
                continue

            sig = None
            with pending_plc_lock:
                if pending_plc:
                    sig = pending_plc.pop(0)

            if sig is not None:
                crane_line.play_signal(sig)
                continue

            try:
                item = plc_audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item.get("kind") == "PLC":
                new_items = crane_line.flatten_signals(item["crane_out"])
                with pending_plc_lock:
                    pending_plc.clear()
                    pending_plc.extend(new_items)

        except Exception as e:
            print("[AUDIO WORKER ERROR]", e)
            time.sleep(0.1)


def should_save_evidence(alert_name: str, now: float) -> bool:
    ws = evi_window_start.get(alert_name)
    if ws is None or (now - ws) >= EVIDENCE_WINDOW_SEC:
        evi_window_start[alert_name] = now
        evi_window_count[alert_name] = 0

    cnt = evi_window_count.get(alert_name, 0)
    if cnt < EVIDENCE_MAX_PER_WINDOW:
        evi_window_count[alert_name] = cnt + 1
        return True

    return False


def log_debug_once_per_sec(vision_out: VisionOut, driver_out: DriverStateOut, hands_out: HandsOut, crane_out: Dict[str, Any], alert: Any, phone_usage_out: PhoneUsageOut) -> None:
    global last_dbg_log_time

    if not ENABLE_DEBUG_LOGS:
        return

    now = time.time()
    if now - last_dbg_log_time < DEBUG_LOG_EVERY_SEC:
        return

    last_dbg_log_time = now

    try:
        print(
            "DBG",
            "alert=", alert,
            "phone_raw=", len(vision_out.get("phones", [])),
            "phone_using=", phone_usage_out.get("phone_using"),
            "phone_score=", phone_usage_out.get("score"),
            "drowsy=", driver_out.get("drowsy"),
            "yawning=", driver_out.get("yawning"),
            "head_down=", driver_out.get("head_down"),
            "forward_working=", driver_out.get("forward_working"),
            "face_ok=", driver_out.get("face_ok"),
            "no_hand=", hands_out.get("no_hand"),
        )
    except Exception:
        pass


def log_vision_once_per_sec(vision_out: VisionOut) -> None:
    global last_dets_log_time

    if not ENABLE_DEBUG_LOGS:
        return

    now = time.time()
    if now - last_dets_log_time < DEBUG_LOG_EVERY_SEC:
        return

    last_dets_log_time = now
    print(
        "VISION",
        "dets=", len(vision_out.get("dets", [])),
        "persons=", len(vision_out.get("persons", [])),
        "phones=", len(vision_out.get("phones", []))
    )


def log_driver_once_per_sec(driver_out: DriverStateOut) -> None:
    global last_drv_log_time

    if not ENABLE_DEBUG_LOGS:
        return

    now = time.time()
    if now - last_drv_log_time < DEBUG_LOG_EVERY_SEC:
        return

    last_drv_log_time = now
    print(
        "DRV",
        "ear=", driver_out.get("ear"),
        "drowsy=", driver_out.get("drowsy"),
        "yawning=", driver_out.get("yawning"),
        "mar=", driver_out.get("mar"),
        "head_down=", driver_out.get("head_down"),
        "face_ok=", driver_out.get("face_ok")
    )


print("=== MAIN STARTED BY LAUNCHER ===")
print("Edge AI started. Press 'q' to quit." if not HEADLESS else "Edge AI started (HEADLESS=1).")

threading.Thread(target=plc_worker, daemon=True).start()
threading.Thread(target=audio_worker, daemon=True).start()

if not start_camera_system():
    print("[SYS] Cannot start camera, exiting main.")
    raise SystemExit(1)

_pipeline_threads = [
    threading.Thread(target=hands_worker_fn, daemon=True, name="hands_worker"),
    threading.Thread(target=vision_worker_fn, daemon=True, name="vision_worker"),
    threading.Thread(target=driver_state_worker_fn, daemon=True, name="driver_worker"),
]
for _t in _pipeline_threads:
    _t.start()
print("[SYS] Pipeline worker threads started")

try:
    while True:
        with crane_lock:
            crane_raw = copy.deepcopy(latest_crane_out)

        crane_out = crane_raw.get("signals", {})

        frame, frame_ts = get_latest_frame()
        if frame is None:
            time.sleep(MAIN_IDLE_SLEEP_SEC)
            continue

        frame_age = time.time() - frame_ts if frame_ts > 0 else 999.0
        if frame_age > FRAME_STALE_SEC:
            time.sleep(MAIN_IDLE_SLEEP_SEC)
            continue

        now = time.time()

        # Read latest results from pipeline workers
        hands_out = hands_result.get()
        vision_out = vision_result.get()
        driver_out = driver_result.get()

        log_vision_once_per_sec(vision_out)
        log_driver_once_per_sec(driver_out)

        person_box = None
        if vision_out.get("persons"):
            person_box = vision_out.get("persons", [{}])[0].get("xyxy")  # type: ignore

        driver_body_box = make_driver_body_box(person_box)

        filtered_phone_candidates = filter_phone_candidates_for_usage(
            vision_out=vision_out,
            hands_out=hands_out,
            driver_out=driver_out,
            driver_body_box=driver_body_box,
            frame_shape=frame.shape,
        )

        hand_centers = hands_out.get("hand_centers", [])
        face_center = driver_out.get("face_center")

        if (now - last_phone_run_time) >= PHONE_RUN_INTERVAL_SEC:
            cached_context_out = phone_context_pipeline.run(
                frame=frame,
                person_box=driver_body_box,
                phone_candidates=filtered_phone_candidates,
                hand_centers=hand_centers,
                face_center=face_center,
                head_down=driver_out.get("head_down", False)
            )

            cached_phone_usage_out = phone_usage_pipeline.run(
                person_box=driver_body_box,
                phone_candidates=filtered_phone_candidates,
                hand_centers=hand_centers,
                face_center=face_center,
                context_out=cached_context_out,
                left_wrist=hands_out.get("left_wrist"),
                right_wrist=hands_out.get("right_wrist"),
                left_elbow=hands_out.get("left_elbow"),
                right_elbow=hands_out.get("right_elbow"),
                left_shoulder=hands_out.get("left_shoulder"),
                right_shoulder=hands_out.get("right_shoulder"),
                left_thigh_center=hands_out.get("left_thigh_center"),
                right_thigh_center=hands_out.get("right_thigh_center"),
            )
            last_phone_run_time = now

        context_out = cached_context_out
        phone_usage_out = cached_phone_usage_out

        alert = policy.decide(vision_out, driver_out, hands_out, crane_out, phone_usage_out)
        log_debug_once_per_sec(vision_out, driver_out, hands_out, crane_out, alert, phone_usage_out)

        if alert.name != "NONE":
            repeat_sec = AI_REPEAT_SEC
            if alert.name == "HIGH":
                repeat_sec = 5.0   # repeat drowsy/yawning alerts every 5s
            elif alert.name == "MEDIUM":
                repeat_sec = 30.0

            if (alert.name != last_ai_audio) or ((now - last_ai_audio_time) >= repeat_sec):
                print("AI ENQUEUE:", alert.name, "phone_using=", phone_usage_out.get("phone_using"))
                ai_audio_q.put({
                    "kind": "AI",
                    "alert": alert,
                    "driver_out": driver_out,
                    "vision_out": {
                        **vision_out,
                        "phone": phone_usage_out.get("phone_using", False)
                    }
                })
                last_ai_audio = alert.name
                last_ai_audio_time = now

        if isinstance(crane_out, dict) and "error" not in crane_out and crane_out:
            plc_key = ",".join(sorted(crane_out.keys()))
            with pending_plc_lock:
                no_pending_plc = not pending_plc

            if no_pending_plc and plc_audio_q.empty():
                if (plc_key != last_plc_audio) or ((now - last_plc_audio_time) >= PLC_REPEAT_SEC):
                    plc_audio_q.put({
                        "kind": "PLC",
                        "crane_out": crane_out
                    })
                    last_plc_audio = plc_key
                    last_plc_audio_time = now

        if ENABLE_EVIDENCE_SAVE and alert.name != "NONE":
            if should_save_evidence(alert.name, now):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"output/evidence/alert_{alert.name}_{timestamp}.jpg"

                evidence_frame = frame.copy()
                cv2.putText(
                    evidence_frame, f"EVIDENCE: {alert.name}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2
                )

                ok_write = cv2.imwrite(filename, evidence_frame)
                print("Evidence saved:", filename, "ok=", ok_write)

                try:
                    api_logger.log_alert(
                        alert_level=alert.name,
                        crane_status=crane_out,
                        driver_state={
                            **driver_out,
                            "phone_using_score": phone_usage_out.get("score"),
                            "phone_using": phone_usage_out.get("phone_using")
                        },
                        image_path=filename
                    )
                except Exception as e:
                    print("[API LOGGER ERROR]", e)

        for det in vision_out.get("dets", []):
            cls = int(det.get("cls", -1))
            conf = float(det.get("conf", 0.0))
            box = det.get("xyxy", [0, 0, 0, 0])
            x1, y1, x2, y2 = map(int, box)

            if cls == 0:
                label = "person"
            elif cls == 67:
                label = "cell phone"
                if not is_box_kept(box, filtered_phone_candidates, tol=14.0):
                    continue
            else:
                continue

            color = COLOR_GREEN
            if label == "cell phone":
                color = COLOR_RED
            if label == "person" and (driver_out.get("drowsy") or driver_out.get("distracted")):
                color = COLOR_RED

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2
            )

        if SHOW_DEBUG_BOXES and driver_body_box is not None:
            bx1, by1, bx2, by2 = map(int, driver_body_box)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_CYAN, 1)

        if SHOW_DEBUG_BOXES:
            for key in ["lap_zone_box", "mid_leg_zone_box", "left_leg_zone_box", "right_leg_zone_box"]:
                box = context_out.get(key)
                if box is not None:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, 1)

        best_phone_box = phone_usage_out.get("best_phone_box")
        if phone_usage_out.get("phone_using") and best_phone_box:
            x1, y1, x2, y2 = map(int, best_phone_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_YELLOW, 2)
            cv2.putText(
                frame,
                f"PHONE_USE {phone_usage_out.get('score', -1)} {phone_usage_out.get('source')}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_YELLOW,
                2
            )

        if SHOW_DEBUG_BOXES:
            face_box = driver_out.get("face_box")
            if face_box is not None:
                x1, y1, x2, y2 = map(int, face_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)
                cv2.putText(
                    frame,
                    "face_v2",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLOR_GREEN,
                    2
                )

        if SHOW_DEBUG_POINTS:
            for hc in hands_out.get("hand_centers", []):
                cv2.circle(frame, hc, 6, COLOR_CYAN, -1)

            for tp in [hands_out.get("left_thigh_center"), hands_out.get("right_thigh_center")]:
                if tp is not None:
                    cv2.circle(frame, tp, 6, COLOR_GREEN, -1)

            draw_yaw_vector(
                frame,
                driver_out.get("nose_point"),
                driver_out.get("yaw"),
                threshold=15.0
            )

            draw_eye_indicator(
                frame,
                driver_out.get("left_eye_center"),
                driver_out.get("right_eye_center"),
                driver_out.get("ear"),
                threshold=0.25
            )

        draw_overlay(frame, vision_out, driver_out, crane_out, alert, phone_usage_out)

        if not HEADLESS:
            cv2.imshow("Edge AI Safety Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            time.sleep(0.001)

finally:
    stop_event.set()
    for _t in _pipeline_threads:
        try:
            _t.join(timeout=3.0)
        except Exception:
            pass

    try:
        voice_alert.close()
    except Exception:
        pass

    try:
        hands_pipeline.close()
    except Exception:
        pass

    try:
        driver_state.close()
    except Exception:
        pass

    try:
        crane_line.close()
    except Exception:
        pass

    try:
        phone_usage_pipeline.reset()
    except Exception:
        pass

    try:
        vision.close()
    except Exception:
        pass

    try:
        import pygame
        pygame.mixer.stop()
        pygame.mixer.quit()
    except Exception:
        pass

    stop_camera_system()

    if not HEADLESS:
        cv2.destroyAllWindows()

    print("Edge AI stopped.")