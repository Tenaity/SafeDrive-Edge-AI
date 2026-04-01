import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import time
from datetime import datetime
import threading
import queue
import copy

import cv2
import numpy as np

from pipelines.vision_pipeline import VisionPipeline
from pipelines.driver_state_pipeline import DriverStatePipeline
from pipelines.crane_pipeline import CranePipeline
from pipelines.hands_pipeline import HandsPipeline
from pipelines.phone_usage_pipeline import PhoneUsagePipeline
from pipelines.phone_context_pipeline import PhoneContextPipeline
from policy.policy_engine import PolicyEngine
from alerts.voice_alert import VoiceAlert
from utils.api_logger import APILogger


COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)

HEADLESS = os.environ.get("HEADLESS", "0") == "1"
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

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
EVIDENCE_WINDOW_SEC = 60
EVIDENCE_MAX_PER_WINDOW = 2
evi_window_start = {}
evi_window_count = {}

# ===== CAMERA STATE =====
cap = None

# ===== PLC SHARED STATE =====
latest_crane_out = {}
crane_lock = threading.Lock()

# ===== AUDIO STATE =====
ai_audio_q = queue.Queue()
plc_audio_q = queue.Queue()
pending_plc = []
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
DEBUG_LOG_EVERY_SEC = 1.0


def open_camera() -> bool:
    global cap

    print(f"[SYS] open_camera called, CAMERA_INDEX={CAMERA_INDEX}")

    if cap is not None:
        return True

    cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if cam is None or not cam.isOpened():
        print(f"[SYS] Camera open failed, index={CAMERA_INDEX}")
        try:
            if cam is not None:
                cam.release()
        except Exception:
            pass
        cap = None
        return False

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap = cam
    print(f"[SYS] Camera opened, index={CAMERA_INDEX}")
    return True


def close_camera():
    global cap
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
        cap = None
        print("[SYS] Camera closed")


def draw_yaw_vector(frame, nose_point, yaw_deg, threshold=15.0):
    if nose_point is None or yaw_deg is None:
        return

    length = 80
    angle_rad = -yaw_deg * np.pi / 180.0
    end_x = int(nose_point[0] + length * np.sin(angle_rad))
    end_y = int(nose_point[1])

    color = COLOR_GREEN if abs(yaw_deg) < threshold else COLOR_RED
    cv2.arrowedLine(frame, nose_point, (end_x, end_y), color, 2, tipLength=0.3)


def draw_eye_indicator(frame, left_eye_center, right_eye_center, ear, threshold=0.25):
    if ear is None or left_eye_center is None or right_eye_center is None:
        return

    if ear > threshold:
        cv2.circle(frame, left_eye_center, 4, COLOR_WHITE, -1)
        cv2.circle(frame, right_eye_center, 4, COLOR_WHITE, -1)


def draw_overlay(frame, vision_out, driver_out, crane_out, alert, phone_usage_out):
    y = 30
    cv2.putText(
        frame, f"ALERT: {alert.name}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        COLOR_YELLOW if alert.name != "NONE" else COLOR_GREEN,
        2
    )

    y += 30
    has_plc_signal = bool(crane_out) if isinstance(crane_out, dict) else False
    status_text = "PLC ACTIVE" if has_plc_signal else "FREE"
    status_color = COLOR_RED if has_plc_signal else COLOR_GREEN
    cv2.putText(frame, f"CRANE: {status_text}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    y += 30
    if driver_out.get("yaw") is not None:
        cv2.putText(
            frame, f"Yaw: {driver_out['yaw']} deg",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            COLOR_WHITE, 1
        )

    y += 25
    if driver_out.get("ear") is not None:
        cv2.putText(
            frame, f"EAR: {driver_out['ear']}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            COLOR_WHITE, 1
        )

    y += 25
    baseline_ear = driver_out.get("baseline_ear")
    if baseline_ear is not None:
        cv2.putText(
            frame, f"BASELINE: {baseline_ear}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            COLOR_WHITE, 1
        )
        
    y += 25
    cv2.putText(
        frame,
        f"HEAD_DOWN: {driver_out.get('head_down', False)}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        COLOR_WHITE, 1
    )

    y += 25
    cv2.putText(
        frame,
        f"MAR: {driver_out.get('mar')}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        COLOR_WHITE, 1
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
        frame, f"PHONE_RAW: {len(vision_out.get('phones', []))}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        COLOR_WHITE, 1
    )


def plc_worker():
    global latest_crane_out

    while True:
        try:
            data = crane_line.run() if crane_line is not None else {}
        except Exception as e:
            data = {"signals": {}, "error": str(e)}

        with crane_lock:
            latest_crane_out = copy.deepcopy(data)

        if isinstance(data, dict) and "error" in data:
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
                    if crane_line is not None and hasattr(crane_line, "plc_channel"):
                        crane_line.plc_channel.stop()
                except Exception:
                    pass

                try:
                    if voice_alert is not None:
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
                if crane_line is not None:
                    crane_line.play_signal(sig)
                continue

            try:
                item = plc_audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                continue

            if item.get("kind") == "PLC" and crane_line is not None:
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


def log_debug_once_per_sec(vision_out, driver_out, hands_out, crane_out, alert, phone_usage_out):
    global last_dbg_log_time
    now = time.time()

    if now - last_dbg_log_time < DEBUG_LOG_EVERY_SEC:
        return

    last_dbg_log_time = now

    try:
        print(
            "DBG",
            "alert=", alert.name,
            "phone_raw=", len(vision_out.get("phones", [])),
            "phone_using=", phone_usage_out.get("phone_using"),
            "phone_score=", phone_usage_out.get("score"),
            "drowsy=", driver_out.get("drowsy"),
            "distracted=", driver_out.get("distracted"),
            "no_hand=", hands_out.get("no_hand") if isinstance(hands_out, dict) else None,
            "hand_centers=", hands_out.get("hand_centers") if isinstance(hands_out, dict) else None,
            "crane_keys=", list(crane_out.keys()) if isinstance(crane_out, dict) else crane_out
        )
    except Exception:
        pass


def log_vision_once_per_sec(vision_out):
    global last_dets_log_time
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


def log_driver_once_per_sec(driver_out):
    global last_drv_log_time
    now = time.time()

    if now - last_drv_log_time < DEBUG_LOG_EVERY_SEC:
        return

    last_drv_log_time = now
    print(
        "DRV",
        "yaw=", driver_out.get("yaw"),
        "ear=", driver_out.get("ear"),
        "baseline=", driver_out.get("baseline_ear"),
        "th_on=", driver_out.get("ear_threshold_on"),
        "drowsy=", driver_out.get("drowsy"),
        "yawning=", driver_out.get("yawning"),
        "mar=", driver_out.get("mar"),
        "head_down=", driver_out.get("head_down"),
        "distracted=", driver_out.get("distracted"),
        "nose=", driver_out.get("nose_point"),
        "face_ok=", driver_out.get("face_ok")
    )

print("=== MAIN STARTED BY LAUNCHER ===")
print("Edge AI started. Press 'q' to quit." if not HEADLESS else "Edge AI started (HEADLESS=1).")

threading.Thread(target=plc_worker, daemon=True).start()
threading.Thread(target=audio_worker, daemon=True).start()

if not open_camera():
    print("[SYS] Cannot start camera, exiting main.")
    raise SystemExit(1)

try:
    while True:
        with crane_lock:
            crane_raw = copy.deepcopy(latest_crane_out)

        crane_out = {}
        if isinstance(crane_raw, dict):
            crane_out = crane_raw.get("signals", {})

        if cap is None:
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            close_camera()
            time.sleep(0.3)
            if not open_camera():
                time.sleep(0.5)
            continue

        vision_out = vision.run(frame)
        log_vision_once_per_sec(vision_out)

        driver_out = driver_state.run(frame)
        log_driver_once_per_sec(driver_out)

        hands_out = hands_pipeline.run(frame)

        person_box = None
        if vision_out.get("persons"):
            person_box = vision_out["persons"][0].get("xyxy")

        phone_candidates = [p.get("xyxy") for p in vision_out.get("phones", []) if p.get("xyxy")]
        hand_centers = hands_out.get("hand_centers", [])
        face_center = driver_out.get("face_center")

        context_out = phone_context_pipeline.run(
            frame=frame,
            person_box=person_box,
            phone_candidates=phone_candidates,
            hand_centers=hand_centers,
            face_center=face_center,
            head_down=driver_out.get("head_down", False)
        )

        phone_usage_out = phone_usage_pipeline.run(
            person_box=person_box,
            phone_candidates=phone_candidates,
            hand_centers=hand_centers,
            face_center=face_center,
            context_out=context_out,
            left_wrist=hands_out.get("left_wrist"),
            right_wrist=hands_out.get("right_wrist"),
            left_elbow=hands_out.get("left_elbow"),
            right_elbow=hands_out.get("right_elbow"),
            left_shoulder=hands_out.get("left_shoulder"),
            right_shoulder=hands_out.get("right_shoulder"),
            left_thigh_center=hands_out.get("left_thigh_center"),
            right_thigh_center=hands_out.get("right_thigh_center"),
        )
        
        alert = policy.decide(vision_out, driver_out, hands_out, crane_out, phone_usage_out)
        log_debug_once_per_sec(vision_out, driver_out, hands_out, crane_out, alert, phone_usage_out)

        now = time.time()

        # ===== AI AUDIO =====
        if alert.name != "NONE":
            repeat_sec = AI_REPEAT_SEC
            if alert.name == "MEDIUM":
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

        # ===== PLC AUDIO =====
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

        # ===== EVIDENCE =====
        if alert.name != "NONE":
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

                os.makedirs(os.path.dirname(filename), exist_ok=True)
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

        # ===== DRAW YOLO BOXES =====
        for det in vision_out.get("dets", []):
            cls = int(det.get("cls", -1))
            conf = float(det.get("conf", 0.0))
            x1, y1, x2, y2 = map(int, det.get("xyxy", [0, 0, 0, 0]))

            if cls == 0:
                label = "person"
            elif cls == 67:
                label = "cell phone"
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

        # draw best confirmed phone box from phone usage pipeline
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

        # draw hand centers
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
    try:
        if voice_alert is not None:
            voice_alert.close()
    except Exception:
        pass

    try:
        if hands_pipeline is not None:
            hands_pipeline.close()
    except Exception:
        pass

    try:
        if driver_state is not None:
            driver_state.close()
    except Exception:
        pass

    try:
        if crane_line is not None:
            crane_line.close()
    except Exception:
        pass

    try:
        if phone_usage_pipeline is not None:
            phone_usage_pipeline.reset()
    except Exception:
        pass

    try:
        if vision is not None:
            vision.close()
    except Exception:
        pass

    try:
        import pygame
        pygame.mixer.stop()
    except Exception:
        pass

    close_camera()

    if not HEADLESS:
        cv2.destroyAllWindows()

    print("Edge AI stopped.")