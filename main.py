import cv2
import numpy as np
from pipelines.vision_pipeline import VisionPipeline
from pipelines.driver_state_pipeline import DriverStatePipeline
from pipelines.crane_pipeline import CranePipeline
from policy.policy_engine import PolicyEngine
from alerts.voice_alert import VoiceAlert
import os
import time
from datetime import datetime
from pipelines.hands_pipeline import HandsPipeline
from utils.api_logger import APILogger

COLOR_GREEN = (0, 255, 0)
COLOR_RED   = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

vision = VisionPipeline("models/yolov8.pt")
driver_state = DriverStatePipeline()
crane_line = CranePipeline()
policy = PolicyEngine()
voice_alert = VoiceAlert(cooldown_sec=5)
hands_pipeline = HandsPipeline(no_hand_time=3.0)
api_logger = APILogger()

last_evidence_time = 0
EVIDENCE_COOLDOWN = 2.0 # seconds

IMPORTANT_OBJECTS = {"person", "cell phone"}

cap = cv2.VideoCapture(0)

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

    # UX: only show when eyes are "open enough"
    if ear > threshold:
        cv2.circle(frame, left_eye_center, 4, COLOR_WHITE, -1)
        cv2.circle(frame, right_eye_center, 4, COLOR_WHITE, -1)

def draw_overlay(frame, vision_out, driver_out, crane_out, alert):
    y = 30
    cv2.putText(
        frame, f"ALERT: {alert.name}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        COLOR_YELLOW if alert.name != "NONE" else COLOR_GREEN,
        2
    )

    # Show Crane Status
    y += 30
    is_lifting = crane_out.get("is_lifting", False)
    status_text = "LIFTING" if is_lifting else "FREE"
    status_color = COLOR_RED if is_lifting else COLOR_GREEN
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
    cv2.putText(
        frame, f"Phone: {vision_out.get('phone', False)}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        COLOR_WHITE, 1
    )

print("Edge AI started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    # ---- PIPELINES ----
    vision_out = vision.run(frame)
    driver_out = driver_state.run(frame)
    hands_out = hands_pipeline.run(frame)
    crane_out = crane_line.run()

    alert = policy.decide(vision_out, driver_out, hands_out, crane_out)
    voice_alert.speak(alert, driver_out, vision_out)

    # ---- Evidence Capture ----
    if alert.name != "NONE":
        now = time.time()
        if now - last_evidence_time > EVIDENCE_COOLDOWN:
            last_evidence_time = now
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/evidence/alert_{alert.name}_{timestamp}.jpg"
            # Draw overlay before saving? Or save raw? User asked for "evidence", usually implies what was seen.
            # But drawing overlay is done later. Let's save RAW frame or COPY of frame.
            # To show WHY, maybe save frame with some info?
            # For legal evidence, raw + metadata is best. But for simple review, overlay is good.
            # Let's write the alert level on the image locally before saving
            evidence_frame = frame.copy()
            cv2.putText(evidence_frame, f"EVIDENCE: {alert.name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imwrite(filename, evidence_frame)
            print(f"Evidence saved: {filename}")
            
            # Log to API
            api_logger.log_alert(
                alert_level=alert.name,
                crane_status=crane_out,
                driver_state=driver_out,
                image_path=filename
            )


    # ---- Draw YOLO boxes (only important objects) ----
    results = vision_out["yolo_result"]

    for box in results.boxes:
        cls = int(box.cls[0])
        label = vision.model.names[cls]
        if label not in IMPORTANT_OBJECTS:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        color = COLOR_GREEN

        if label == "cell phone":
            color = COLOR_RED

        if label == "person" and (driver_out.get("drowsy") or driver_out.get("distracted")):
            color = COLOR_RED

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"{label} {conf:.2f}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 2
        )

    # ---- Draw yaw + eye indicator ----
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

    # ---- Overlay ----
    draw_overlay(frame, vision_out, driver_out, crane_out, alert)

    cv2.imshow("Edge AI Safety Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Edge AI stopped.")
