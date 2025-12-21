import cv2
import numpy as np
from pipelines.vision_pipeline import VisionPipeline
from pipelines.driver_state_pipeline import DriverStatePipeline
from policy.policy_engine import PolicyEngine
from alerts.voice_alert import VoiceAlert
from pipelines.hands_pipeline import HandsPipeline

COLOR_GREEN = (0, 255, 0)
COLOR_RED   = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

vision = VisionPipeline("models/yolov8.pt")
driver_state = DriverStatePipeline()
policy = PolicyEngine()
voice_alert = VoiceAlert(cooldown_sec=5)
hands_pipeline = HandsPipeline(no_hand_time=3.0)

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

def draw_overlay(frame, vision_out, driver_out, alert):
    y = 30
    cv2.putText(
        frame, f"ALERT: {alert.name}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        COLOR_YELLOW if alert.name != "NONE" else COLOR_GREEN,
        2
    )

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

    alert = policy.decide(vision_out, driver_out, hands_out)
    voice_alert.speak(alert, driver_out, vision_out)


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
    draw_overlay(frame, vision_out, driver_out, alert)

    cv2.imshow("Edge AI Safety Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Edge AI stopped.")
