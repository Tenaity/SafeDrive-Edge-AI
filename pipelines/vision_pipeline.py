import os
import time
import cv2
import requests


class VisionPipeline:
    """
    VisionPipeline (HTTP):
    - Send frame to YOLO service via YOLO_URL
    - Filter by confidence
    - Expose safety-relevant objects (person, cell phone)
    - Smooth phone detection with a short hold window
    """

    def __init__(self):
        self.yolo_url = os.environ.get("YOLO_URL", "http://127.0.0.1:8000/detect")

        # Confidence thresholds (tune later)
        self.CONF_PERSON = 0.40
        self.CONF_PHONE = 0.4

        # COCO class ids
        self.CLS_PERSON = 0
        self.CLS_PHONE = 67

        # Network timeout (YOLO service on Jetson can be slow)
        self.TIMEOUT_SEC = 2.0

        # Phone smoothing (hold detection for a few frames)
        self.phone_hold = 0
        self.PHONE_HOLD_FRAMES = 6

        # Throttle YOLO calls to reduce lag (Jetson Nano)
        self.last_call = 0.0
        self.CALL_EVERY_SEC = 0.20  # ~5 FPS YOLO
        self.last_out = {"phone": False, "dets": []}

    def run(self, frame):
        now = time.time()
        if now - self.last_call < self.CALL_EVERY_SEC:
            return self.last_out
        self.last_call = now
        # Encode JPEG to send
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return {"phone": False, "dets": []}

        try:
            r = requests.post(
                self.yolo_url,
                files={"image": ("frame.jpg", jpg.tobytes(), "image/jpeg")},
                timeout=self.TIMEOUT_SEC,
            )
            data = r.json()
        except Exception:
            # On any network/JSON error, return empty (caller may cache last dets if desired)
            return {"phone": False, "dets": []}

        if not data.get("ok"):
            return {"phone": False, "dets": []}

        dets = []
        phone_detected_now = False

        for det in data.get("dets", []):
            try:
                cls = int(det.get("cls", -1))
                conf = float(det.get("conf", 0.0))
                xyxy = det.get("xyxy", None)
                if not xyxy or len(xyxy) != 4:
                    continue
            except Exception:
                continue

            # Filter & keep only relevant objects
            if cls == self.CLS_PERSON and conf >= self.CONF_PERSON:
                dets.append(det)

            elif cls == self.CLS_PHONE and conf >= self.CONF_PHONE:
                phone_detected_now = True
                dets.append(det)

        # Smooth phone: hold it for a few frames to avoid flicker
        if phone_detected_now:
            self.phone_hold = self.PHONE_HOLD_FRAMES
        else:
            self.phone_hold = max(0, self.phone_hold - 1)

        phone_detected = (self.phone_hold > 0)

        return {
            "phone": phone_detected,
            "dets": dets,
        }