import os
import time
import cv2
import requests


class VisionPipeline:
    """
    Fast VisionPipeline:
    - Send smaller JPEG to YOLO server
    - Lower request rate to reduce lag
    - Keep structured person/phone candidates
    - Reuse HTTP session for lower overhead
    - Avoid stale output holding forever on repeated failures
    """

    def __init__(self):
        self.yolo_url = os.environ.get("YOLO_URL", "http://127.0.0.1:8000/detect")
        print(f"[YOLO URL] {self.yolo_url}")

        # timeout vừa phải để không treo pipeline
        self.TIMEOUT_SEC = 1.5

        # giảm tần suất gọi để đỡ lag
        self.last_call = 0.0
        self.CALL_EVERY_SEC = 0.30

        self.last_out = {
            "phone": False,
            "dets": [],
            "persons": [],
            "phones": []
        }

        # resize trước khi gửi API
        self.SEND_LONG_SIDE = 960
        self.JPEG_QUALITY = 65

        # quản lý lỗi / stale output
        self.consecutive_failures = 0
        self.MAX_FAILURES_BEFORE_CLEAR = 3

        # tái sử dụng HTTP connection
        self.session = requests.Session()

    def _resize_keep_ratio(self, frame, long_side):
        h, w = frame.shape[:2]
        if max(h, w) <= long_side:
            return frame

        scale = long_side / float(max(h, w))
        nw = int(w * scale)
        nh = int(h * scale)
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    def _clear_output(self):
        self.last_out = {
            "phone": False,
            "dets": [],
            "persons": [],
            "phones": []
        }

    def run(self, frame):
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return self.last_out

        now = time.time()
        if now - self.last_call < self.CALL_EVERY_SEC:
            return self.last_out

        self.last_call = now

        send_frame = self._resize_keep_ratio(frame, self.SEND_LONG_SIDE)

        ok, jpg = cv2.imencode(
            ".jpg",
            send_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.JPEG_QUALITY]
        )
        if not ok:
            return self.last_out

        try:
            r = self.session.post(
                self.yolo_url,
                files={"image": ("frame.jpg", jpg.tobytes(), "image/jpeg")},
                timeout=self.TIMEOUT_SEC,
            )
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.MAX_FAILURES_BEFORE_CLEAR:
                    self._clear_output()
                return self.last_out

        except Exception as e:
            print(f"Vision API error: {e}")
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.MAX_FAILURES_BEFORE_CLEAR:
                self._clear_output()
            return self.last_out

        if not data.get("ok"):
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.MAX_FAILURES_BEFORE_CLEAR:
                self._clear_output()
            return self.last_out

        self.consecutive_failures = 0

        self.last_out = {
            "phone": False,
            "dets": data.get("dets", []),
            "persons": data.get("persons", []),
            "phones": data.get("phones", [])
        }
        return self.last_out

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass