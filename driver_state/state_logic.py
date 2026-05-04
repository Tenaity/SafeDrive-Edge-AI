from collections import deque


class DriverStateLogic:
    # Blink score thresholds (0.0 = fully open, 1.0 = fully closed)
    BLINK_CLOSED_THRESHOLD = 0.35
    BLINK_OPEN_THRESHOLD = 0.25

    # Frames of continuous closure to trigger / clear drowsy
    EAR_CONSEC_FRAMES_ON = 11
    EAR_CONSEC_FRAMES_OFF = 8

    MAR_THRESHOLD = 0.46
    PERCLOS_WINDOW = 90  # frames (~3s at 30fps)

    def __init__(self):
        self._closed_counter = 0
        self._open_counter = 0
        self._eyes_closed = False
        self._drowsy = False
        self._yawning = False
        self._perclos_buf: deque[int] = deque(maxlen=self.PERCLOS_WINDOW)

    def _compute_mar(self, lm_out: dict) -> float | None:
        mouth_top = lm_out.get("mouth_top")
        mouth_bot = lm_out.get("mouth_bot")
        mouth_l = lm_out.get("mouth_l")
        mouth_r = lm_out.get("mouth_r")

        if not all([mouth_top, mouth_bot, mouth_l, mouth_r]):
            return None

        if mouth_bot is None or mouth_top is None or mouth_r is None or mouth_l is None:
            return None

        vert = abs(mouth_bot[1] - mouth_top[1])
        horiz = abs(mouth_r[0] - mouth_l[0])
        if horiz < 1e-3:
            return None
        return vert / horiz

    def update(self, face_out: dict, landmark_out: dict, pose_out: dict) -> dict:
        bbox = face_out.get("bbox") if isinstance(face_out, dict) else None

        if bbox is None:
            return {}
        score = face_out.get("score", 0.0) if isinstance(face_out, dict) else 0.0
        lm_ok = landmark_out.get("ok", False)

        face_ok = bool(bbox is not None and len(bbox) == 4 and score > 0.0)

        face_center = None
        if face_ok:
            x1, y1, x2, y2 = bbox
            face_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # --- Blink score (drowsiness) ---
        blink_left = float(landmark_out.get("eye_blink_left", 0.0))
        blink_right = float(landmark_out.get("eye_blink_right", 0.0))
        blink_score = (blink_left + blink_right) / 2.0

        if lm_ok:
            currently_closed = blink_score > self.BLINK_CLOSED_THRESHOLD
            currently_open = blink_score < self.BLINK_OPEN_THRESHOLD

            if currently_closed:
                self._closed_counter += 1
                self._open_counter = 0
                self._eyes_closed = True
            elif currently_open:
                self._open_counter += 1
                if self._open_counter >= self.EAR_CONSEC_FRAMES_OFF:
                    self._eyes_closed = False
                    self._closed_counter = 0
            # hysteresis zone: counters frozen, state unchanged

            self._perclos_buf.append(1 if self._eyes_closed else 0)

            if not self._drowsy and self._closed_counter >= self.EAR_CONSEC_FRAMES_ON:
                self._drowsy = True
            elif self._drowsy and not self._eyes_closed:
                self._drowsy = False

        else:
            # No landmark: decay toward open/awake
            self._perclos_buf.append(0)
            self._closed_counter = 0
            self._open_counter = min(self._open_counter + 1, self.EAR_CONSEC_FRAMES_OFF)
            if self._open_counter >= self.EAR_CONSEC_FRAMES_OFF:
                self._eyes_closed = False
                self._drowsy = False

        perclos = sum(self._perclos_buf) / len(self._perclos_buf) if self._perclos_buf else 0.0

        # Eyes state label
        if not lm_ok:
            eyes_state = "unknown"
            eye_signal_conf = 0.0
        elif self._eyes_closed:
            eyes_state = "closed"
            eye_signal_conf = min(1.0, blink_score / max(self.BLINK_CLOSED_THRESHOLD, 1e-6))
        else:
            eyes_state = "open"
            eye_signal_conf = min(1.0, (1.0 - blink_score) / max(1.0 - self.BLINK_OPEN_THRESHOLD, 1e-6))

        # --- MAR (yawning) ---
        mar = self._compute_mar(landmark_out) if lm_ok else None
        self._yawning = bool(mar is not None and mar > self.MAR_THRESHOLD)

        # --- Distracted / Head down ---
        head_down = pose_out.get("head_down", False)
        looking_away = pose_out.get("looking_away", False)
        forward_working = bool(face_ok and lm_ok and not head_down and not looking_away)

        return {
            "face_ok": face_ok,
            "face_box": bbox,
            "face_center": face_center,
            "left_eye_center": landmark_out.get("left_eye"),
            "right_eye_center": landmark_out.get("right_eye"),
            "nose_point": landmark_out.get("nose"),
            "yaw": pose_out.get("yaw", 0.0),
            "pitch": pose_out.get("pitch", 0.0),
            "roll": pose_out.get("roll", 0.0),
            "head_down": head_down,
            "distracted": looking_away,
            "drowsy": self._drowsy,
            "yawning": self._yawning,
            "ear": None,
            "mar": mar,
            "baseline_ear": None,
            "ear_threshold_on": self.BLINK_CLOSED_THRESHOLD,
            "ear_threshold_off": self.BLINK_OPEN_THRESHOLD,
            "forward_working": forward_working,
            "eyes_state": eyes_state,
            "eye_signal_conf": eye_signal_conf,
            "perclos": perclos,
        }
