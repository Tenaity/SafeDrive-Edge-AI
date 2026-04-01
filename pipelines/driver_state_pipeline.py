import cv2
import numpy as np
import mediapipe as mp
from collections import deque


class DriverStatePipeline:
    """
    DriverStatePipeline:
    - EAR-based drowsiness
    - head_down detection
    - yawn detection (MAR)
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.EAR_CONSEC_FRAMES_ON = 22
        self.EAR_CONSEC_FRAMES_OFF = 6

        self.BASELINE_ALPHA = 0.015
        self.MIN_BASELINE_SAMPLES = 20
        self.FALLBACK_EAR_THRESHOLD = 0.23

        self.THRESHOLD_RATIO_ON = 0.70
        self.THRESHOLD_RATIO_OFF = 0.78

        self.ear_history = deque(maxlen=7)
        self.mar_history = deque(maxlen=5)

        self.eye_counter_closed = 0
        self.eye_counter_open = 0
        self.baseline_ear = None
        self.baseline_count = 0
        self.is_drowsy_state = False

        self.yawn_counter = 0
        self.YAWN_CONSEC_FRAMES = 8
        self.MAR_THRESHOLD = 0.55

        self.MIN_FACE_WIDTH_RATIO = 0.12
        self.MIN_FACE_HEIGHT_RATIO = 0.12

        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [263, 387, 385, 362, 380, 373]
        self.face_box_idx = [10, 152, 234, 454]

        self.NOSE_IDX = 1
        self.CHIN_IDX = 152
        self.LEFT_EYE_TOP = 159
        self.RIGHT_EYE_TOP = 386

        # mouth landmarks
        self.UPPER_LIP = 13
        self.LOWER_LIP = 14
        self.MOUTH_LEFT = 78
        self.MOUTH_RIGHT = 308

    @staticmethod
    def _euclid(p1, p2):
        return float(np.linalg.norm(p1 - p2))

    def _compute_ear(self, eye_pts: np.ndarray):
        if eye_pts is None or len(eye_pts) != 6:
            return None
        try:
            p1, p2, p3, p4, p5, p6 = eye_pts
            A = self._euclid(p2, p6)
            B = self._euclid(p3, p5)
            C = self._euclid(p1, p4)
            if C <= 1e-6:
                return None
            return (A + B) / (2.0 * C)
        except Exception:
            return None

    @staticmethod
    def _center_of_points(points: np.ndarray):
        if points is None or len(points) == 0:
            return None
        return (int(points[:, 0].mean()), int(points[:, 1].mean()))

    def _smooth(self, value, hist: deque):
        if value is None:
            return None
        hist.append(float(value))
        arr = np.array(hist, dtype=np.float32)
        return float(np.mean(arr))

    def _face_quality_ok(self, lm, w, h):
        try:
            top = np.array([lm[self.face_box_idx[0]].x * w, lm[self.face_box_idx[0]].y * h], dtype=np.float32)
            bottom = np.array([lm[self.face_box_idx[1]].x * w, lm[self.face_box_idx[1]].y * h], dtype=np.float32)
            left = np.array([lm[self.face_box_idx[2]].x * w, lm[self.face_box_idx[2]].y * h], dtype=np.float32)
            right = np.array([lm[self.face_box_idx[3]].x * w, lm[self.face_box_idx[3]].y * h], dtype=np.float32)

            face_w = abs(right[0] - left[0])
            face_h = abs(bottom[1] - top[1])

            return face_w >= w * self.MIN_FACE_WIDTH_RATIO and face_h >= h * self.MIN_FACE_HEIGHT_RATIO
        except Exception:
            return False

    def _update_baseline(self, ear):
        if ear is None:
            return
        if ear > 0.18:
            if self.baseline_ear is None:
                self.baseline_ear = ear
            else:
                self.baseline_ear = (
                    (1.0 - self.BASELINE_ALPHA) * self.baseline_ear
                    + self.BASELINE_ALPHA * ear
                )
            self.baseline_count += 1

    def _get_threshold_on(self):
        if self.baseline_ear is not None and self.baseline_count >= self.MIN_BASELINE_SAMPLES:
            return self.baseline_ear * self.THRESHOLD_RATIO_ON
        return self.FALLBACK_EAR_THRESHOLD

    def _get_threshold_off(self):
        if self.baseline_ear is not None and self.baseline_count >= self.MIN_BASELINE_SAMPLES:
            return self.baseline_ear * self.THRESHOLD_RATIO_OFF
        return self.FALLBACK_EAR_THRESHOLD + 0.02

    def _reset_temporal_state(self, clear_history=False):
        self.eye_counter_closed = 0
        self.eye_counter_open = 0
        if clear_history:
            self.ear_history.clear()
            self.mar_history.clear()

    def _head_down(self, lm, w, h):
        try:
            nose = np.array([lm[self.NOSE_IDX].x * w, lm[self.NOSE_IDX].y * h], dtype=np.float32)
            chin = np.array([lm[self.CHIN_IDX].x * w, lm[self.CHIN_IDX].y * h], dtype=np.float32)
            le = np.array([lm[self.LEFT_EYE_TOP].x * w, lm[self.LEFT_EYE_TOP].y * h], dtype=np.float32)
            re = np.array([lm[self.RIGHT_EYE_TOP].x * w, lm[self.RIGHT_EYE_TOP].y * h], dtype=np.float32)

            eye_mid = (le + re) / 2.0
            eye_to_chin = chin[1] - eye_mid[1]
            if eye_to_chin <= 1:
                return False

            ratio = (nose[1] - eye_mid[1]) / eye_to_chin
            return ratio > 0.52
        except Exception:
            return False

    def _compute_mar(self, lm, w, h):
        try:
            upper = np.array([lm[self.UPPER_LIP].x * w, lm[self.UPPER_LIP].y * h], dtype=np.float32)
            lower = np.array([lm[self.LOWER_LIP].x * w, lm[self.LOWER_LIP].y * h], dtype=np.float32)
            left = np.array([lm[self.MOUTH_LEFT].x * w, lm[self.MOUTH_LEFT].y * h], dtype=np.float32)
            right = np.array([lm[self.MOUTH_RIGHT].x * w, lm[self.MOUTH_RIGHT].y * h], dtype=np.float32)

            vertical = self._euclid(upper, lower)
            horizontal = self._euclid(left, right)
            if horizontal <= 1e-6:
                return None
            return vertical / horizontal
        except Exception:
            return None

    def run(self, frame):
        empty_out = {
            "ear": None,
            "yaw": None,
            "drowsy": False,
            "distracted": False,
            "nose_point": None,
            "face_center": None,
            "left_eye_center": None,
            "right_eye_center": None,
            "baseline_ear": None if self.baseline_ear is None else round(float(self.baseline_ear), 3),
            "ear_threshold_on": round(float(self._get_threshold_on()), 3),
            "ear_threshold_off": round(float(self._get_threshold_off()), 3),
            "face_ok": False,
            "head_down": False,
            "mar": None,
            "yawning": False,
        }

        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            self._reset_temporal_state(clear_history=True)
            self.is_drowsy_state = False
            self.yawn_counter = 0
            return empty_out

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        ear = None
        mar = None
        nose_point = None
        face_center = None
        left_eye_center = None
        right_eye_center = None
        face_ok = False
        head_down = False
        yawning = False

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            face_ok = self._face_quality_ok(lm, w, h)

            if face_ok:
                left_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in self.left_eye_idx], dtype=np.float32)
                right_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in self.right_eye_idx], dtype=np.float32)

                left_eye_center = self._center_of_points(left_eye)
                right_eye_center = self._center_of_points(right_eye)

                ear_left = self._compute_ear(left_eye)
                ear_right = self._compute_ear(right_eye)

                if ear_left is not None and ear_right is not None:
                    raw_ear = (ear_left + ear_right) / 2.0
                    ear = self._smooth(raw_ear, self.ear_history)
                    self._update_baseline(ear)
                else:
                    ear = None
                    self._reset_temporal_state(clear_history=False)

                raw_mar = self._compute_mar(lm, w, h)
                mar = self._smooth(raw_mar, self.mar_history)

                if mar is not None and mar > self.MAR_THRESHOLD:
                    self.yawn_counter += 1
                    if self.yawn_counter >= self.YAWN_CONSEC_FRAMES:
                        yawning = True
                else:
                    self.yawn_counter = 0

                try:
                    nose_point = (int(lm[1].x * w), int(lm[1].y * h))
                    face_center = nose_point
                except Exception:
                    nose_point = None
                    face_center = None

                head_down = self._head_down(lm, w, h)
            else:
                self._reset_temporal_state(clear_history=False)
                self.yawn_counter = 0
        else:
            self._reset_temporal_state(clear_history=True)
            self.yawn_counter = 0

        threshold_on = self._get_threshold_on()
        threshold_off = self._get_threshold_off()

        if ear is None or not face_ok:
            self.eye_counter_open += 1
            self.eye_counter_closed = 0
            if self.eye_counter_open >= self.EAR_CONSEC_FRAMES_OFF:
                self.is_drowsy_state = False
        else:
            if not self.is_drowsy_state:
                if ear < threshold_on:
                    self.eye_counter_closed += 1
                    self.eye_counter_open = 0
                else:
                    self.eye_counter_closed = 0
                    self.eye_counter_open += 1

                if self.eye_counter_closed >= self.EAR_CONSEC_FRAMES_ON:
                    self.is_drowsy_state = True
                    self.eye_counter_open = 0
            else:
                if ear > threshold_off:
                    self.eye_counter_open += 1
                    self.eye_counter_closed = 0
                    if self.eye_counter_open >= self.EAR_CONSEC_FRAMES_OFF:
                        self.is_drowsy_state = False
                        self.eye_counter_closed = 0
                else:
                    self.eye_counter_open = 0

        yaw = None
        distracted = False

        return {
            "ear": None if ear is None else round(float(ear), 3),
            "yaw": yaw,
            "drowsy": bool(self.is_drowsy_state),
            "distracted": distracted,
            "nose_point": nose_point,
            "face_center": face_center,
            "left_eye_center": left_eye_center,
            "right_eye_center": right_eye_center,
            "baseline_ear": None if self.baseline_ear is None else round(float(self.baseline_ear), 3),
            "ear_threshold_on": round(float(threshold_on), 3),
            "ear_threshold_off": round(float(threshold_off), 3),
            "face_ok": face_ok,
            "head_down": head_down,
            "mar": None if mar is None else round(float(mar), 3),
            "yawning": yawning,
        }

    def close(self):
        try:
            if self.face_mesh is not None:
                self.face_mesh.close()
        except Exception:
            pass