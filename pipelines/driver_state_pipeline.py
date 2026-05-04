import cv2
import math
import numpy as np
import mediapipe as mp
from collections import deque

from utils.types import DriverStateOut


class DriverStatePipeline:
    """
    DriverStatePipeline:
    - Person ROI -> FaceDetection -> FaceMesh
    - Focus ROI mắt trái/phải
    - Kết hợp EAR + eye ROI ratio khi đáng tin
    - Giảm nhầm giữa head_down và drowsy
    - Dùng shoulder/hip posture để nhận ra tư thế chúi người làm việc
    - Chịu tốt hơn khi camera lệch trái/phải bằng cách ưu tiên mắt rõ hơn
    - Khi đeo kính / phản sáng / ROI mắt không đáng tin thì giảm trọng số ROI mắt
    - Thêm eye_signal_conf / eyes_state để tránh kết luận quá mạnh khi tín hiệu yếu
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.30
        )

        self.EAR_CONSEC_FRAMES_ON = 11
        self.EAR_CONSEC_FRAMES_OFF = 8

        self.BASELINE_ALPHA = 0.008
        self.MIN_BASELINE_SAMPLES = 12

        self.FALLBACK_EAR_THRESHOLD = 0.21
        self.THRESHOLD_RATIO_ON = 0.80
        self.THRESHOLD_RATIO_OFF = 0.88

        self.ear_history = deque(maxlen=11)
        self.mar_history = deque(maxlen=5)
        self.eye_roi_ratio_history = deque(maxlen=7)
        self.eye_closed_window = deque(maxlen=36)

        self.eye_counter_closed = 0
        self.eye_counter_open = 0
        self.baseline_ear = None
        self.baseline_count = 0
        self.is_drowsy_state = False

        self.baseline_eye_ratio = None
        self.eye_ratio_count = 0

        self.face_lost_counter = 0
        self.FACE_LOST_HOLD_FRAMES = 16

        self.yawn_counter = 0
        self.YAWN_CONSEC_FRAMES = 6
        self.MAR_THRESHOLD = 0.46

        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [263, 387, 385, 362, 380, 373]

        self.NOSE_IDX = 1
        self.CHIN_IDX = 152
        self.LEFT_EYE_TOP = 159
        self.RIGHT_EYE_TOP = 386

        self.UPPER_LIP = 13
        self.LOWER_LIP = 14
        self.MOUTH_LEFT = 78
        self.MOUTH_RIGHT = 308

        self.head_down_counter = 0
        self.head_down_state = False

        self.torso_angle_baseline = None
        self.torso_angle_count = 0
        self.forward_counter = 0
        self.forward_working_state = False

        self.head_down_hold_counter = 0

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.clahe_eye = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(v, hi))

    @staticmethod
    def _euclid(p1, p2):
        return float(np.linalg.norm(p1 - p2))

    @staticmethod
    def _center_of_points(points: np.ndarray):
        if points is None or len(points) == 0:
            return None
        return (int(points[:, 0].mean()), int(points[:, 1].mean()))

    def _compute_ear(self, eye_pts: np.ndarray) -> float | None:
        if eye_pts is None or len(eye_pts) != 6:
            return None
        try:
            p1, p2, p3, p4, p5, p6 = eye_pts
            a = self._euclid(p2, p6)
            b = self._euclid(p3, p5)
            c = self._euclid(p1, p4)
            if c <= 1e-6:
                return None
            return (a + b) / (2.0 * c)
        except Exception:
            return None

    def _compute_mar(self, lm, w: float, h: float) -> float | None:
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

    def _smooth(self, value: float | None, hist: deque) -> float | None:
        if value is None:
            return None
        hist.append(float(value))
        arr = np.array(hist, dtype=np.float32)
        med = float(np.median(arr))
        clipped = arr[np.abs(arr - med) < 0.05]
        if len(clipped) == 0:
            clipped = arr
        return float(np.mean(clipped))

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

    def _update_eye_ratio_baseline(self, eye_ratio):
        if eye_ratio is None:
            return
        if eye_ratio > 0.08:
            if self.baseline_eye_ratio is None:
                self.baseline_eye_ratio = eye_ratio
            else:
                self.baseline_eye_ratio = (
                    (1.0 - self.BASELINE_ALPHA) * self.baseline_eye_ratio
                    + self.BASELINE_ALPHA * eye_ratio
                )
            self.eye_ratio_count += 1

    def _get_threshold_on(self) -> float:
        if self.baseline_ear is not None and self.baseline_count >= self.MIN_BASELINE_SAMPLES:
            return max(0.185, self.baseline_ear * self.THRESHOLD_RATIO_ON)
        return self.FALLBACK_EAR_THRESHOLD

    def _get_threshold_off(self) -> float:
        if self.baseline_ear is not None and self.baseline_count >= self.MIN_BASELINE_SAMPLES:
            return max(0.205, self.baseline_ear * self.THRESHOLD_RATIO_OFF)
        return self.FALLBACK_EAR_THRESHOLD + 0.025

    def _get_eye_ratio_threshold_on(self):
        if self.baseline_eye_ratio is not None and self.eye_ratio_count >= self.MIN_BASELINE_SAMPLES:
            return max(0.10, self.baseline_eye_ratio * 0.72)
        return 0.16

    def _get_eye_ratio_threshold_off(self):
        if self.baseline_eye_ratio is not None and self.eye_ratio_count >= self.MIN_BASELINE_SAMPLES:
            return max(0.12, self.baseline_eye_ratio * 0.82)
        return 0.19

    def _reset_temporal_state(self, clear_history=False):
        self.eye_counter_closed = 0
        self.eye_counter_open = 0
        if clear_history:
            self.ear_history.clear()
            self.mar_history.clear()
            self.eye_roi_ratio_history.clear()
            self.eye_closed_window.clear()

    def _head_down_ratio(self, lm, w: float, h: float) -> float:
        try:
            nose = np.array([lm[self.NOSE_IDX].x * w, lm[self.NOSE_IDX].y * h], dtype=np.float32)
            chin = np.array([lm[self.CHIN_IDX].x * w, lm[self.CHIN_IDX].y * h], dtype=np.float32)
            le = np.array([lm[self.LEFT_EYE_TOP].x * w, lm[self.LEFT_EYE_TOP].y * h], dtype=np.float32)
            re = np.array([lm[self.RIGHT_EYE_TOP].x * w, lm[self.RIGHT_EYE_TOP].y * h], dtype=np.float32)

            eye_mid = (le + re) / 2.0
            eye_to_chin = chin[1] - eye_mid[1]
            if eye_to_chin <= 1:
                return None

            ratio = (nose[1] - eye_mid[1]) / eye_to_chin
            return float(ratio)
        except Exception:
            return None

    def _update_head_down_state(self, ratio):
        if ratio is None:
            self.head_down_counter = max(0, self.head_down_counter - 1)
        else:
            if ratio > 0.62:
                self.head_down_counter += 1
            elif ratio < 0.55:
                self.head_down_counter = max(0, self.head_down_counter - 1)

        if self.head_down_counter >= 4:
            self.head_down_state = True
        elif self.head_down_counter == 0:
            self.head_down_state = False

        if self.head_down_state:
            self.head_down_hold_counter += 1
        else:
            self.head_down_hold_counter = 0

        return self.head_down_state

    def _torso_angle(self, left_shoulder, right_shoulder, left_hip, right_hip):
        if left_shoulder is None or right_shoulder is None or left_hip is None or right_hip is None:
            return None

        sx = (left_shoulder[0] + right_shoulder[0]) / 2.0
        sy = (left_shoulder[1] + right_shoulder[1]) / 2.0
        hx = (left_hip[0] + right_hip[0]) / 2.0
        hy = (left_hip[1] + right_hip[1]) / 2.0

        dx = sx - hx
        dy = hy - sy
        if abs(dy) < 1:
            return None

        angle_deg = math.degrees(math.atan2(abs(dx), max(1.0, dy)))
        return float(angle_deg)

    def _update_forward_working(self, torso_angle):
        if torso_angle is None:
            self.forward_counter = max(0, self.forward_counter - 1)
            if self.forward_counter == 0:
                self.forward_working_state = False
            return self.forward_working_state

        if torso_angle < 18.0:
            if self.torso_angle_baseline is None:
                self.torso_angle_baseline = torso_angle
            else:
                self.torso_angle_baseline = (
                    0.99 * self.torso_angle_baseline + 0.01 * torso_angle
                )
            self.torso_angle_count += 1

        if self.torso_angle_baseline is None or self.torso_angle_count < 15:
            threshold = 16.0
        else:
            threshold = max(16.0, self.torso_angle_baseline + 6.0)

        if torso_angle > threshold:
            self.forward_counter += 1
        else:
            self.forward_counter = max(0, self.forward_counter - 1)

        if self.forward_counter >= 3:
            self.forward_working_state = True
        elif self.forward_counter == 0:
            self.forward_working_state = False

        return self.forward_working_state

    def _make_driver_roi(self, frame, person_box=None):
        h, w = frame.shape[:2]

        if person_box is None:
            return frame, (0, 0, w, h)

        px1, py1, px2, py2 = map(float, person_box)
        pw = px2 - px1
        ph = py2 - py1

        x1 = int(self._clamp(px1 - pw * 0.22, 0, w - 1))
        x2 = int(self._clamp(px2 + pw * 0.22, 0, w - 1))
        y1 = int(self._clamp(py1 - ph * 0.14, 0, h - 1))
        y2 = int(self._clamp(py1 + ph * 0.68, 0, h - 1))

        if x2 <= x1 or y2 <= y1:
            return frame, (0, 0, w, h)

        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return frame, (0, 0, w, h)

        return crop, (x1, y1, x2, y2)

    def _refine_face_roi(self, driver_roi):
        h, w = driver_roi.shape[:2]
        rgb = cv2.cvtColor(driver_roi, cv2.COLOR_BGR2RGB)
        det = self.face_detector.process(rgb)

        if not det.detections:
            return driver_roi, (0, 0, w, h)

        best = det.detections[0]
        box = best.location_data.relative_bounding_box

        x = int(box.xmin * w)
        y = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)

        pad_x = int(bw * 0.40)
        pad_y = int(bh * 0.50)

        x1 = self._clamp(x - pad_x, 0, w - 1)
        y1 = self._clamp(y - pad_y, 0, h - 1)
        x2 = self._clamp(x + bw + pad_x, 0, w - 1)
        y2 = self._clamp(y + bh + pad_y, 0, h - 1)

        if x2 <= x1 or y2 <= y1:
            return driver_roi, (0, 0, w, h)

        face_crop = driver_roi[y1:y2, x1:x2]
        if face_crop is None or face_crop.size == 0:
            return driver_roi, (0, 0, w, h)

        return face_crop, (x1, y1, x2, y2)

    def _upscale_for_far_face(self, face_crop):
        h, w = face_crop.shape[:2]
        target_min = 560
        scale = 1.0

        if min(h, w) < target_min:
            scale = target_min / float(min(h, w))

        if scale <= 1.01:
            return face_crop, 1.0

        new_w = int(w * scale)
        new_h = int(h * scale)
        up = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return up, scale

    def _eye_bbox_from_landmarks(self, eye_pts, img_w, img_h, pad_ratio=0.45):
        if eye_pts is None or len(eye_pts) == 0:
            return None

        xs = eye_pts[:, 0]
        ys = eye_pts[:, 1]

        x1 = float(np.min(xs))
        y1 = float(np.min(ys))
        x2 = float(np.max(xs))
        y2 = float(np.max(ys))

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        pad_x = bw * pad_ratio
        pad_y = bh * 1.00

        rx1 = int(self._clamp(x1 - pad_x, 0, img_w - 1))
        ry1 = int(self._clamp(y1 - pad_y, 0, img_h - 1))
        rx2 = int(self._clamp(x2 + pad_x, 0, img_w - 1))
        ry2 = int(self._clamp(y2 + pad_y, 0, img_h - 1))

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        return (rx1, ry1, rx2, ry2)

    def _enhance_eye_roi(self, eye_roi):
        if eye_roi is None or eye_roi.size == 0:
            return None

        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        gray = self.clahe_eye.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray

    def _eye_open_ratio_from_roi(self, gray_eye):
        if gray_eye is None or gray_eye.size == 0:
            return None

        h, w = gray_eye.shape[:2]
        if h < 8 or w < 8:
            return None

        th = cv2.adaptiveThreshold(
            gray_eye,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        kernel = np.ones((2, 2), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0:
            return None
        return float(bh / float(bw))

    def _eye_box_quality(self, eye_box):
        if eye_box is None:
            return 0.0
        x1, y1, x2, y2 = eye_box
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        return float(w * h)

    def _eye_roi_reliability(self, gray_eye):
        if gray_eye is None or gray_eye.size == 0:
            return False

        h, w = gray_eye.shape[:2]
        if h < 10 or w < 14:
            return False

        mean_val = float(np.mean(gray_eye))
        std_val = float(np.std(gray_eye))
        if mean_val > 210 or mean_val < 25:
            return False

        if std_val < 12:
            return False
        return True

    def _select_eye_signal(self, ear_left, ear_right, ratio_left, ratio_right, q_left, q_right):
        left_ok = ear_left is not None and q_left > 0
        right_ok = ear_right is not None and q_right > 0

        if left_ok and right_ok:
            bigger = max(q_left, q_right)
            smaller = min(q_left, q_right)

            if smaller >= 0.60 * bigger:
                ratios = [r for r in [ratio_left, ratio_right] if r is not None]
                fused_ratio = sum(ratios) / len(ratios) if ratios else None
                return (ear_left + ear_right) / 2.0, fused_ratio, "both"

            if q_left > q_right:
                return ear_left, ratio_left, "left"
            return ear_right, ratio_right, "right"

        if left_ok:
            return ear_left, ratio_left, "left"
        if right_ok:
            return ear_right, ratio_right, "right"
        return None, None, "none"

    def _perclos(self) -> float:
        valid = [v for v in self.eye_closed_window if v is not None]
        if not valid:
            return 0.0
        return float(sum(1 for v in valid if v) / len(valid))

    def run(
        self,
        frame: np.ndarray,
        person_box=None,
        left_shoulder=None,
        right_shoulder=None,
        left_hip=None,
        right_hip=None,
    ) -> DriverStateOut:
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
            "forward_working": False,
            "eyes_state": "unknown",
            "eye_signal_conf": 0.0,
            "perclos": round(self._perclos(), 3),
        }

        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            self._reset_temporal_state(clear_history=True)
            self.is_drowsy_state = False
            self.yawn_counter = 0
            self.face_lost_counter = 0
            self.head_down_counter = 0
            self.head_down_state = False
            self.forward_counter = 0
            self.forward_working_state = False
            return empty_out

        torso_angle = self._torso_angle(left_shoulder, right_shoulder, left_hip, right_hip)
        forward_working = self._update_forward_working(torso_angle)

        driver_roi, driver_roi_box = self._make_driver_roi(frame, person_box)
        face_crop, face_box_in_driver = self._refine_face_roi(driver_roi)

        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray_face = self.clahe.apply(gray_face)
        face_crop_enhanced = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)

        face_up, up_scale = self._upscale_for_far_face(face_crop_enhanced)

        rgb = cv2.cvtColor(face_up, cv2.COLOR_BGR2RGB)
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

        eyes_state = "unknown"
        eye_signal_conf = 0.0

        eye_roi_open_ratio = None
        eye_roi_closed = False
        ear_closed = False
        side_mode = "none"
        threshold_on = self._get_threshold_on()
        threshold_off = self._get_threshold_off()
        eye_ratio_th_on = self._get_eye_ratio_threshold_on()
        eye_ratio_th_off = self._get_eye_ratio_threshold_off()
        asym_bad = False

        if res.multi_face_landmarks:
            self.face_lost_counter = 0

            lm = res.multi_face_landmarks[0].landmark
            h_up, w_up = face_up.shape[:2]

            left_eye = np.array([(lm[i].x * w_up, lm[i].y * h_up) for i in self.left_eye_idx], dtype=np.float32)
            right_eye = np.array([(lm[i].x * w_up, lm[i].y * h_up) for i in self.right_eye_idx], dtype=np.float32)

            left_eye_center_local = self._center_of_points(left_eye)
            right_eye_center_local = self._center_of_points(right_eye)

            ear_left = self._compute_ear(left_eye)
            ear_right = self._compute_ear(right_eye)

            head_ratio = self._head_down_ratio(lm, w_up, h_up)
            head_down = self._update_head_down_state(head_ratio)

            raw_mar = self._compute_mar(lm, w_up, h_up)
            mar = self._smooth(raw_mar, self.mar_history)

            if mar is not None and mar > self.MAR_THRESHOLD:
                self.yawn_counter += 1
                if self.yawn_counter >= self.YAWN_CONSEC_FRAMES:
                    yawning = True
            else:
                self.yawn_counter = 0

            left_eye_box = self._eye_bbox_from_landmarks(left_eye, w_up, h_up)
            right_eye_box = self._eye_bbox_from_landmarks(right_eye, w_up, h_up)

            ratio_left = None
            ratio_right = None

            q_left = self._eye_box_quality(left_eye_box)
            q_right = self._eye_box_quality(right_eye_box)

            if left_eye_box is not None:
                ex1, ey1, ex2, ey2 = left_eye_box
                eye_roi = face_up[ey1:ey2, ex1:ex2]
                gray_eye = self._enhance_eye_roi(eye_roi)
                if self._eye_roi_reliability(gray_eye):
                    ratio_left = self._eye_open_ratio_from_roi(gray_eye)

            if right_eye_box is not None:
                ex1, ey1, ex2, ey2 = right_eye_box
                eye_roi = face_up[ey1:ey2, ex1:ex2]
                gray_eye = self._enhance_eye_roi(eye_roi)
                if self._eye_roi_reliability(gray_eye):
                    ratio_right = self._eye_open_ratio_from_roi(gray_eye)

            fused_ear, fused_eye_ratio, side_mode = self._select_eye_signal(
                ear_left, ear_right,
                ratio_left, ratio_right,
                q_left, q_right
            )

            if fused_ear is not None:
                ear = self._smooth(fused_ear, self.ear_history)
                face_ok = True

            if fused_eye_ratio is not None:
                eye_roi_open_ratio = self._smooth(fused_eye_ratio, self.eye_roi_ratio_history)

            if ear is not None and not head_down:
                self._update_baseline(ear)

            if eye_roi_open_ratio is not None and not head_down:
                self._update_eye_ratio_baseline(eye_roi_open_ratio)

            dx1, dy1, _, _ = driver_roi_box
            fx1, fy1, _, _ = face_box_in_driver

            def to_frame(pt):
                if pt is None:
                    return None
                x = int(pt[0] / up_scale) + fx1 + dx1
                y = int(pt[1] / up_scale) + fy1 + dy1
                return (x, y)

            try:
                nose_local = (int(lm[self.NOSE_IDX].x * w_up), int(lm[self.NOSE_IDX].y * h_up))
                nose_point = to_frame(nose_local)
                face_center = nose_point
            except Exception:
                nose_point = None
                face_center = None

            left_eye_center = to_frame(left_eye_center_local)
            right_eye_center = to_frame(right_eye_center_local)

            if ear is not None:
                ear_closed = ear < threshold_on
            if eye_roi_open_ratio is not None:
                eye_roi_closed = eye_roi_open_ratio < eye_ratio_th_on

            if side_mode == "both" and ear_left is not None and ear_right is not None:
                if abs(ear_left - ear_right) > 0.09:
                    asym_bad = True

            face_area = float(face_up.shape[0] * face_up.shape[1])
            size_conf = 1.0 if face_area >= 140000 else max(0.35, face_area / 140000.0)
            side_conf = 1.0 if side_mode == "both" else (0.78 if side_mode in ("left", "right") else 0.0)
            roi_conf = 0.0
            if eye_roi_open_ratio is not None and side_mode == "both":
                roi_conf = 1.0
            elif eye_roi_open_ratio is not None:
                roi_conf = 0.75
            elif ear is not None:
                roi_conf = 0.55

            eye_signal_conf = max(0.0, min(1.0, 0.45 * side_conf + 0.35 * roi_conf + 0.20 * size_conf))
        else:
            self.face_lost_counter += 1
            self.yawn_counter = 0
            self.head_down_counter = max(0, self.head_down_counter - 1)
            if self.head_down_counter == 0:
                self.head_down_state = False
            head_down = self.head_down_state
            eye_signal_conf = 0.0

        strong_closed = False
        weak_closed = False
        closed_evidence_now = False

        if ear is None or not face_ok:
            self.eye_closed_window.append(None)
            if self.face_lost_counter > self.FACE_LOST_HOLD_FRAMES:
                self.eye_counter_open += 1
                self.eye_counter_closed = 0
                if self.eye_counter_open >= self.EAR_CONSEC_FRAMES_OFF:
                    self.is_drowsy_state = False
            eyes_state = "unknown"
        else:
            if head_down and forward_working:
                self.eye_closed_window.append(False)
                self.eye_counter_closed = 0
                self.eye_counter_open += 1
                if self.eye_counter_open >= self.EAR_CONSEC_FRAMES_OFF:
                    self.is_drowsy_state = False
                eyes_state = "unknown" if eye_signal_conf < 0.65 else ("closed" if ear_closed and eye_roi_closed else "open")
            elif head_down and eye_signal_conf < 0.62:
                self.eye_closed_window.append(None)
                self.eye_counter_closed = max(0, self.eye_counter_closed - 1)
                self.eye_counter_open += 1
                if self.eye_counter_open >= self.EAR_CONSEC_FRAMES_OFF:
                    self.is_drowsy_state = False
                eyes_state = "unknown"
            else:
                if not asym_bad:
                    if side_mode == "both":
                        if ear_closed and eye_roi_closed:
                            strong_closed = True
                        elif ear_closed or eye_roi_closed:
                            weak_closed = True
                    elif side_mode in ("left", "right"):
                        if ear_closed and eye_signal_conf >= 0.72:
                            strong_closed = True
                        elif ear_closed:
                            weak_closed = True

                closed_evidence_now = strong_closed or weak_closed
                self.eye_closed_window.append(closed_evidence_now)

                if strong_closed:
                    eyes_state = "closed"
                elif weak_closed and eye_signal_conf >= 0.72:
                    eyes_state = "closed"
                elif eye_signal_conf >= 0.55:
                    eyes_state = "open"
                else:
                    eyes_state = "unknown"

                if not self.is_drowsy_state:
                    if strong_closed:
                        self.eye_counter_closed += 2
                        self.eye_counter_open = 0
                    elif weak_closed:
                        self.eye_counter_closed += 1
                        self.eye_counter_open = 0
                    else:
                        self.eye_counter_closed = 0
                        self.eye_counter_open += 1

                    perclos = self._perclos()
                    trigger = self.eye_counter_closed >= self.EAR_CONSEC_FRAMES_ON or (
                        perclos >= 0.62 and len(self.eye_closed_window) >= 18 and not head_down
                    )

                    if trigger and eye_signal_conf >= 0.62:
                        self.is_drowsy_state = True
                        self.eye_counter_open = 0
                else:
                    eye_reopen_ok = True
                    if eye_roi_open_ratio is not None and side_mode == "both":
                        eye_reopen_ok = eye_roi_open_ratio > eye_ratio_th_off

                    if ear is not None and ear > threshold_off and eye_reopen_ok:
                        self.eye_counter_open += 1
                        self.eye_counter_closed = 0
                        if self.eye_counter_open >= self.EAR_CONSEC_FRAMES_OFF:
                            self.is_drowsy_state = False
                            self.eye_counter_closed = 0
                    else:
                        self.eye_counter_open = 0

        if head_down and forward_working:
            self.is_drowsy_state = False

        if head_down and eye_signal_conf < 0.60 and self.head_down_hold_counter >= 4:
            self.is_drowsy_state = False

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
            "ear_threshold_on": round(float(self._get_threshold_on()), 3),
            "ear_threshold_off": round(float(self._get_threshold_off()), 3),
            "face_ok": face_ok,
            "head_down": head_down,
            "mar": None if mar is None else round(float(mar), 3),
            "yawning": yawning,
            "forward_working": forward_working,
            "eyes_state": eyes_state,
            "eye_signal_conf": round(float(eye_signal_conf), 3),
            "perclos": round(float(self._perclos()), 3),
        }

    def close(self):
        try:
            if self.face_mesh is not None:
                self.face_mesh.close()
        except Exception:
            pass
        try:
            if self.face_detector is not None:
                self.face_detector.close()
        except Exception:
            pass