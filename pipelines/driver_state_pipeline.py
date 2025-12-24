import cv2
import time
import numpy as np
import mediapipe as mp


class DriverStatePipeline:
    """
    DriverStatePipeline:
    - EAR (Eye Aspect Ratio) for drowsiness
    - Head yaw for distraction (temporal + hysteresis)
    - Exposes geometry points for visualization on main.py:
        - nose_point
        - left_eye_center
        - right_eye_center
    - solvePnP guarded to avoid crashes when landmarks are unstable
    """

    def __init__(self):
        # ---- Thresholds (tuned for vehicle / port environment) ----
        self.EAR_THRESHOLD = 0.22
        self.EAR_CONSEC_FRAMES = 20          # ~0.8s at ~25 FPS (approx)
        self.YAW_THRESHOLD_ON = 20.0         # start counting distraction
        self.YAW_THRESHOLD_OFF = 12.0        # hysteresis to stop distraction
        self.DISTRACTION_TIME = 5.0          # seconds continuously

        # ---- Internal state ----
        self.eye_counter = 0
        self.yaw_start_time = None
        self.distracted = False

        # ---- MediaPipe FaceMesh ----
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Camera approximation for head pose (good enough for yaw visualization)
        self.focal_length = 640
        self.camera_matrix = np.array([
            [self.focal_length, 0, 320],
            [0, self.focal_length, 240],
            [0, 0, 1]
        ], dtype="double")
        self.dist_coeffs = np.zeros((4, 1))

        # Landmark indices used for EAR (6 points each eye)
        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [362, 385, 387, 263, 373, 380]

    # ---------- EAR ----------
    def _compute_ear(self, eye_points: np.ndarray) -> float:
        """
        eye_points shape: (6, 2)
        """
        # Guard against degenerate cases
        if eye_points is None or len(eye_points) < 6:
            return None

        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])

        if C == 0:
            return None

        return (A + B) / (2.0 * C)

    # ---------- Head Yaw ----------
    def _estimate_yaw(self, lm, shape):
        """
        Estimate head yaw angle using solvePnP.
        Return None if landmarks are insufficient or unstable.
        """
        h, w = shape[:2]

        REQUIRED_IDXS = [1, 152, 33, 263, 61, 291]

        # ---- SAFETY CHECK 1: ensure indices exist ----
        try:
            for idx in REQUIRED_IDXS:
                _ = lm[idx]
        except Exception:
            return None

        # ---- Build image points (2D) ----
        try:
            image_pts = np.array(
                [
                    (lm[1].x * w, lm[1].y * h),       # Nose tip
                    (lm[152].x * w, lm[152].y * h),   # Chin
                    (lm[33].x * w, lm[33].y * h),     # Left eye corner
                    (lm[263].x * w, lm[263].y * h),   # Right eye corner
                    (lm[61].x * w, lm[61].y * h),     # Left mouth corner
                    (lm[291].x * w, lm[291].y * h),   # Right mouth corner
                ],
                dtype="double",
            )
        except Exception:
            return None

        # ---- 3D model points (generic face model) ----
        model_pts = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype="double",
        )

        # ---- SAFETY CHECK 2: solvePnP guarded ----
        try:
            ok, rvec, _ = cv2.solvePnP(
                model_pts,
                image_pts,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                return None

            R, _ = cv2.Rodrigues(rvec)
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            return float(yaw)

        except cv2.error:
            # OpenCV occasionally throws when geometry is unstable
            return None

    @staticmethod
    def _center_of_points(points: np.ndarray):
        """
        Return integer (x, y) center of Nx2 points.
        """
        if points is None or len(points) == 0:
            return None
        return (int(points[:, 0].mean()), int(points[:, 1].mean()))

    def run(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        ear = None
        yaw = None
        drowsy = False

        # Visualization points (default None)
        nose_point = None
        left_eye_center = None
        right_eye_center = None

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            # ---- EAR (eyes) ----
            left_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in self.left_eye_idx], dtype=np.float32)
            right_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in self.right_eye_idx], dtype=np.float32)

            left_eye_center = self._center_of_points(left_eye)
            right_eye_center = self._center_of_points(right_eye)

            ear_left = self._compute_ear(left_eye)
            ear_right = self._compute_ear(right_eye)

            if ear_left is not None and ear_right is not None:
                ear = (ear_left + ear_right) / 2.0

                # Drowsiness temporal logic: require consecutive frames below threshold
                if ear < self.EAR_THRESHOLD:
                    self.eye_counter += 1
                    if self.eye_counter >= self.EAR_CONSEC_FRAMES:
                        drowsy = True
                else:
                    self.eye_counter = 0
            else:
                # If EAR cannot be computed reliably this frame, do not update counters
                ear = None

            # ---- Nose point (for yaw vector drawing) ----
            try:
                nose_point = (int(lm[1].x * w), int(lm[1].y * h))
            except Exception:
                nose_point = None

            # ---- YAW (head pose) ----
            yaw = self._estimate_yaw(lm, frame.shape)

            # Distraction temporal logic with hysteresis
            if yaw is not None:
                if abs(yaw) > self.YAW_THRESHOLD_ON:
                    if self.yaw_start_time is None:
                        self.yaw_start_time = time.time()
                    elif time.time() - self.yaw_start_time >= self.DISTRACTION_TIME:
                        self.distracted = True
                elif abs(yaw) < self.YAW_THRESHOLD_OFF:
                    self.yaw_start_time = None
                    self.distracted = False
            # If yaw is None, keep previous distracted state and timer as-is
            # (prevents flicker when face tracking temporarily fails)

        return {
            # Core signals
            "ear": None if ear is None else round(float(ear), 3),
            "yaw": None if yaw is None else round(float(yaw), 1),
            "drowsy": drowsy,
            "distracted": self.distracted,

            # Visualization helpers for main.py
            "nose_point": nose_point,
            "left_eye_center": left_eye_center,
            "right_eye_center": right_eye_center,
        }
