import cv2
import time
import numpy as np
import mediapipe as mp

class DriverStatePipeline:
    """
    DriverStatePipeline:
    - EAR for drowsiness
    - Head yaw for distraction
    - Temporal logic to avoid flicker
    """

    def __init__(self):
        # ---- Thresholds (tuned for vehicle / port environment) ----
        self.EAR_THRESHOLD = 0.22
        self.EAR_CONSEC_FRAMES = 20          # ~0.8s
        self.YAW_THRESHOLD_ON = 20.0         # start distraction
        self.YAW_THRESHOLD_OFF = 12.0        # hysteresis
        self.DISTRACTION_TIME = 5.0          # seconds

        # ---- Internal state ----
        self.eye_counter = 0
        self.yaw_start_time = None
        self.distracted = False

        # ---- MediaPipe ----
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Camera approximation for head pose
        self.focal_length = 640
        self.camera_matrix = np.array([
            [self.focal_length, 0, 320],
            [0, self.focal_length, 240],
            [0, 0, 1]
        ], dtype="double")

        self.dist_coeffs = np.zeros((4, 1))

    # ---------- EAR ----------
    def _compute_ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    # ---------- Head Yaw ----------
    def _estimate_yaw(self, lm, shape):
        """
        Estimate head yaw angle using solvePnP.
        Return None if landmarks are insufficient or unstable.
        """
        h, w = shape[:2]

        REQUIRED_IDXS = [1, 152, 33, 263, 61, 291]

        # ---- SAFETY CHECK 1: đủ landmark ----
        try:
            for idx in REQUIRED_IDXS:
                _ = lm[idx]
        except Exception:
            return None

        # ---- Build image points ----
        try:
            image_pts = np.array(
                [
                    (lm[1].x * w, lm[1].y * h),     # Nose
                    (lm[152].x * w, lm[152].y * h), # Chin
                    (lm[33].x * w, lm[33].y * h),   # Left eye
                    (lm[263].x * w, lm[263].y * h), # Right eye
                    (lm[61].x * w, lm[61].y * h),   # Left mouth
                    (lm[291].x * w, lm[291].y * h), # Right mouth
                ],
                dtype="double",
            )
        except Exception:
            return None

        # ---- 3D model points (generic face) ----
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


    def run(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        ear = None
        yaw = None
        drowsy = False

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            # ---- EAR ----
            left_eye_idx = [33,160,158,133,153,144]
            right_eye_idx = [362,385,387,263,373,380]

            left_eye = np.array([(lm[i].x*w, lm[i].y*h) for i in left_eye_idx])
            right_eye = np.array([(lm[i].x*w, lm[i].y*h) for i in right_eye_idx])

            ear = (self._compute_ear(left_eye) + self._compute_ear(right_eye)) / 2

            if ear < self.EAR_THRESHOLD:
                self.eye_counter += 1
                if self.eye_counter >= self.EAR_CONSEC_FRAMES:
                    drowsy = True
            else:
                self.eye_counter = 0

            # ---- YAW ----
            yaw = self._estimate_yaw(lm, frame.shape)

            if yaw is not None:
                if abs(yaw) > self.YAW_THRESHOLD_ON:
                    if self.yaw_start_time is None:
                        self.yaw_start_time = time.time()
                    elif time.time() - self.yaw_start_time >= self.DISTRACTION_TIME:
                        self.distracted = True
                elif abs(yaw) < self.YAW_THRESHOLD_OFF:
                    self.yaw_start_time = None
                    self.distracted = False

        return {
            "ear": None if ear is None else round(ear, 3),
            "yaw": None if yaw is None else round(yaw, 1),
            "drowsy": drowsy,
            "distracted": self.distracted
        }
