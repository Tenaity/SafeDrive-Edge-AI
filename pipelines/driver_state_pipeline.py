import time
import cv2
import numpy as np
import mediapipe as mp


class DriverStatePipeline:
    """
    DriverStatePipeline: MediaPipe FaceMesh -> EAR-based drowsiness only.
    - Computes EAR from eye landmarks
    - Drowsy=True if EAR below threshold for N consecutive frames
    - Distraction/yaw logic disabled (always False)
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

        # --- EAR config ---
        # Tune these two values
        self.EAR_THRESHOLD = 0.25
        self.EAR_CONSEC_FRAMES = 20  # adjust based on FPS

        # Counters / state
        self.eye_counter = 0

        # Indices for FaceMesh eyes (common set)
        # You may already have these in your codebase; keep yours if different.
        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [263, 387, 385, 362, 380, 373]

    @staticmethod
    def _euclid(p1, p2):
        return float(np.linalg.norm(p1 - p2))

    def _compute_ear(self, eye_pts: np.ndarray):
        """
        eye_pts: 6x2 array in order:
          [p1, p2, p3, p4, p5, p6]
        EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
        """
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

    def run(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        ear = None
        drowsy = False

        # Visualization points
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

            # ---- Nose point (optional, for overlay) ----
            try:
                nose_point = (int(lm[1].x * w), int(lm[1].y * h))
            except Exception:
                nose_point = None

        # yaw/distracted disabled
        yaw = None
        distracted = False

        return {
            "ear": None if ear is None else round(float(ear), 3),
            "yaw": yaw,
            "drowsy": drowsy,
            "distracted": distracted,
            "nose_point": nose_point,
            "left_eye_center": left_eye_center,
            "right_eye_center": right_eye_center,
        }