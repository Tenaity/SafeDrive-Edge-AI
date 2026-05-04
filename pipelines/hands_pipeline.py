import time
import cv2
import mediapipe as mp
import numpy as np

from utils.types import HandsOut


class HandsPipeline:
    """
    HandsPipeline (Pose-based):
    - Use MediaPipe Pose instead of MediaPipe Hands
    - More robust when camera is mounted off-center
    - Returns wrists / elbows / shoulders / hips / knees / thigh centers
    - no_hand=True if no usable wrist visible for N seconds
    """

    def __init__(self, no_hand_time=3.0):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.no_hand_time = no_hand_time
        self.no_hand_since = None

        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26

        self.MIN_VIS = 0.45

    def _get_point(self, lm, idx, w, h):
        try:
            p = lm[idx]
            vis = getattr(p, "visibility", 0.0)
            if vis < self.MIN_VIS:
                return None
            return (int(p.x * w), int(p.y * h))
        except Exception:
            return None

    @staticmethod
    def _midpoint(p1, p2):
        if p1 is None or p2 is None:
            return None
        return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

    def run(self, frame: np.ndarray) -> HandsOut:
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return {
                "hands_present": False,
                "no_hand": False,
                "hand_centers": [],
                "left_wrist": None,
                "right_wrist": None,
                "left_elbow": None,
                "right_elbow": None,
                "left_shoulder": None,
                "right_shoulder": None,
                "left_hip": None,
                "right_hip": None,
                "left_knee": None,
                "right_knee": None,
                "left_thigh_center": None,
                "right_thigh_center": None,
            }

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        left_wrist = None
        right_wrist = None
        left_elbow = None
        right_elbow = None
        left_shoulder = None
        right_shoulder = None
        left_hip = None
        right_hip = None
        left_knee = None
        right_knee = None
        left_thigh_center = None
        right_thigh_center = None

        hand_centers = []

        if res.pose_landmarks:
            h, w = frame.shape[:2]
            lm = res.pose_landmarks.landmark

            left_shoulder = self._get_point(lm, self.LEFT_SHOULDER, w, h)
            right_shoulder = self._get_point(lm, self.RIGHT_SHOULDER, w, h)
            left_elbow = self._get_point(lm, self.LEFT_ELBOW, w, h)
            right_elbow = self._get_point(lm, self.RIGHT_ELBOW, w, h)
            left_wrist = self._get_point(lm, self.LEFT_WRIST, w, h)
            right_wrist = self._get_point(lm, self.RIGHT_WRIST, w, h)
            left_hip = self._get_point(lm, self.LEFT_HIP, w, h)
            right_hip = self._get_point(lm, self.RIGHT_HIP, w, h)
            left_knee = self._get_point(lm, self.LEFT_KNEE, w, h)
            right_knee = self._get_point(lm, self.RIGHT_KNEE, w, h)

            left_thigh_center = self._midpoint(left_hip, left_knee)
            right_thigh_center = self._midpoint(right_hip, right_knee)

            if left_wrist is not None:
                hand_centers.append(left_wrist)
            if right_wrist is not None:
                hand_centers.append(right_wrist)

        hands_present = len(hand_centers) > 0

        no_hand = False
        if not hands_present:
            if self.no_hand_since is None:
                self.no_hand_since = time.time()
            elif time.time() - self.no_hand_since >= self.no_hand_time:
                no_hand = True
        else:
            self.no_hand_since = None

        return {
            "hands_present": hands_present,
            "no_hand": no_hand,
            "hand_centers": hand_centers,
            "left_wrist": left_wrist,
            "right_wrist": right_wrist,
            "left_elbow": left_elbow,
            "right_elbow": right_elbow,
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "left_hip": left_hip,
            "right_hip": right_hip,
            "left_knee": left_knee,
            "right_knee": right_knee,
            "left_thigh_center": left_thigh_center,
            "right_thigh_center": right_thigh_center,
        }

    def close(self):
        try:
            if self.pose is not None:
                self.pose.close()
        except Exception:
            pass