import time

from driver_state_v2.face_detector import FaceDetectorV2
from driver_state_v2.face_landmark import FaceLandmarkV2
from driver_state_v2.head_pose import HeadPoseEstimatorV2
from driver_state_v2.state_logic import DriverStateLogicV2


class DriverStatePipelineV2:
    """
    Pipeline driver-state mới:
    - có cache để giảm lag
    - face detector chỉ chạy theo nhịp, không chạy dày mọi frame
    - ưu tiên ROI theo person_box
    """

    def __init__(self):
        self.face_detector = FaceDetectorV2()
        self.face_landmark = FaceLandmarkV2()
        self.head_pose = HeadPoseEstimatorV2()
        self.state_logic = DriverStateLogicV2()

        self.last_run_time = 0.0
        self.RUN_INTERVAL_SEC = 0.28

        self.last_out = {
            "face_ok": False,
            "face_box": None,
            "face_center": None,
            "left_eye_center": None,
            "right_eye_center": None,
            "nose_point": None,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "head_down": False,
            "distracted": False,
            "drowsy": False,
            "yawning": False,
            "ear": None,
            "mar": None,
            "baseline_ear": None,
            "ear_threshold_on": 0.0,
            "ear_threshold_off": 0.0,
            "forward_working": False,
            "eyes_state": "unknown",
            "eye_signal_conf": 0.0,
            "perclos": 0.0,
        }

    def run(self, frame, person_box=None, left_shoulder=None, right_shoulder=None, left_hip=None, right_hip=None):
        now = time.time()
        if now - self.last_run_time < self.RUN_INTERVAL_SEC:
            return self.last_out
        self.last_run_time = now

        faces = self.face_detector.detect(frame, person_box=person_box)

        if faces:
            face_box = faces[0].get("bbox")
            landmark_out = self.face_landmark.predict(frame, face_box)
            face_out = faces[0]
        else:
            face_out = {"bbox": None, "score": 0.0}
            landmark_out = {
                "landmarks": [],
                "left_eye": None,
                "right_eye": None,
                "nose": None,
                "mouth": None,
                "face_center": None,
                "ok": False,
            }

        pose_out = self.head_pose.estimate(landmark_out)
        out = self.state_logic.update(face_out, landmark_out, pose_out)

        self.last_out = out
        return out

    def close(self):
        try:
            self.face_detector.close()
        except Exception:
            pass

        try:
            self.face_landmark.close()
        except Exception:
            pass