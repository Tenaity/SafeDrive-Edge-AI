# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
import os
import time
from typing import Any, cast, Sequence
from pathlib import Path

from driver_state.face_detector import FaceDetector
from driver_state.face_landmark import FaceLandmark
from driver_state.head_pose import HeadPoseEstimator
from driver_state.state_logic import DriverStateLogic
from utils.types import DriverStateOut

Frame = Any
Box = Sequence[float] | None
Point = tuple[float, float] | None


class DriverStatePipeline:
    """
    Pipeline driver-state mới:
    - có cache để giảm lag
    - face detector chỉ chạy theo nhịp, không chạy dày mọi frame
    - ưu tiên ROI theo person_box
    """

    def __init__(self):
        _base = Path(__file__).resolve().parent.parent
        _model_rel = os.getenv(
            "MEDIAPIPE_FACE_LANDMARKER_MODEL",
            "mediapipe_models/models/face_landmarker_v2_with_blendshapes.task",
        )
        _model_path = str(_base / _model_rel) if not os.path.isabs(_model_rel) else _model_rel

        self.face_detector = FaceDetector()
        self.face_landmark = FaceLandmark(model_path=_model_path)
        self.head_pose = HeadPoseEstimator()
        self.state_logic = DriverStateLogic()

        self.last_run_time = 0.0
        self.RUN_INTERVAL_SEC = 0.28

        self.last_out: DriverStateOut = {
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

    def run(
        self,
        frame: Frame,
        person_box: Box = None,
        left_shoulder: Point = None,
        right_shoulder: Point = None,
        left_hip: Point = None,
        right_hip: Point = None,
    ) -> DriverStateOut:
        now = time.time()
        if now - self.last_run_time < self.RUN_INTERVAL_SEC:
            return self.last_out
        self.last_run_time = now

        faces: list[dict[str, Any]] = cast(Any, self.face_detector.detect)(frame, person_box=person_box)

        if faces:
            face_box: Box = cast(Any, faces[0].get("bbox"))
            landmark_out: dict[str, Any] = cast(Any, self.face_landmark.predict)(frame, face_box)
            face_out: dict[str, Any] = cast(Any, faces[0])
        else:
            face_out = {"bbox": None, "score": 0.0}
            landmark_out: dict[str, Any] = {
                "landmarks": [],
                "left_eye": None,
                "right_eye": None,
                "nose": None,
                "mouth": None,
                "face_center": None,
                "ok": False,
            }

        pose_out: dict[str, Any] = cast(Any, self.head_pose.estimate)(landmark_out)
        out: DriverStateOut = cast(DriverStateOut, cast(Any, self.state_logic.update)(face_out, landmark_out, pose_out))

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