from typing import TypedDict


class HandsOut(TypedDict, total=False):
    hands_present: bool
    no_hand: bool
    hand_centers: list
    left_wrist: tuple | None
    right_wrist: tuple | None
    left_elbow: tuple | None
    right_elbow: tuple | None
    left_shoulder: tuple | None
    right_shoulder: tuple | None
    left_hip: tuple | None
    right_hip: tuple | None
    left_knee: tuple | None
    right_knee: tuple | None
    left_thigh_center: tuple | None
    right_thigh_center: tuple | None


class VisionOut(TypedDict, total=False):
    phone: bool
    dets: list
    persons: list
    phones: list


class DriverStateOut(TypedDict, total=False):
    ear: float | None
    yaw: float | None
    drowsy: bool
    distracted: bool
    nose_point: tuple | None
    face_center: tuple | None
    left_eye_center: tuple | None
    right_eye_center: tuple | None
    baseline_ear: float | None
    ear_threshold_on: float
    ear_threshold_off: float
    face_ok: bool
    head_down: bool
    mar: float | None
    yawning: bool
    forward_working: bool
    eyes_state: str
    eye_signal_conf: float
    perclos: float


class PhoneContextOut(TypedDict, total=False):
    phone_contexts: list
    lap_zone_box: list | None
    mid_leg_zone_box: list | None
    left_leg_zone_box: list | None
    right_leg_zone_box: list | None
    lap_attention: bool
    left_side_attention: bool
    right_side_attention: bool
    lap_attention_strength: float
    left_side_attention_strength: float
    right_side_attention_strength: float
    head_down: bool
    body_box: list | None
    face_center: tuple | None


class PhoneUsageOut(TypedDict, total=False):
    phone_using: bool
    best_phone_box: list | None
    score: int
    source: str | None
    temporal_track_len: int


DEFAULT_HANDS_OUT: HandsOut = {
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

DEFAULT_VISION_OUT: VisionOut = {
    "phone": False,
    "dets": [],
    "persons": [],
    "phones": [],
}

DEFAULT_DRIVER_OUT: DriverStateOut = {
    "ear": None,
    "yaw": None,
    "drowsy": False,
    "distracted": False,
    "nose_point": None,
    "face_center": None,
    "left_eye_center": None,
    "right_eye_center": None,
    "baseline_ear": None,
    "ear_threshold_on": 0.0,
    "ear_threshold_off": 0.0,
    "face_ok": False,
    "head_down": False,
    "mar": None,
    "yawning": False,
    "forward_working": False,
    "eyes_state": "unknown",
    "eye_signal_conf": 0.0,
    "perclos": 0.0,
}

DEFAULT_CONTEXT_OUT: PhoneContextOut = {
    "phone_contexts": [],
    "lap_zone_box": None,
    "mid_leg_zone_box": None,
    "left_leg_zone_box": None,
    "right_leg_zone_box": None,
    "lap_attention": False,
    "left_side_attention": False,
    "right_side_attention": False,
    "lap_attention_strength": 0.0,
    "left_side_attention_strength": 0.0,
    "right_side_attention_strength": 0.0,
    "head_down": False,
    "body_box": None,
    "face_center": None,
}

DEFAULT_PHONE_USAGE_OUT: PhoneUsageOut = {
    "phone_using": False,
    "best_phone_box": None,
    "score": -999,
    "source": None,
    "temporal_track_len": 0,
}
