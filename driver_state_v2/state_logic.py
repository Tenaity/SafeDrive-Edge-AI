class DriverStateLogicV2:
    def __init__(self):
        pass

    def update(self, face_out, landmark_out, pose_out):
        bbox = face_out.get("bbox") if isinstance(face_out, dict) else None
        score = face_out.get("score", 0.0) if isinstance(face_out, dict) else 0.0

        face_ok = bool(bbox is not None and len(bbox) == 4 and score > 0.0)

        face_center = None
        if face_ok:
            x1, y1, x2, y2 = bbox
            face_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        return {
            "face_ok": face_ok,
            "face_box": bbox,
            "face_center": face_center,
            "left_eye_center": landmark_out.get("left_eye"),
            "right_eye_center": landmark_out.get("right_eye"),
            "nose_point": landmark_out.get("nose"),
            "yaw": pose_out.get("yaw"),
            "pitch": pose_out.get("pitch"),
            "roll": pose_out.get("roll"),
            "head_down": pose_out.get("head_down", False),
            "distracted": pose_out.get("looking_away", False),
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