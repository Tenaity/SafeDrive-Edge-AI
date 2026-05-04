import math

from driver_state.utils import euclid

# Landmark indices
_IDX_NOSE_TIP = 4
_IDX_CHIN = 152
_IDX_FOREHEAD = 10
_IDX_LEFT_EYE_OUT = 33      # outer corner of left eye
_IDX_RIGHT_EYE_OUT = 263    # outer corner of right eye
_IDX_LEFT_EYE_IN = 133      # inner corner of left eye
_IDX_RIGHT_EYE_IN = 362     # inner corner of right eye

_DEFAULT = {
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "head_down": False,
    "looking_away": False,
}


class HeadPoseEstimator:
    # head_down: nose-to-forehead ratio drops when head tilts down
    HEAD_DOWN_RATIO_THRESHOLD = 0.50
    # looking_away: left/right eye span asymmetry
    LOOKING_AWAY_ASYM_THRESHOLD = 0.22

    def __init__(self):
        pass

    def estimate(self, landmark_out: dict) -> dict:
        lm = landmark_out.get("landmarks")
        if not lm or len(lm) < 468:
            return dict(_DEFAULT)

        w = landmark_out.get("w", 640)
        h = landmark_out.get("h", 480)

        def px(idx):
            pt = lm[idx]
            return (pt.x * w, pt.y * h)

        nose = px(_IDX_NOSE_TIP)
        chin = px(_IDX_CHIN)
        forehead = px(_IDX_FOREHEAD)
        left_out = px(_IDX_LEFT_EYE_OUT)
        right_out = px(_IDX_RIGHT_EYE_OUT)
        left_in = px(_IDX_LEFT_EYE_IN)
        right_in = px(_IDX_RIGHT_EYE_IN)

        # --- HEAD DOWN ---
        # Ratio of (nose→forehead) / (nose→chin + nose→forehead)
        # When head tilts down, forehead moves away from nose faster than chin does
        nose_to_chin_y = chin[1] - nose[1]
        nose_to_forehead_y = nose[1] - forehead[1]
        total_y = nose_to_chin_y + nose_to_forehead_y
        if total_y > 1.0:
            down_ratio = nose_to_forehead_y / total_y
        else:
            down_ratio = 0.5
        head_down = bool(down_ratio < self.HEAD_DOWN_RATIO_THRESHOLD)

        # --- PITCH (degrees, positive = looking down) ---
        eye_y = (left_out[1] + right_out[1]) / 2.0
        face_height = abs(chin[1] - forehead[1]) if abs(chin[1] - forehead[1]) > 1.0 else 1.0
        pitch = (nose[1] - eye_y) / face_height * 90.0

        # --- YAW via eye-span asymmetry ---
        # When turning right: left eye span shrinks, right eye span grows
        left_span = euclid(left_out, left_in)
        right_span = euclid(right_out, right_in)
        total_span = left_span + right_span
        if total_span > 1.0:
            asym = (left_span - right_span) / total_span
        else:
            asym = 0.0
        yaw = asym * 90.0
        looking_away = bool(abs(asym) > self.LOOKING_AWAY_ASYM_THRESHOLD)

        # --- ROLL: angle of eye line ---
        eye_dx = right_out[0] - left_out[0]
        eye_dy = right_out[1] - left_out[1]
        roll = math.degrees(math.atan2(eye_dy, eye_dx))

        return {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "head_down": head_down,
            "looking_away": looking_away,
        }
