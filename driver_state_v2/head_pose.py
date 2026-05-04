class HeadPoseEstimatorV2:
    def __init__(self):
        pass

    def estimate(self, landmarks_out):
        return {
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "head_down": False,
            "looking_away": False,
        }