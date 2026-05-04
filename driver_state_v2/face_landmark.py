class FaceLandmarkV2:
    def __init__(self, model_path=None):
        self.model_path = model_path

    def predict(self, frame, face_box):
        return {
            "landmarks": [],
            "left_eye": None,
            "right_eye": None,
            "nose": None,
            "mouth": None,
            "face_center": None,
            "ok": False,
        }

    def close(self):
        pass