# pyright: reportMissingImports=false
import os
import time
import numpy as np
import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _MP_OK = True
except Exception:
    _MP_OK = False
    mp = None
    _mp_python = None
    _mp_vision = None

# Blendshape names
_BLINK_LEFT = "eyeBlinkLeft"
_BLINK_RIGHT = "eyeBlinkRight"

# Landmark indices (MediaPipe 468/478-point model)
_IDX_NOSE_TIP = 4
_IDX_LEFT_EYE_CTR = 159    # upper left eyelid center
_IDX_RIGHT_EYE_CTR = 386   # upper right eyelid center
_IDX_MOUTH_TOP = 13        # inner upper lip
_IDX_MOUTH_BOT = 14        # inner lower lip
_IDX_MOUTH_L = 78          # mouth left corner
_IDX_MOUTH_R = 308         # mouth right corner

_EMPTY = {
    "landmarks": [],
    "left_eye": None,
    "right_eye": None,
    "nose": None,
    "mouth": None,
    "mouth_top": None,
    "mouth_bot": None,
    "mouth_l": None,
    "mouth_r": None,
    "face_center": None,
    "eye_blink_left": 0.0,
    "eye_blink_right": 0.0,
    "w": 0,
    "h": 0,
    "ok": False,
}


class FaceLandmark:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._landmarker = None
        self._ts_ms = 0
        self._ok = False

        if not _MP_OK:
            print("[FaceLandmark] MediaPipe not available")
            return
        if not model_path:
            print("[FaceLandmark] No model path provided")
            return
        if not os.path.isfile(model_path):
            print(f"[FaceLandmark] Model not found: {model_path}")
            return

        self._init_landmarker(model_path)

    def _init_landmarker(self, model_path: str):
        try:
            base_opts = _mp_python.BaseOptions(model_asset_path=model_path)  # type: ignore
            opts = _mp_vision.FaceLandmarkerOptions(  # type: ignore
                base_options=base_opts,
                running_mode=_mp_vision.RunningMode.VIDEO,  # type: ignore
                output_face_blendshapes=True,
                num_faces=1,
                min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self._landmarker = _mp_vision.FaceLandmarker.create_from_options(opts)  # type: ignore
            self._ok = True
            print(f"[FaceLandmark] Loaded: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"[FaceLandmark] Init failed: {e}")
            self._landmarker = None
            self._ok = False

    def predict(self, frame: np.ndarray, face_box=None) -> dict:
        if not self._ok or self._landmarker is None or frame is None:
            return dict(_EMPTY)

        try:
            work = frame
            off_x, off_y = 0, 0

            if face_box is not None and len(face_box) == 4:
                fh, fw = frame.shape[:2]
                x1 = max(0, int(face_box[0]))
                y1 = max(0, int(face_box[1]))
                x2 = min(fw, int(face_box[2]))
                y2 = min(fh, int(face_box[3]))
                if x2 > x1 and y2 > y1:
                    work = frame[y1:y2, x1:x2]
                    off_x, off_y = x1, y1

            h, w = work.shape[:2]
            rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  # type: ignore

            # Strictly monotonically increasing timestamp in ms
            self._ts_ms = max(self._ts_ms + 1, int(time.time() * 1000))
            result = self._landmarker.detect_for_video(mp_img, self._ts_ms)

            if not result.face_landmarks:
                return dict(_EMPTY)

            lm = result.face_landmarks[0]

            def to_px(idx):
                pt = lm[idx]
                return (pt.x * w + off_x, pt.y * h + off_y)

            left_eye = to_px(_IDX_LEFT_EYE_CTR)
            right_eye = to_px(_IDX_RIGHT_EYE_CTR)
            nose = to_px(_IDX_NOSE_TIP)
            mouth_top = to_px(_IDX_MOUTH_TOP)
            mouth_bot = to_px(_IDX_MOUTH_BOT)
            mouth_l = to_px(_IDX_MOUTH_L)
            mouth_r = to_px(_IDX_MOUTH_R)
            mouth_center = (
                (mouth_top[0] + mouth_bot[0]) / 2.0,
                (mouth_top[1] + mouth_bot[1]) / 2.0,
            )
            face_center = (
                (left_eye[0] + right_eye[0]) / 2.0,
                (left_eye[1] + right_eye[1]) / 2.0,
            )

            blink_left = 0.0
            blink_right = 0.0
            if result.face_blendshapes:
                for cat in result.face_blendshapes[0]:
                    if cat.category_name == _BLINK_LEFT:
                        blink_left = float(cat.score)
                    elif cat.category_name == _BLINK_RIGHT:
                        blink_right = float(cat.score)

            return {
                "landmarks": lm,
                "left_eye": left_eye,
                "right_eye": right_eye,
                "nose": nose,
                "mouth": mouth_center,
                "mouth_top": mouth_top,
                "mouth_bot": mouth_bot,
                "mouth_l": mouth_l,
                "mouth_r": mouth_r,
                "face_center": face_center,
                "eye_blink_left": blink_left,
                "eye_blink_right": blink_right,
                "w": w,
                "h": h,
                "ok": True,
            }

        except Exception as e:
            print(f"[FaceLandmark] predict error: {e}")
            return dict(_EMPTY)

    def close(self):
        try:
            if self._landmarker is not None:
                self._landmarker.close()
        except Exception:
            pass
        self._landmarker = None
