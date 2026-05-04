# pyright: reportMissingImports=false
from pathlib import Path
import os
import cv2

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from openvino.runtime import Core
except Exception:
    Core = None


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name, "1" if default else "0").strip().lower()
    return val in {"1", "true", "yes", "on"}


class FaceDetector:
    """
    Offline-safe face detector:
    - chỉ load model local
    - ưu tiên OpenVINO IR
    - fallback ONNX Runtime
    - cuối cùng fallback Haar
    - có tối ưu ROI + resize để giảm lag
    """

    def __init__(self, model_path=None):
        self.backend = "none"
        self.score_threshold = env_float("DRIVER_STATE_FACE_SCORE_THRESHOLD", 0.50)
        self.allow_haar = env_bool("DRIVER_STATE_ALLOW_HAAR_FALLBACK", True)
        self.prefer_backend = env_str("DRIVER_STATE_BACKEND", "auto").lower()
        self.ov_device = env_str("DRIVER_STATE_OPENVINO_DEVICE", "CPU")

        _base = Path(__file__).resolve().parent.parent
        model_dir = _base / "models" / "driver_state" / "face_detector"
        self.xml_path = model_dir / "face_detector.xml"
        self.bin_path = model_dir / "face_detector.bin"
        self.onnx_path = model_dir / "face_detector.onnx"

        self.ov_core = None
        self.ov_compiled_model = None
        self.ov_input = None
        self.ov_output = None

        self.ort_session = None
        self.ort_input_name = None

        self.haar = None

        # ===== PERF TUNING =====
        self.HAAR_ROI_EXPAND_X = 0.18
        self.HAAR_ROI_EXPAND_Y_TOP = 0.20
        self.HAAR_ROI_EXPAND_Y_BOTTOM = 0.10
        self.HAAR_DETECT_LONG_SIDE = 480
        self.HAAR_MIN_FACE = 48

        self._init_backend()

    def _init_backend(self):
        order = []
        if self.prefer_backend == "openvino":
            order = ["openvino", "onnx", "haar"]
        elif self.prefer_backend == "onnx":
            order = ["onnx", "openvino", "haar"]
        elif self.prefer_backend == "haar":
            order = ["haar"]
        else:
            order = ["openvino", "onnx", "haar"]

        for item in order:
            if item == "openvino" and self._try_init_openvino():
                return
            if item == "onnx" and self._try_init_onnx():
                return
            if item == "haar" and self._try_init_haar():
                return

        print("[FaceDetector] No backend available")

    def _try_init_openvino(self):
        if Core is None:
            return False
        if not self.xml_path.exists() or not self.bin_path.exists():
            return False
        try:
            self.ov_core = Core()
            model = self.ov_core.read_model(model=str(self.xml_path))
            self.ov_compiled_model = self.ov_core.compile_model(model=model, device_name=self.ov_device)
            self.ov_input = self.ov_compiled_model.input(0)
            self.ov_output = self.ov_compiled_model.output(0)
            self.backend = "openvino"
            print(f"[FaceDetector] Using OpenVINO IR: {self.xml_path.name} on {self.ov_device}")
            return True
        except Exception as e:
            print(f"[FaceDetector] OpenVINO init failed: {e}")
            self.ov_core = None
            self.ov_compiled_model = None
            return False

    def _try_init_onnx(self):
        if ort is None:
            return False
        if not self.onnx_path.exists():
            return False
        try:
            self.ort_session = ort.InferenceSession(
                str(self.onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self.ort_input_name = self.ort_session.get_inputs()[0].name
            self.backend = "onnx"
            print(f"[FaceDetector] Using ONNX Runtime: {self.onnx_path.name}")
            return True
        except Exception as e:
            print(f"[FaceDetector] ONNX init failed: {e}")
            self.ort_session = None
            self.ort_input_name = None
            return False

    def _try_init_haar(self):
        if not self.allow_haar:
            return False
        try:
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
            )
            if self.haar.empty():
                return False
            self.backend = "haar"
            print("[FaceDetector] Using OpenCV Haar fallback")
            return True
        except Exception as e:
            print(f"[FaceDetector] Haar init failed: {e}")
            self.haar = None
            return False

    def _resize_keep_ratio(self, frame, long_side):
        h, w = frame.shape[:2]
        if max(h, w) <= long_side:
            return frame, 1.0
        scale = long_side / float(max(h, w))
        nw = int(w * scale)
        nh = int(h * scale)
        out = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return out, scale

    def _build_driver_roi(self, frame, person_box):
        h, w = frame.shape[:2]

        if person_box is None or len(person_box) != 4:
            return frame, (0, 0)

        try:
            x1, y1, x2, y2 = map(float, person_box)
        except Exception:
            return frame, (0, 0)

        pw = x2 - x1
        ph = y2 - y1
        if pw <= 1 or ph <= 1:
            return frame, (0, 0)

        rx1 = max(0, int(x1 - pw * self.HAAR_ROI_EXPAND_X))
        rx2 = min(w, int(x2 + pw * self.HAAR_ROI_EXPAND_X))
        ry1 = max(0, int(y1 - ph * self.HAAR_ROI_EXPAND_Y_TOP))
        ry2 = min(h, int(y1 + ph * 0.65 + ph * self.HAAR_ROI_EXPAND_Y_BOTTOM))

        if rx2 <= rx1 or ry2 <= ry1:
            return frame, (0, 0)

        roi = frame[ry1:ry2, rx1:rx2]
        if roi is None or roi.size == 0:
            return frame, (0, 0)

        return roi, (rx1, ry1)

    def _detect_haar(self, frame, person_box=None):
        roi, (off_x, off_y) = self._build_driver_roi(frame, person_box)
        small, scale = self._resize_keep_ratio(roi, self.HAAR_DETECT_LONG_SIDE)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        min_face = max(32, int(self.HAAR_MIN_FACE * scale))

        faces = self.haar.detectMultiScale(  # type: ignore
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face, min_face),
        )

        out = []
        inv = 1.0 / scale if scale > 0 else 1.0

        for (x, y, w, h) in faces:
            x1 = int(x * inv + off_x)
            y1 = int(y * inv + off_y)
            x2 = int((x + w) * inv + off_x)
            y2 = int((y + h) * inv + off_y)
            out.append({
                "bbox": [x1, y1, x2, y2],
                "score": 1.0,
            })

        out.sort(
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True,
        )
        return out

    def _detect_stub_openvino(self, frame, person_box=None):
        return []

    def _detect_stub_onnx(self, frame, person_box=None):
        return []

    def detect(self, frame, person_box=None):
        if frame is None:
            return []

        if self.backend == "openvino":
            return self._detect_stub_openvino(frame, person_box)

        if self.backend == "onnx":
            return self._detect_stub_onnx(frame, person_box)

        if self.backend == "haar" and self.haar is not None:
            return self._detect_haar(frame, person_box)

        return []

    def close(self):
        self.ov_core = None
        self.ov_compiled_model = None
        self.ort_session = None
        self.haar = None