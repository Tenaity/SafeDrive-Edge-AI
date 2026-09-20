from pathlib import Path
import sys
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.driver_state_pipeline import DriverStatePipeline


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Không mở được camera")

    pipe = DriverStatePipeline()
    backend = getattr(pipe.face_detector, "backend", "unknown")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            out = pipe.run(frame)

            face_box = out.get("face_box")
            if face_box is not None:
                x1, y1, x2, y2 = map(int, face_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text1 = f"backend: {backend}"
            text2 = f"face_ok: {out.get('face_ok', False)}"
            text3 = f"yaw: {out.get('yaw', 0.0)}"
            text4 = f"head_down: {out.get('head_down', False)}"
            text5 = f"distracted: {out.get('distracted', False)}"

            cv2.putText(frame, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, text3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, text4, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, text5, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("driver_state_test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipe.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()