from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import requests


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

HEALTH_URL = "http://127.0.0.1:8000/health"
DETECT_URL = "http://127.0.0.1:8000/detect"


def post_with_field(image_path: Path, field_name: str) -> None:
    print()
    print("=" * 80)
    print(f"TEST /detect field='{field_name}'")
    print("=" * 80)

    try:
        with open(image_path, "rb") as f:
            files = {field_name: ("frame.jpg", f, "image/jpeg")}
            r = requests.post(DETECT_URL, files=files, timeout=10)

        print("STATUS:", r.status_code)
        print("TEXT:")
        print(r.text[:5000])

        out_json = LOGS / f"detect_response_{field_name}.json"
        try:
            data = r.json()
            out_json.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("SAVED:", out_json)
        except Exception as e:
            print("JSON ERROR:", repr(e))

    except Exception as e:
        print("POST ERROR:", repr(e))


def main() -> None:
    print("TEST YOLO PHONE DETECT")
    print("HEALTH:", requests.get(HEALTH_URL, timeout=3).text)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("CAMERA ERROR: cannot open camera 0")
        return

    frame = None
    for _ in range(30):
        ok, img = cap.read()
        if ok and img is not None:
            frame = img
            break
        time.sleep(0.05)

    cap.release()

    if frame is None:
        print("CAMERA ERROR: cannot read frame")
        return

    image_path = LOGS / "detect_test_frame.jpg"
    cv2.imwrite(str(image_path), frame)
    print("SAVED FRAME:", image_path)

    # Thử 3 tên field thường gặp.
    for field in ("file", "image", "frame"):
        post_with_field(image_path, field)


if __name__ == "__main__":
    main()
