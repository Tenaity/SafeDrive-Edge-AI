import os
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolo11s.pt")

CLS_PERSON = 0
CLS_PHONE = 67

CONF_MIN_PERSON = 0.42
CONF_MIN_PHONE = 0.55

MIN_PHONE_AREA_RATIO = 0.0008
MAX_PHONE_AREA_RATIO = 0.10
MIN_PHONE_ASPECT = 0.18
MAX_PHONE_ASPECT = 6.00

# resize inference cho nhanh
INFER_LONG_SIDE = 640

app = FastAPI(title="YOLO Detect API")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)


def resize_keep_ratio(frame, long_side=640):
    h, w = frame.shape[:2]
    if max(h, w) <= long_side:
        return frame, 1.0

    scale = long_side / float(max(h, w))
    nw = int(w * scale)
    nh = int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def scale_box_back(box, scale):
    if scale <= 0:
        return box
    x1, y1, x2, y2 = map(float, box)
    return [x1 / scale, y1 / scale, x2 / scale, y2 / scale]


def valid_phone_box(xyxy, frame_w: int, frame_h: int) -> bool:
    try:
        x1, y1, x2, y2 = map(float, xyxy)
    except Exception:
        return False

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 2 or bh <= 2:
        return False

    frame_area = float(frame_w * frame_h)
    if frame_area <= 1:
        return False

    area_ratio = (bw * bh) / frame_area
    if area_ratio < MIN_PHONE_AREA_RATIO or area_ratio > MAX_PHONE_AREA_RATIO:
        return False

    aspect = bw / bh if bh > 1e-6 else 999.0
    if aspect < MIN_PHONE_ASPECT or aspect > MAX_PHONE_ASPECT:
        return False

    return True


def box_area(box) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_driver_person(person_boxes, frame_w, frame_h) -> Optional[List[float]]:
    if not person_boxes:
        return None

    cx_frame = frame_w / 2.0
    cy_frame = frame_h / 2.0

    best_score = -1e18
    best_box = None

    for box in person_boxes:
        x1, y1, x2, y2 = map(float, box)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = box_area(box)
        dist2 = (cx - cx_frame) ** 2 + (cy - cy_frame) ** 2

        score = area - 1.8 * dist2
        if score > best_score:
            best_score = score
            best_box = box

    return best_box


@app.on_event("startup")
def startup_event():
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        model(dummy, verbose=False, classes=[CLS_PERSON, CLS_PHONE], imgsz=640)
        print(f"[YOLO SERVER] Warmup done. Model: {MODEL_PATH}")
    except Exception as e:
        print(f"[YOLO SERVER] Warmup failed: {e}")


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_PATH}


@app.post("/detect")
async def detect(image: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        raw = await image.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"ok": False, "error": "invalid image", "dets": [], "persons": [], "phones": []}

        frame_h, frame_w = frame.shape[:2]

        # resize xuống để infer nhanh hơn
        infer_frame, scale = resize_keep_ratio(frame, INFER_LONG_SIDE)
        infer_h, infer_w = infer_frame.shape[:2]

        results = model(
            infer_frame,
            verbose=False,
            classes=[CLS_PERSON, CLS_PHONE],
            conf=min(CONF_MIN_PERSON, CONF_MIN_PHONE),
            imgsz=INFER_LONG_SIDE
        )

        persons = []
        phones = []
        dets = []

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
            conf = boxes.conf.cpu().numpy() if boxes.conf is not None else []
            cls = boxes.cls.cpu().numpy() if boxes.cls is not None else []

            for i in range(len(xyxy)):
                c = int(cls[i])
                score = float(conf[i])
                box_small = [float(v) for v in xyxy[i].tolist()]
                box = scale_box_back(box_small, scale)

                if c == CLS_PERSON and score >= CONF_MIN_PERSON:
                    persons.append({"cls": c, "conf": score, "xyxy": box})

                elif c == CLS_PHONE and score >= CONF_MIN_PHONE:
                    if valid_phone_box(box, frame_w, frame_h):
                        phones.append({"cls": c, "conf": score, "xyxy": box})

        driver = select_driver_person([p["xyxy"] for p in persons], frame_w, frame_h)

        persons_out = []
        if driver is not None:
            persons_out.append({"cls": CLS_PERSON, "conf": 1.0, "xyxy": driver})
            dets.append({"cls": CLS_PERSON, "conf": 1.0, "xyxy": driver})

        for ph in phones:
            dets.append(ph)

        return {
            "ok": True,
            "dets": dets,
            "persons": persons_out,
            "phones": phones
        }

    except Exception as e:
        return {"ok": False, "error": str(e), "dets": [], "persons": [], "phones": []}


if __name__ == "__main__":
    uvicorn.run("yolo_server:app", host="127.0.0.1", port=8000, reload=False)