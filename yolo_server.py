import os
from typing import List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

app = FastAPI(title="YOLO Detect API")

model = YOLO(MODEL_PATH)

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_PATH}

@app.post("/detect")
async def detect(image: UploadFile = File(...)) -> Dict[str, Any]:
    raw = await image.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"ok": False, "error": "invalid image", "dets": []}

    results = model(frame, verbose=False)
    dets: List[Dict[str, Any]] = []

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        cls = boxes.cls.cpu().numpy() if boxes.cls is not None else []

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            dets.append({
                "cls": int(cls[i]),
                "conf": float(conf[i]),
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            })

    return {"ok": True, "dets": dets}

if __name__ == "__main__":
    uvicorn.run("yolo_server:app", host="127.0.0.1", port=8000, reload=False)