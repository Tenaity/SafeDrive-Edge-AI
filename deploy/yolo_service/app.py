import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File

# Use ONNX with OpenCV DNN + TensorRT backend (no pycuda needed)
ONNX_PATH = os.environ.get("ONNX_PATH", "/models/yolov8n.onnx")
INPUT = 640
CONF = 0.25

app = FastAPI()

net = cv2.dnn.readNetFromONNX(ONNX_PATH)

# Backend/target: force CPU (avoid OpenCV DNN CUDA mismatch in this container)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def letterbox(im, new_shape=(640, 640), color=(114,114,114)):
    h, w = im.shape[:2]
    nh, nw = new_shape
    r = min(nw / w, nh / h)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = nw - new_unpad[0], nh - new_unpad[1]
    dw /= 2; dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"ok": False, "error": "bad image"}

    img, r, (padw, padh) = letterbox(frame, (INPUT, INPUT))
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (INPUT, INPUT), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()

    # YOLOv8 ONNX: (1,84,8400) typically
    out = out[0]
    if out.shape[0] < out.shape[1]:
        out = out.T  # (N,84)

    boxes = out[:, :4]
    scores = out[:, 4:]
    cls = np.argmax(scores, axis=1)
    conf = scores[np.arange(scores.shape[0]), cls]

    keep = conf > CONF
    boxes, conf, cls = boxes[keep], conf[keep], cls[keep]

    # NMS to remove duplicates
    if boxes.size:
        # boxes currently xywh(center) in 'boxes' before conversion below
        # Convert to xyxy in INPUT space for NMS
        b = boxes.copy()
        b[:, 0] -= b[:, 2] / 2
        b[:, 1] -= b[:, 3] / 2
        b[:, 2] += b[:, 0]
        b[:, 3] += b[:, 1]
        idxs = cv2.dnn.NMSBoxes(
            b.tolist(),
            conf.tolist(),
            score_threshold=float(CONF),
            nms_threshold=0.45
        )
        if len(idxs):
            idxs = np.array(idxs).reshape(-1)
            boxes, conf, cls = boxes[idxs], conf[idxs], cls[idxs]


    dets = []
    if boxes.size:
        xywh = boxes.copy()
        xywh[:, 0] -= xywh[:, 2] / 2
        xywh[:, 1] -= xywh[:, 3] / 2
        xywh[:, 2] += xywh[:, 0]
        xywh[:, 3] += xywh[:, 1]

        xywh[:, [0,2]] -= padw
        xywh[:, [1,3]] -= padh
        xywh /= r

        h, w = frame.shape[:2]
        xywh[:, [0,2]] = np.clip(xywh[:, [0,2]], 0, w-1)
        xywh[:, [1,3]] = np.clip(xywh[:, [1,3]], 0, h-1)

        for (x1,y1,x2,y2), c, k in zip(xywh, conf, cls):
            dets.append({"cls": int(k), "conf": float(c), "xyxy": [float(x1),float(y1),float(x2),float(y2)]})

    return {"ok": True, "dets": dets}
