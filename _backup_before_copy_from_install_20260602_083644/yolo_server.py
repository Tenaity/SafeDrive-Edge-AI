import os
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== PERSON MODEL =====
PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolo11s.pt")

# ===== NEW MULTICLASS MODEL FROM KAGGLE =====
MULTI_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "phone_multiclass_kaggle_v13",
    "best.pt",
)

# ===== PERSON MODEL CLASSES =====
PERSON_MODEL_CLS_PERSON = 0

# ===== MULTICLASS MODEL CLASSES =====
CLS_PHONE = 0
CLS_WALKIE = 1
CLS_MOUSE = 2
CLS_CIGARETTE_PACK = 3
CLS_REMOTE = 4

# ===== OUTPUT CLASSES (KEEP MAIN APP COMPATIBLE) =====
OUT_CLS_PERSON = 0
OUT_CLS_PHONE = 67

# ===== THRESHOLDS =====
CONF_MIN_PERSON = 0.42

CONF_MIN_PHONE = 0.50
CONF_MIN_WALKIE = 0.35
CONF_MIN_MOUSE = 0.45
CONF_MIN_CIGARETTE = 0.45
CONF_MIN_REMOTE = 0.45

MIN_BOX_AREA_RATIO = 0.0004
MAX_BOX_AREA_RATIO = 0.12
MIN_BOX_ASPECT = 0.15
MAX_BOX_ASPECT = 6.50

INFER_LONG_SIDE = 960

app = FastAPI(title="YOLO Detect API")

if not os.path.exists(PERSON_MODEL_PATH):
    raise FileNotFoundError(f"Person model not found: {PERSON_MODEL_PATH}")

if not os.path.exists(MULTI_MODEL_PATH):
    raise FileNotFoundError(f"Multiclass model not found: {MULTI_MODEL_PATH}")

person_model = YOLO(PERSON_MODEL_PATH)
multi_model = YOLO(MULTI_MODEL_PATH)


def resize_keep_ratio(frame, long_side=960):
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


def box_area(box) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box):
    x1, y1, x2, y2 = map(float, box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance(box_a, box_b) -> float:
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def valid_small_object_box(xyxy, frame_w: int, frame_h: int) -> bool:
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
    if area_ratio < MIN_BOX_AREA_RATIO or area_ratio > MAX_BOX_AREA_RATIO:
        return False

    aspect = bw / bh if bh > 1e-6 else 999.0
    if aspect < MIN_BOX_ASPECT or aspect > MAX_BOX_ASPECT:
        return False

    return True


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


def box_inside_driver_context(box, driver_box) -> bool:
    if driver_box is None:
        return True

    x1, y1, x2, y2 = map(float, box)
    dx1, dy1, dx2, dy2 = map(float, driver_box)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    dw = dx2 - dx1
    dh = dy2 - dy1

    ex1 = dx1 - dw * 0.25
    ex2 = dx2 + dw * 0.25
    ey1 = dy1 - dh * 0.15
    ey2 = dy2 + dh * 0.45

    return ex1 <= cx <= ex2 and ey1 <= cy <= ey2


def dedup_boxes(items: List[Dict[str, Any]], dist_thresh: float = 18.0) -> List[Dict[str, Any]]:
    if not items:
        return []

    items_sorted = sorted(items, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
    kept = []

    for item in items_sorted:
        box = item.get("xyxy")
        if box is None:
            continue

        duplicate = False
        for k in kept:
            kb = k.get("xyxy")
            if kb is None:
                continue
            if center_distance(box, kb) <= dist_thresh:
                duplicate = True
                break

        if not duplicate:
            kept.append(item)

    return kept


def run_person_model(infer_frame, scale):
    persons = []

    results = person_model(
        infer_frame,
        verbose=False,
        classes=[PERSON_MODEL_CLS_PERSON],
        conf=CONF_MIN_PERSON,
        imgsz=INFER_LONG_SIDE,
    )

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
            if c != PERSON_MODEL_CLS_PERSON or score < CONF_MIN_PERSON:
                continue

            box_small = [float(v) for v in xyxy[i].tolist()]
            box = scale_box_back(box_small, scale)

            persons.append({
                "cls": OUT_CLS_PERSON,
                "conf": score,
                "xyxy": box
            })

    return persons


def run_multi_model(infer_frame, scale, frame_w, frame_h, driver_box):
    phones = []
    walkies = []
    mice = []
    cigarette_packs = []
    remotes = []

    results = multi_model(
        infer_frame,
        verbose=False,
        classes=[CLS_PHONE, CLS_WALKIE, CLS_MOUSE, CLS_CIGARETTE_PACK, CLS_REMOTE],
        conf=min(CONF_MIN_PHONE, CONF_MIN_WALKIE, CONF_MIN_MOUSE, CONF_MIN_CIGARETTE, CONF_MIN_REMOTE),
        imgsz=INFER_LONG_SIDE,
    )

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

            if not valid_small_object_box(box, frame_w, frame_h):
                continue
            if not box_inside_driver_context(box, driver_box):
                continue

            item = {"cls": c, "conf": score, "xyxy": box}

            if c == CLS_PHONE and score >= CONF_MIN_PHONE:
                phones.append(item)
            elif c == CLS_WALKIE and score >= CONF_MIN_WALKIE:
                walkies.append(item)
            elif c == CLS_MOUSE and score >= CONF_MIN_MOUSE:
                mice.append(item)
            elif c == CLS_CIGARETTE_PACK and score >= CONF_MIN_CIGARETTE:
                cigarette_packs.append(item)
            elif c == CLS_REMOTE and score >= CONF_MIN_REMOTE:
                remotes.append(item)

    phones = dedup_boxes(phones, dist_thresh=18.0)
    walkies = dedup_boxes(walkies, dist_thresh=18.0)
    mice = dedup_boxes(mice, dist_thresh=18.0)
    cigarette_packs = dedup_boxes(cigarette_packs, dist_thresh=18.0)
    remotes = dedup_boxes(remotes, dist_thresh=18.0)

    filtered_phones = []
    for ph in phones:
        ph_box = ph["xyxy"]
        ph_conf = float(ph["conf"])
        reject = False

        for other_group in [walkies, mice, cigarette_packs, remotes]:
            for ot in other_group:
                if center_distance(ph_box, ot["xyxy"]) <= 22.0 and float(ot["conf"]) >= ph_conf + 0.03:
                    reject = True
                    break
            if reject:
                break

        if not reject:
            filtered_phones.append({
                "cls": OUT_CLS_PHONE,
                "conf": ph_conf,
                "xyxy": ph_box
            })

    return {
        "phones": filtered_phones,
        "walkies": walkies,
        "mice": mice,
        "cigarette_packs": cigarette_packs,
        "remotes": remotes,
    }


@app.on_event("startup")
def startup_event():
    dummy = np.zeros((INFER_LONG_SIDE, INFER_LONG_SIDE, 3), dtype=np.uint8)

    try:
        person_model(dummy, verbose=False, classes=[PERSON_MODEL_CLS_PERSON], imgsz=INFER_LONG_SIDE)
        print(f"[YOLO SERVER] Person warmup done. Model: {PERSON_MODEL_PATH}, imgsz={INFER_LONG_SIDE}")
    except Exception as e:
        print(f"[YOLO SERVER] Person warmup failed: {e}")

    try:
        multi_model(
            dummy,
            verbose=False,
            classes=[CLS_PHONE, CLS_WALKIE, CLS_MOUSE, CLS_CIGARETTE_PACK, CLS_REMOTE],
            imgsz=INFER_LONG_SIDE
        )
        print(f"[YOLO SERVER] Multiclass warmup done. Model: {MULTI_MODEL_PATH}, imgsz={INFER_LONG_SIDE}")
    except Exception as e:
        print(f"[YOLO SERVER] Multiclass warmup failed: {e}")


@app.get("/health")
def health():
    return {
        "ok": True,
        "person_model": PERSON_MODEL_PATH,
        "multiclass_model": MULTI_MODEL_PATH,
        "imgsz": INFER_LONG_SIDE
    }


@app.post("/detect")
async def detect(image: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        raw = await image.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {
                "ok": False,
                "error": "invalid image",
                "dets": [],
                "persons": [],
                "phones": [],
                "walkies": [],
                "mice": [],
                "cigarette_packs": [],
                "remotes": [],
            }

        frame_h, frame_w = frame.shape[:2]
        infer_frame, scale = resize_keep_ratio(frame, INFER_LONG_SIDE)

        persons = run_person_model(infer_frame, scale)
        driver = select_driver_person([p["xyxy"] for p in persons], frame_w, frame_h)

        multi_out = run_multi_model(
            infer_frame=infer_frame,
            scale=scale,
            frame_w=frame_w,
            frame_h=frame_h,
            driver_box=driver,
        )

        persons_out = []
        dets = []

        if driver is not None:
            persons_out.append({"cls": OUT_CLS_PERSON, "conf": 1.0, "xyxy": driver})
            dets.append({"cls": OUT_CLS_PERSON, "conf": 1.0, "xyxy": driver})

        for ph in multi_out["phones"]:
            dets.append(ph)

        return {
            "ok": True,
            "dets": dets,
            "persons": persons_out,
            "phones": multi_out["phones"],
            "walkies": multi_out["walkies"],
            "mice": multi_out["mice"],
            "cigarette_packs": multi_out["cigarette_packs"],
            "remotes": multi_out["remotes"],
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "dets": [],
            "persons": [],
            "phones": [],
            "walkies": [],
            "mice": [],
            "cigarette_packs": [],
            "remotes": [],
        }


if __name__ == "__main__":
    uvicorn.run("yolo_server:app", host="127.0.0.1", port=8000, reload=False)