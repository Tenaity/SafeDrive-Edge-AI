from pathlib import Path
import re
import shutil
from datetime import datetime

ROOT = Path(r"D:\project_detectfaceandphone\SafeDrive-Edge-AI")
YOLO_SERVER = ROOT / "yolo_server.py"
NEW_MODEL = ROOT / "models" / "phone_radio_best.pt"
BACKUP_DIR = ROOT / "_backup_yolo_before_verify_walkie"

BACKUP_DIR.mkdir(parents=True, exist_ok=True)

if not NEW_MODEL.exists():
    raise FileNotFoundError(f"Missing model: {NEW_MODEL}")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = BACKUP_DIR / f"yolo_server_before_verify_walkie_{ts}.py"
shutil.copy2(YOLO_SERVER, backup)

text = YOLO_SERVER.read_text(encoding="utf-8", errors="replace")

# Add verify model path after MULTI_MODEL_PATH block
if "VERIFY_MODEL_PATH" not in text:
    text = text.replace(
        'MULTI_MODEL_PATH = os.path.join(\n    BASE_DIR,\n    "models",\n    "phone_multiclass_kaggle_v13",\n    "best.pt",\n)',
        'MULTI_MODEL_PATH = os.path.join(\n    BASE_DIR,\n    "models",\n    "phone_multiclass_kaggle_v13",\n    "best.pt",\n)\n\nVERIFY_MODEL_PATH = os.path.join(BASE_DIR, "models", "phone_radio_best.pt")'
    )

# Add verify model existence check
if "Verify model not found" not in text:
    text = text.replace(
        'if not os.path.exists(MULTI_MODEL_PATH):\n    raise FileNotFoundError(f"Multiclass model not found: {MULTI_MODEL_PATH}")',
        'if not os.path.exists(MULTI_MODEL_PATH):\n    raise FileNotFoundError(f"Multiclass model not found: {MULTI_MODEL_PATH}")\n\nif not os.path.exists(VERIFY_MODEL_PATH):\n    raise FileNotFoundError(f"Verify model not found: {VERIFY_MODEL_PATH}")'
    )

# Load verify model
if "verify_model = YOLO(VERIFY_MODEL_PATH)" not in text:
    text = text.replace(
        "multi_model = YOLO(MULTI_MODEL_PATH)",
        "multi_model = YOLO(MULTI_MODEL_PATH)\nverify_model = YOLO(VERIFY_MODEL_PATH)"
    )

# Add verify constants
if "VERIFY_CLS_PHONE" not in text:
    text = text.replace(
        "CLS_REMOTE = 4",
        "CLS_REMOTE = 4\n\n# New verify model classes: phone_radio_best.pt\nVERIFY_CLS_PHONE = 0\nVERIFY_CLS_WALKIE = 1"
    )

if "VERIFY_CONF_WALKIE" not in text:
    text = text.replace(
        "CONF_MIN_REMOTE = 0.45",
        "CONF_MIN_REMOTE = 0.45\n\nVERIFY_CONF_PHONE = 0.25\nVERIFY_CONF_WALKIE = 0.25\nVERIFY_WALKIE_IOU_THRESH = 0.12\nVERIFY_WALKIE_CENTER_DIST = 90.0"
    )

new_func = r'''
def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    area_a = box_area(a)
    area_b = box_area(b)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def run_verify_model_once(infer_frame, scale, frame_w, frame_h, driver_box):
    walkies = []
    phones = []

    results = verify_model(
        infer_frame,
        verbose=False,
        classes=[VERIFY_CLS_PHONE, VERIFY_CLS_WALKIE],
        conf=min(VERIFY_CONF_PHONE, VERIFY_CONF_WALKIE),
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

            if c == VERIFY_CLS_WALKIE and score >= VERIFY_CONF_WALKIE:
                walkies.append(item)
            elif c == VERIFY_CLS_PHONE and score >= VERIFY_CONF_PHONE:
                phones.append(item)

    return {
        "verify_walkies": dedup_boxes(walkies, dist_thresh=18.0),
        "verify_phones": dedup_boxes(phones, dist_thresh=18.0),
    }


def run_multi_model(infer_frame, scale, frame_w, frame_h, driver_box):
    phones = []
    walkies = []
    mice = []
    cigarette_packs = []
    remotes = []

    # Stage 1: model cu bat phone truoc
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

    # Stage 2: chi khi model cu da thay phone moi chay model moi verify bo dam
    verify_walkies = []
    verify_phones = []

    if len(phones) > 0:
        verify_out = run_verify_model_once(
            infer_frame=infer_frame,
            scale=scale,
            frame_w=frame_w,
            frame_h=frame_h,
            driver_box=driver_box,
        )
        verify_walkies = verify_out["verify_walkies"]
        verify_phones = verify_out["verify_phones"]

    filtered_phones = []

    for ph in phones:
        ph_box = ph["xyxy"]
        ph_conf = float(ph["conf"])
        reject = False

        # Logic cu: walkie/mouse/cigarette/remote cua model cu co the chan phone
        for other_group in [walkies, mice, cigarette_packs, remotes]:
            for ot in other_group:
                if center_distance(ph_box, ot["xyxy"]) <= 22.0 and float(ot["conf"]) >= ph_conf + 0.03:
                    reject = True
                    break
            if reject:
                break

        # Logic moi: model moi thay walkie gan/trung phone thi chan phone
        if not reject:
            for wk in verify_walkies:
                iou = box_iou(ph_box, wk["xyxy"])
                dist = center_distance(ph_box, wk["xyxy"])

                if iou >= VERIFY_WALKIE_IOU_THRESH or dist <= VERIFY_WALKIE_CENTER_DIST:
                    reject = True
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
        "verify_walkies": verify_walkies,
        "verify_phones": verify_phones,
    }
'''

start = text.find("def run_multi_model(")
end = text.find("@app.on_event", start)

if start < 0 or end < 0:
    raise RuntimeError("Cannot find run_multi_model block")

text = text[:start] + new_func.strip() + "\n\n" + text[end:]

# Warmup verify model
if "Verify warmup done" not in text:
    text = text.replace(
        '    try:\n        multi_model(\n            dummy,\n            verbose=False,\n            classes=[CLS_PHONE, CLS_WALKIE, CLS_MOUSE, CLS_CIGARETTE_PACK, CLS_REMOTE],\n            imgsz=INFER_LONG_SIDE\n        )\n        print(f"[YOLO SERVER] Multiclass warmup done. Model: {MULTI_MODEL_PATH}, imgsz={INFER_LONG_SIDE}")\n    except Exception as e:\n        print(f"[YOLO SERVER] Multiclass warmup failed: {e}")',
        '    try:\n        multi_model(\n            dummy,\n            verbose=False,\n            classes=[CLS_PHONE, CLS_WALKIE, CLS_MOUSE, CLS_CIGARETTE_PACK, CLS_REMOTE],\n            imgsz=INFER_LONG_SIDE\n        )\n        print(f"[YOLO SERVER] Multiclass warmup done. Model: {MULTI_MODEL_PATH}, imgsz={INFER_LONG_SIDE}")\n    except Exception as e:\n        print(f"[YOLO SERVER] Multiclass warmup failed: {e}")\n\n    try:\n        verify_model(\n            dummy,\n            verbose=False,\n            classes=[VERIFY_CLS_PHONE, VERIFY_CLS_WALKIE],\n            imgsz=INFER_LONG_SIDE\n        )\n        print(f"[YOLO SERVER] Verify warmup done. Model: {VERIFY_MODEL_PATH}, imgsz={INFER_LONG_SIDE}")\n    except Exception as e:\n        print(f"[YOLO SERVER] Verify warmup failed: {e}")'
    )

# Health includes verify model
if '"verify_model"' not in text:
    text = text.replace(
        '"multiclass_model": MULTI_MODEL_PATH,',
        '"multiclass_model": MULTI_MODEL_PATH,\n        "verify_model": VERIFY_MODEL_PATH,'
    )

YOLO_SERVER.write_text(text, encoding="utf-8")

print("[OK] Patched yolo_server.py")
print(f"Backup: {backup}")
