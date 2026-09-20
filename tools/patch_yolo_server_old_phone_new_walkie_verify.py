from pathlib import Path
import re
import shutil
from datetime import datetime

ROOT = Path(r"D:\project_detectfaceandphone\SafeDrive-Edge-AI")
YOLO_SERVER = ROOT / "yolo_server.py"
BACKUP_DIR = ROOT / "backup_before_phone_radio"
NEW_MODEL = ROOT / "models" / "phone_radio_best.pt"

BACKUP_DIR.mkdir(parents=True, exist_ok=True)

if not YOLO_SERVER.exists():
    raise FileNotFoundError(f"Missing yolo_server.py: {YOLO_SERVER}")

if not NEW_MODEL.exists():
    raise FileNotFoundError(f"Missing verify model: {NEW_MODEL}")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = BACKUP_DIR / f"yolo_server.py.old_phone_new_walkie_verify_{ts}.bak"
shutil.copy2(YOLO_SERVER, backup)

text = YOLO_SERVER.read_text(encoding="utf-8", errors="replace")

# --------------------------------------------------
# 1) Force old model back to old phone/multiclass model.
#    Add new verify model separately.
# --------------------------------------------------
multi_block_new = '''MULTI_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "phone_multiclass_kaggle_v13",
    "best.pt",
)

VERIFY_MODEL_PATH = os.path.join(BASE_DIR, "models", "phone_radio_best.pt")'''

# Replace current MULTI_MODEL_PATH block, whether it is old multiline or previously patched single line.
text = re.sub(
    r'MULTI_MODEL_PATH\s*=\s*os\.path\.join\(\s*BASE_DIR\s*,\s*"models"\s*,\s*"phone_radio_best\.pt"\s*\)',
    multi_block_new,
    text,
    flags=re.DOTALL,
)

text = re.sub(
    r'MULTI_MODEL_PATH\s*=\s*os\.path\.join\(\s*BASE_DIR\s*,\s*"models"\s*,\s*"phone_multiclass_kaggle_v13"\s*,\s*"best\.pt"\s*,?\s*\)',
    multi_block_new,
    text,
    flags=re.DOTALL,
)

# Avoid duplicate VERIFY_MODEL_PATH if patch is run again
text = re.sub(
    r'(VERIFY_MODEL_PATH\s*=\s*os\.path\.join\(BASE_DIR,\s*"models",\s*"phone_radio_best\.pt"\)\s*)+',
    'VERIFY_MODEL_PATH = os.path.join(BASE_DIR, "models", "phone_radio_best.pt")\n',
    text,
)

# --------------------------------------------------
# 2) Add verify model existence check.
# --------------------------------------------------
if "Verify model not found" not in text:
    text = text.replace(
        'if not os.path.exists(MULTI_MODEL_PATH):\n    raise FileNotFoundError(f"Multiclass model not found: {MULTI_MODEL_PATH}")',
        'if not os.path.exists(MULTI_MODEL_PATH):\n    raise FileNotFoundError(f"Multiclass model not found: {MULTI_MODEL_PATH}")\n\nif not os.path.exists(VERIFY_MODEL_PATH):\n    raise FileNotFoundError(f"Verify model not found: {VERIFY_MODEL_PATH}")'
    )

# --------------------------------------------------
# 3) Add verify model load.
# --------------------------------------------------
if "verify_model = YOLO(VERIFY_MODEL_PATH)" not in text:
    text = text.replace(
        "multi_model = YOLO(MULTI_MODEL_PATH)",
        "multi_model = YOLO(MULTI_MODEL_PATH)\nverify_model = YOLO(VERIFY_MODEL_PATH)"
    )

# --------------------------------------------------
# 4) Ensure class mapping.
# Old model:
#   phone_multiclass_kaggle_v13:
#       0 phone
#       1 walkie
#       2 mouse
#       3 cigarette pack
#       4 remote
# New verify model:
#       0 phone
#       1 walkie_talkie
# --------------------------------------------------
text = re.sub(r"^CLS_PHONE\s*=.*$", "CLS_PHONE = 0", text, flags=re.MULTILINE)
text = re.sub(r"^CLS_WALKIE\s*=.*$", "CLS_WALKIE = 1", text, flags=re.MULTILINE)

if "VERIFY_CLS_PHONE" not in text:
    insert_after = re.search(r"^CLS_PHONE\s*=\s*0\s*$", text, flags=re.MULTILINE)
    if insert_after:
        pos = insert_after.end()
        text = (
            text[:pos]
            + "\nVERIFY_CLS_PHONE = 0\nVERIFY_CLS_WALKIE = 1"
            + text[pos:]
        )

# Thresholds
text = re.sub(r"^CONF_MIN_PHONE\s*=.*$", "CONF_MIN_PHONE = 0.50", text, flags=re.MULTILINE)
text = re.sub(r"^CONF_MIN_WALKIE\s*=.*$", "CONF_MIN_WALKIE = 0.35", text, flags=re.MULTILINE)

if "VERIFY_CONF_PHONE" not in text:
    marker = re.search(r"^CONF_MIN_WALKIE\s*=.*$", text, flags=re.MULTILINE)
    if marker:
        pos = marker.end()
        text = (
            text[:pos]
            + "\nVERIFY_CONF_PHONE = 0.25\nVERIFY_CONF_WALKIE = 0.25\nVERIFY_WALKIE_IOU_THRESH = 0.12\nVERIFY_WALKIE_CENTER_DIST = 90.0"
            + text[pos:]
        )

# --------------------------------------------------
# 5) Replace run_multi_model with old-phone-first/new-walkie-verify version.
# --------------------------------------------------
new_func = r'''
def run_multi_model(infer_frame, scale, frame_w, frame_h, driver_box):
    """
    Two-stage phone detection.

    Stage 1:
        Old phone_multiclass model detects phone candidates first.

    Stage 2:
        New phone_radio_best.pt checks whether the same region is walkie_talkie.
        If a walkie overlaps / is very close to an old phone candidate, suppress that phone.

    Return format stays compatible with main.py:
        {"phones": [...]}
    """
    phones = []
    old_others = []
    verify_walkies = []
    verify_phones = []

    def to_original_box(xyxy):
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        if scale and scale > 0:
            x1 /= scale
            y1 /= scale
            x2 /= scale
            y2 /= scale

        x1 = max(0, min(int(x1), frame_w - 1))
        y1 = max(0, min(int(y1), frame_h - 1))
        x2 = max(0, min(int(x2), frame_w - 1))
        y2 = max(0, min(int(y2), frame_h - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        return [x1, y1, x2, y2]

    def box_center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def box_area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    def box_iou(a, b):
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        if inter <= 0:
            return 0.0

        union = box_area(a) + box_area(b) - inter
        if union <= 0:
            return 0.0

        return inter / union

    def inside_driver_region(box):
        if driver_box is None:
            return True

        cx, cy = box_center(box)
        x1, y1, x2, y2 = driver_box

        margin_x = 160
        margin_y = 180

        return (
            cx >= x1 - margin_x
            and cx <= x2 + margin_x
            and cy >= y1 - margin_y
            and cy <= y2 + margin_y
        )

    # --------------------------------------------------
    # Stage 1: old model detects phone candidate first
    # --------------------------------------------------
    try:
        old_results = multi_model(
            infer_frame,
            verbose=False,
            imgsz=INFER_LONG_SIDE,
            classes=[CLS_PHONE, CLS_WALKIE, CLS_MOUSE, CLS_CIGARETTE_PACK, CLS_REMOTE],
            conf=min(CONF_MIN_PHONE, CONF_MIN_WALKIE, CONF_MIN_MOUSE, CONF_MIN_CIGARETTE, CONF_MIN_REMOTE),
        )
    except Exception as e:
        print(f"[YOLO SERVER] Old multiclass model error: {e}")
        old_results = []

    for r in old_results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue

        cls = boxes.cls.cpu().numpy() if boxes.cls is not None else []
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []

        for i in range(len(xyxy)):
            c = int(cls[i])
            score = float(conf[i])
            box = to_original_box(xyxy[i])

            if box is None:
                continue

            if not inside_driver_region(box):
                continue

            item = {"cls": c, "conf": score, "xyxy": box}

            if c == CLS_PHONE and score >= CONF_MIN_PHONE:
                phones.append(item)
            elif c == CLS_WALKIE and score >= CONF_MIN_WALKIE:
                old_others.append(item)
            elif c == CLS_MOUSE and score >= CONF_MIN_MOUSE:
                old_others.append(item)
            elif c == CLS_CIGARETTE_PACK and score >= CONF_MIN_CIGARETTE:
                old_others.append(item)
            elif c == CLS_REMOTE and score >= CONF_MIN_REMOTE:
                old_others.append(item)

    phones = dedup_boxes(phones, dist_thresh=18.0)
    old_others = dedup_boxes(old_others, dist_thresh=18.0)

    # --------------------------------------------------
    # Stage 2: new phone/radio model verifies walkie_talkie
    # --------------------------------------------------
    try:
        verify_results = verify_model(
            infer_frame,
            verbose=False,
            imgsz=INFER_LONG_SIDE,
            classes=[VERIFY_CLS_PHONE, VERIFY_CLS_WALKIE],
            conf=min(VERIFY_CONF_PHONE, VERIFY_CONF_WALKIE),
        )
    except Exception as e:
        print(f"[YOLO SERVER] Verify phone/radio model error: {e}")
        verify_results = []

    for r in verify_results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue

        cls = boxes.cls.cpu().numpy() if boxes.cls is not None else []
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []

        for i in range(len(xyxy)):
            c = int(cls[i])
            score = float(conf[i])
            box = to_original_box(xyxy[i])

            if box is None:
                continue

            if not inside_driver_region(box):
                continue

            item = {"cls": c, "conf": score, "xyxy": box}

            if c == VERIFY_CLS_WALKIE and score >= VERIFY_CONF_WALKIE:
                verify_walkies.append(item)
            elif c == VERIFY_CLS_PHONE and score >= VERIFY_CONF_PHONE:
                verify_phones.append(item)

    verify_walkies = dedup_boxes(verify_walkies, dist_thresh=18.0)
    verify_phones = dedup_boxes(verify_phones, dist_thresh=18.0)

    # --------------------------------------------------
    # Suppression logic:
    # - Old non-phone objects can suppress phone, same as before.
    # - New walkie_talkie strongly suppresses old phone if overlap/near.
    # --------------------------------------------------
    filtered_phones = []

    for ph in phones:
        ph_box = ph["xyxy"]
        ph_conf = float(ph["conf"])

        suppressed = False
        suppress_reason = ""

        # Old model anti-false-positive objects
        for ot in old_others:
            if center_distance(ph_box, ot["xyxy"]) <= 22.0 and float(ot["conf"]) >= ph_conf + 0.03:
                suppressed = True
                suppress_reason = "old_other_object"
                break

        # New verify model: walkie_talkie suppresses phone
        if not suppressed:
            for wk in verify_walkies:
                wk_box = wk["xyxy"]
                iou = box_iou(ph_box, wk_box)
                dist = center_distance(ph_box, wk_box)

                if iou >= VERIFY_WALKIE_IOU_THRESH or dist <= VERIFY_WALKIE_CENTER_DIST:
                    suppressed = True
                    suppress_reason = f"verify_walkie iou={iou:.3f} dist={dist:.1f} conf={float(wk['conf']):.2f}"
                    break

        if suppressed:
            # Debug only. Do not return as phone.
            # print(f"[YOLO SERVER] Suppressed old phone as walkie: {suppress_reason}")
            continue

        # Optional metadata: did new model also see phone near this candidate?
        verify_phone_conf = 0.0
        for vp in verify_phones:
            iou = box_iou(ph_box, vp["xyxy"])
            dist = center_distance(ph_box, vp["xyxy"])
            if iou >= 0.08 or dist <= 90.0:
                verify_phone_conf = max(verify_phone_conf, float(vp["conf"]))

        filtered_phones.append({
            "cls": OUT_CLS_PHONE,
            "conf": ph_conf,
            "xyxy": ph_box,
            "source": "old_phone_model",
            "verify_phone_conf": verify_phone_conf,
        })

    return {
        "phones": filtered_phones,
        "verify_walkies": verify_walkies,
        "verify_phones": verify_phones,
    }
'''

start = text.find("def run_multi_model(")
if start < 0:
    raise RuntimeError("Cannot find def run_multi_model(...) in yolo_server.py")

end_candidates = []
for marker in ["\ndef warmup", "\n@app.get", "\n@app.post"]:
    pos = text.find(marker, start + 1)
    if pos > start:
        end_candidates.append(pos)

if not end_candidates:
    raise RuntimeError("Cannot find end of run_multi_model function")

end = min(end_candidates)
text = text[:start] + new_func.strip() + "\n\n" + text[end:]

# --------------------------------------------------
# 6) Health response includes verify model, if the old health block exists.
# --------------------------------------------------
if '"verify_model"' not in text:
    text = text.replace(
        '"multiclass_model": MULTI_MODEL_PATH,',
        '"multiclass_model": MULTI_MODEL_PATH,\n        "verify_model": VERIFY_MODEL_PATH,'
    )

YOLO_SERVER.write_text(text, encoding="utf-8")

print("[OK] Patched yolo_server.py: old phone first + new walkie verify")
print(f"Backup : {backup}")
print(f"Server : {YOLO_SERVER}")
print(f"Verify : {NEW_MODEL}")
