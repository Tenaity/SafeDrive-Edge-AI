from pathlib import Path
import re
import shutil
from datetime import datetime

ROOT = Path(r"D:\project_detectfaceandphone\SafeDrive-Edge-AI")
YOLO_SERVER = ROOT / "yolo_server.py"
NEW_MODEL = ROOT / "models" / "phone_radio_best.pt"
BACKUP_DIR = ROOT / "backup_before_phone_radio"

BACKUP_DIR.mkdir(parents=True, exist_ok=True)

if not YOLO_SERVER.exists():
    raise FileNotFoundError(f"Missing yolo_server.py: {YOLO_SERVER}")

if not NEW_MODEL.exists():
    raise FileNotFoundError(f"Missing new model: {NEW_MODEL}")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = BACKUP_DIR / f"yolo_server.py.before_phone_radio_{ts}.bak"
shutil.copy2(YOLO_SERVER, backup)

text = YOLO_SERVER.read_text(encoding="utf-8", errors="replace")
old = text

# 1) Replace old multiclass model path with new phone/radio model.
pattern_model = re.compile(
    r'MULTI_MODEL_PATH\s*=\s*os\.path\.join\(\s*'
    r'BASE_DIR\s*,\s*["\']models["\']\s*,\s*'
    r'["\']phone_multiclass_kaggle_v13["\']\s*,\s*'
    r'["\']best\.pt["\']\s*,?\s*\)',
    flags=re.DOTALL,
)

text = pattern_model.sub(
    'MULTI_MODEL_PATH = os.path.join(BASE_DIR, "models", "phone_radio_best.pt")',
    text,
)

# 2) Force new class mapping:
# new phone_radio_best.pt:
# class 0 = phone
# class 1 = walkie_talkie
text = re.sub(r"^CLS_PHONE\s*=.*$", "CLS_PHONE = 0", text, flags=re.MULTILINE)

if re.search(r"^CLS_WALKIE\s*=", text, flags=re.MULTILINE):
    text = re.sub(r"^CLS_WALKIE\s*=.*$", "CLS_WALKIE = 1", text, flags=re.MULTILINE)
else:
    text = re.sub(
        r"^(CLS_PHONE\s*=\s*0\s*)$",
        r"\1\nCLS_WALKIE = 1",
        text,
        flags=re.MULTILINE,
    )

# 3) Thresholds for first deployment test.
# Phone not too low to avoid false alarm.
# Walkie lower so walkie can suppress phone confusion.
text = re.sub(r"^CONF_MIN_PHONE\s*=.*$", "CONF_MIN_PHONE = 0.35", text, flags=re.MULTILINE)
text = re.sub(r"^CONF_MIN_WALKIE\s*=.*$", "CONF_MIN_WALKIE = 0.25", text, flags=re.MULTILINE)

# 4) Old model had extra classes mouse/cigarette/remote.
# New model has only phone + walkie, so do not request invalid old classes.
text = re.sub(
    r"classes\s*=\s*\[\s*CLS_PHONE\s*,\s*CLS_WALKIE\s*,\s*CLS_MOUSE\s*,\s*CLS_CIGARETTE_PACK\s*,\s*CLS_REMOTE\s*\]",
    "classes=[CLS_PHONE, CLS_WALKIE]",
    text,
)

# 5) Same for min confidence list.
text = re.sub(
    r"conf\s*=\s*min\(\s*CONF_MIN_PHONE\s*,\s*CONF_MIN_WALKIE\s*,\s*CONF_MIN_MOUSE\s*,\s*CONF_MIN_CIGARETTE\s*,\s*CONF_MIN_REMOTE\s*\)",
    "conf=min(CONF_MIN_PHONE, CONF_MIN_WALKIE)",
    text,
)

if text == old:
    print("[WARN] No text changed. Please inspect yolo_server.py manually.")
else:
    YOLO_SERVER.write_text(text, encoding="utf-8")
    print("[OK] Patched yolo_server.py")

print(f"Backup    : {backup}")
print(f"New model : {NEW_MODEL}")
print(f"Server    : {YOLO_SERVER}")
