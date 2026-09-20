import cv2
from pathlib import Path

ROOT = Path(r"D:\project_detectfaceandphone\SafeDrive-Edge-AI")
RAW = ROOT / "datasets" / "phone_radio_cabin" / "01_frames_raw"

imgs = sorted(RAW.rglob("*.jpg"), key=lambda x: str(x).lower())

print("==================================================")
print("VERIFY EXTRACTED JPGS")
print("==================================================")
print(f"RAW: {RAW}")
print(f"Total jpg: {len(imgs)}")

bad = []
checked = 0

for p in imgs[:200]:
    img = cv2.imread(str(p))
    checked += 1

    if img is None:
        bad.append(str(p))
    else:
        h, w = img.shape[:2]
        print(f"[OK] {p.name} | {w}x{h}")

print("==================================================")
print(f"Checked: {checked}")
print(f"Bad    : {len(bad)}")

if bad:
    print("Bad files:")
    for b in bad[:50]:
        print(b)
else:
    print("All checked JPG files are readable by OpenCV.")
print("==================================================")
