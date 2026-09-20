import csv
import shutil
from pathlib import Path
from datetime import datetime

import cv2


ROOT = Path(r"D:\project_detectfaceandphone\SafeDrive-Edge-AI")
DATASET = ROOT / "datasets" / "phone_radio_cabin"

SRC_ROOT = DATASET / "01_frames_raw"
UNIQUE_ROOT = DATASET / "01_frames_unique_90"
DUP_ROOT = DATASET / "01_frames_duplicate_90"
INV_DIR = DATASET / "00_inventory"

REPORT_CSV = INV_DIR / "dedupe_90_report.csv"
SUMMARY_TXT = INV_DIR / "dedupe_90_summary.txt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# 64-bit dHash.
# Hamming distance <= 6 nghĩa là giống khoảng >= 90%.
HAMMING_THRESHOLD = 6

# Để tránh mất cảnh hiếm, cứ mỗi clip giữ tối thiểu 1 ảnh sau mỗi N ảnh,
# kể cả ảnh rất giống. Với FPS_SAMPLE=1 thì 20 ảnh ~ 20 giây.
FORCE_KEEP_EVERY_N_IMAGES = 20

COPY_MODE = True
# True  = copy ảnh sang folder mới, an toàn nhưng tốn dung lượng
# False = hardlink nếu có thể, nhẹ hơn nhưng phức tạp hơn


def dhash_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Resize về 9x8 để tạo 64 phép so sánh ngang
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]

    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)

    return value


def hamming(a: int, b: int):
    return (a ^ b).bit_count()


def copy_or_link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if COPY_MODE:
        shutil.copy2(src, dst)
        return

    try:
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def process_clip(clip_dir: Path, index: int, total: int):
    images = sorted(
        [p for p in clip_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda x: x.name.lower()
    )

    rel_clip = clip_dir.name
    unique_dir = UNIQUE_ROOT / rel_clip
    dup_dir = DUP_ROOT / rel_clip

    unique_dir.mkdir(parents=True, exist_ok=True)
    dup_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    last_keep_hash = None
    last_keep_name = ""
    since_last_forced_keep = 999999

    kept = 0
    dup = 0
    bad = 0

    for img_path in images:
        img_hash = dhash_image(img_path)

        rel_name = img_path.name

        if img_hash is None:
            bad += 1
            rows.append({
                "ClipName": rel_clip,
                "ImageName": rel_name,
                "Decision": "BAD_IMAGE",
                "SimilarityPercent": "",
                "HammingDistance": "",
                "ReferenceImage": "",
                "SrcPath": str(img_path),
                "OutPath": "",
            })
            continue

        keep = False
        decision = ""
        distance = ""
        similarity = ""
        reference = last_keep_name

        if last_keep_hash is None:
            keep = True
            decision = "KEEP_FIRST"
        else:
            d = hamming(img_hash, last_keep_hash)
            sim = round((1.0 - d / 64.0) * 100.0, 2)

            distance = d
            similarity = sim

            if since_last_forced_keep >= FORCE_KEEP_EVERY_N_IMAGES:
                keep = True
                decision = "KEEP_FORCED_INTERVAL"
            elif d <= HAMMING_THRESHOLD:
                keep = False
                decision = "DUPLICATE_90"
            else:
                keep = True
                decision = "KEEP_DIFFERENT"

        if keep:
            out_path = unique_dir / img_path.name
            copy_or_link(img_path, out_path)
            kept += 1
            last_keep_hash = img_hash
            last_keep_name = img_path.name
            since_last_forced_keep = 0
        else:
            out_path = dup_dir / img_path.name
            copy_or_link(img_path, out_path)
            dup += 1
            since_last_forced_keep += 1

        rows.append({
            "ClipName": rel_clip,
            "ImageName": rel_name,
            "Decision": decision,
            "SimilarityPercent": similarity,
            "HammingDistance": distance,
            "ReferenceImage": reference,
            "SrcPath": str(img_path),
            "OutPath": str(out_path),
        })

    print(
        f"[{index:03d}/{total}] {clip_dir.name} | "
        f"total={len(images)} | keep={kept} | dup={dup} | bad={bad}"
    )

    return rows, kept, dup, bad, len(images)


def main():
    print("==================================================")
    print("STEP 6/20 - DEDUPE FRAMES BY 90% SIMILARITY")
    print("==================================================")
    print(f"SRC_ROOT       : {SRC_ROOT}")
    print(f"UNIQUE_ROOT    : {UNIQUE_ROOT}")
    print(f"DUP_ROOT       : {DUP_ROOT}")
    print(f"HAMMING_THRESH : {HAMMING_THRESHOLD}")
    print(f"FORCE_KEEP_N   : {FORCE_KEEP_EVERY_N_IMAGES}")
    print("==================================================")

    INV_DIR.mkdir(parents=True, exist_ok=True)

    if not SRC_ROOT.exists():
        print(f"[ERROR] SRC_ROOT not found: {SRC_ROOT}")
        raise SystemExit(1)

    # Xóa output cũ để tránh lẫn dữ liệu
    if UNIQUE_ROOT.exists():
        shutil.rmtree(UNIQUE_ROOT)
    if DUP_ROOT.exists():
        shutil.rmtree(DUP_ROOT)

    UNIQUE_ROOT.mkdir(parents=True, exist_ok=True)
    DUP_ROOT.mkdir(parents=True, exist_ok=True)

    clip_dirs = sorted(
        [p for p in SRC_ROOT.iterdir() if p.is_dir()],
        key=lambda x: x.name.lower()
    )

    all_rows = []
    total_images = 0
    total_kept = 0
    total_dup = 0
    total_bad = 0

    for i, clip_dir in enumerate(clip_dirs, start=1):
        rows, kept, dup, bad, total = process_clip(clip_dir, i, len(clip_dirs))
        all_rows.extend(rows)
        total_images += total
        total_kept += kept
        total_dup += dup
        total_bad += bad

    fieldnames = [
        "ClipName",
        "ImageName",
        "Decision",
        "SimilarityPercent",
        "HammingDistance",
        "ReferenceImage",
        "SrcPath",
        "OutPath",
    ]

    with REPORT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    reduction = 0.0
    if total_images > 0:
        reduction = round(total_dup / total_images * 100.0, 2)

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("SAFE DRIVE PHONE/RADIO - DEDUPE 90 SUMMARY\n")
        f.write("==================================================\n")
        f.write(f"Time             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source           : {SRC_ROOT}\n")
        f.write(f"Unique output    : {UNIQUE_ROOT}\n")
        f.write(f"Duplicate output : {DUP_ROOT}\n")
        f.write(f"Total images     : {total_images}\n")
        f.write(f"Kept unique      : {total_kept}\n")
        f.write(f"Duplicates       : {total_dup}\n")
        f.write(f"Bad images       : {total_bad}\n")
        f.write(f"Reduction percent: {reduction}%\n")
        f.write(f"Report CSV       : {REPORT_CSV}\n")
        f.write("==================================================\n")

    print("")
    print("==================================================")
    print("STEP 6/20 DONE")
    print("==================================================")
    print(f"Total images      : {total_images}")
    print(f"Kept unique       : {total_kept}")
    print(f"Duplicates        : {total_dup}")
    print(f"Bad images        : {total_bad}")
    print(f"Reduction percent : {reduction}%")
    print(f"Unique output     : {UNIQUE_ROOT}")
    print(f"Duplicate output  : {DUP_ROOT}")
    print(f"Report CSV        : {REPORT_CSV}")
    print(f"Summary           : {SUMMARY_TXT}")
    print("==================================================")


if __name__ == "__main__":
    main()
