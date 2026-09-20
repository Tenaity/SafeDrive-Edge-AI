import csv
import shutil
from pathlib import Path
from datetime import datetime

import cv2


ROOT = Path(r"D:\project_detectfaceandphone\SafeDrive-Edge-AI")
DATASET = ROOT / "datasets" / "phone_radio_cabin"

SRC_ROOT = DATASET / "01_frames_unique_90"
UNIQUE_ROOT = DATASET / "01_frames_unique_80"
DUP_ROOT = DATASET / "01_frames_duplicate_80_from_unique90"
INV_DIR = DATASET / "00_inventory"

REPORT_CSV = INV_DIR / "dedupe_80_from_unique90_report.csv"
SUMMARY_TXT = INV_DIR / "dedupe_80_from_unique90_summary.txt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# dHash 64-bit:
# similarity >= 80% => distance <= 13 bit
HAMMING_THRESHOLD = 13

# False = lọc đúng theo ngưỡng 80%, không ép giữ ảnh định kỳ
FORCE_KEEP_INTERVAL = 0

COPY_MODE = True


def dhash_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]

    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)

    return value


def hamming(a: int, b: int):
    return (a ^ b).bit_count()


def similarity_percent(distance: int):
    return round((1.0 - distance / 64.0) * 100.0, 2)


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if COPY_MODE:
        shutil.copy2(src, dst)
    else:
        try:
            if dst.exists():
                dst.unlink()
            dst.hardlink_to(src)
        except Exception:
            shutil.copy2(src, dst)


def find_duplicate_against_kept(img_hash, kept_hashes):
    best_distance = None
    best_ref = ""

    for ref_name, ref_hash in kept_hashes:
        d = hamming(img_hash, ref_hash)

        if best_distance is None or d < best_distance:
            best_distance = d
            best_ref = ref_name

        if d <= HAMMING_THRESHOLD:
            return True, d, ref_name

    if best_distance is None:
        return False, None, ""

    return False, best_distance, best_ref


def process_clip(clip_dir: Path, index: int, total_clips: int):
    images = sorted(
        [p for p in clip_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda x: x.name.lower()
    )

    clip_name = clip_dir.name
    unique_dir = UNIQUE_ROOT / clip_name
    dup_dir = DUP_ROOT / clip_name

    unique_dir.mkdir(parents=True, exist_ok=True)
    dup_dir.mkdir(parents=True, exist_ok=True)

    kept_hashes = []

    rows = []
    kept = 0
    dup = 0
    bad = 0
    processed_since_force = 0

    for img_path in images:
        img_hash = dhash_image(img_path)

        if img_hash is None:
            bad += 1
            rows.append({
                "ClipName": clip_name,
                "ImageName": img_path.name,
                "Decision": "BAD_IMAGE",
                "SimilarityPercent": "",
                "HammingDistance": "",
                "ReferenceImage": "",
                "SrcPath": str(img_path),
                "OutPath": "",
            })
            continue

        if not kept_hashes:
            out_path = unique_dir / img_path.name
            copy_file(img_path, out_path)
            kept_hashes.append((img_path.name, img_hash))
            kept += 1
            processed_since_force = 0

            rows.append({
                "ClipName": clip_name,
                "ImageName": img_path.name,
                "Decision": "KEEP_FIRST",
                "SimilarityPercent": "",
                "HammingDistance": "",
                "ReferenceImage": "",
                "SrcPath": str(img_path),
                "OutPath": str(out_path),
            })
            continue

        is_dup, distance, ref_name = find_duplicate_against_kept(img_hash, kept_hashes)

        force_keep = False
        if FORCE_KEEP_INTERVAL and processed_since_force >= FORCE_KEEP_INTERVAL:
            force_keep = True

        if is_dup and not force_keep:
            out_path = dup_dir / img_path.name
            copy_file(img_path, out_path)
            dup += 1
            decision = "DUPLICATE_80"
        else:
            out_path = unique_dir / img_path.name
            copy_file(img_path, out_path)
            kept_hashes.append((img_path.name, img_hash))
            kept += 1
            processed_since_force = 0
            decision = "KEEP_DIFFERENT" if not force_keep else "KEEP_FORCED"

        sim = ""
        if distance is not None:
            sim = similarity_percent(distance)

        rows.append({
            "ClipName": clip_name,
            "ImageName": img_path.name,
            "Decision": decision,
            "SimilarityPercent": sim,
            "HammingDistance": distance if distance is not None else "",
            "ReferenceImage": ref_name,
            "SrcPath": str(img_path),
            "OutPath": str(out_path),
        })

        processed_since_force += 1

    print(
        f"[{index:03d}/{total_clips}] {clip_name} | "
        f"input={len(images)} | keep={kept} | duplicate80={dup} | bad={bad}"
    )

    return rows, len(images), kept, dup, bad


def main():
    print("==================================================")
    print("STEP 7/20 - DEDUPE UNIQUE_90 AGAIN BY 80%")
    print("==================================================")
    print(f"SRC_ROOT          : {SRC_ROOT}")
    print(f"UNIQUE_ROOT       : {UNIQUE_ROOT}")
    print(f"DUP_ROOT          : {DUP_ROOT}")
    print(f"HAMMING_THRESHOLD : {HAMMING_THRESHOLD}")
    print("==================================================")

    if not SRC_ROOT.exists():
        print(f"[ERROR] Source folder not found: {SRC_ROOT}")
        raise SystemExit(1)

    INV_DIR.mkdir(parents=True, exist_ok=True)

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

    total_input = 0
    total_kept = 0
    total_dup = 0
    total_bad = 0

    for i, clip_dir in enumerate(clip_dirs, start=1):
        rows, input_count, kept, dup, bad = process_clip(clip_dir, i, len(clip_dirs))

        all_rows.extend(rows)
        total_input += input_count
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
    if total_input > 0:
        reduction = round(total_dup / total_input * 100.0, 2)

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("SAFE DRIVE PHONE/RADIO - DEDUPE 80 FROM UNIQUE90\n")
        f.write("==================================================\n")
        f.write(f"Time             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source           : {SRC_ROOT}\n")
        f.write(f"Unique output    : {UNIQUE_ROOT}\n")
        f.write(f"Duplicate output : {DUP_ROOT}\n")
        f.write(f"Hamming threshold: {HAMMING_THRESHOLD}\n")
        f.write(f"Total input      : {total_input}\n")
        f.write(f"Kept unique      : {total_kept}\n")
        f.write(f"Duplicates 80    : {total_dup}\n")
        f.write(f"Bad images       : {total_bad}\n")
        f.write(f"Reduction percent: {reduction}%\n")
        f.write(f"Report CSV       : {REPORT_CSV}\n")
        f.write("==================================================\n")

    print("")
    print("==================================================")
    print("STEP 7/20 DONE")
    print("==================================================")
    print(f"Total input       : {total_input}")
    print(f"Kept unique       : {total_kept}")
    print(f"Duplicates 80     : {total_dup}")
    print(f"Bad images        : {total_bad}")
    print(f"Reduction percent : {reduction}%")
    print(f"Unique output     : {UNIQUE_ROOT}")
    print(f"Duplicate output  : {DUP_ROOT}")
    print(f"Report CSV        : {REPORT_CSV}")
    print(f"Summary           : {SUMMARY_TXT}")
    print("==================================================")


if __name__ == "__main__":
    main()
