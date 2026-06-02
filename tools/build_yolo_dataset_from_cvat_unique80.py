import csv
import zipfile
import shutil
import hashlib
from pathlib import Path
from datetime import datetime


ROOT = Path(r"D:\project_detectfaceandphone\SafeDrive-Edge-AI")
DATASET_ROOT = ROOT / "datasets" / "phone_radio_cabin"

EXPORT_DIR = DATASET_ROOT / "04_cvat_export_unique80"
EXTRACT_DIR = DATASET_ROOT / "04_cvat_export_unique80_unzipped"
YOLO_OUT = DATASET_ROOT / "05_yolo_dataset_unique80"
INV_DIR = DATASET_ROOT / "00_inventory"

REPORT_CSV = INV_DIR / "build_yolo_dataset_unique80_report.csv"
SUMMARY_TXT = INV_DIR / "build_yolo_dataset_unique80_summary.txt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CANONICAL_NAMES = ["phone", "walkie_talkie"]

# Chia theo clip để tránh ảnh cùng clip vừa ở train vừa ở val
VAL_RATIO = 0.2


def normalize_name(name: str) -> str:
    s = name.strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    if s in ("phone", "cell_phone", "mobile_phone", "telephone"):
        return "phone"
    if s in ("walkie_talkie", "walkie", "radio", "bo_dam", "bộ_đàm", "bodam"):
        return "walkie_talkie"
    return s


def stable_val_split_key(clip_name: str) -> bool:
    h = hashlib.md5(clip_name.encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return v < VAL_RATIO


def get_clip_name(image_name: str) -> str:
    stem = Path(image_name).stem
    # hiv00023_000374 -> hiv00023
    if "_" in stem:
        return stem.split("_")[0]
    return stem


def find_names_file(root: Path):
    candidates = []
    for name in ("obj.names", "classes.txt", "names.txt"):
        candidates.extend(root.rglob(name))
    return candidates[0] if candidates else None


def read_class_names(extracted_root: Path):
    names_file = find_names_file(extracted_root)

    if names_file is None:
        print("[WARN] Khong tim thay obj.names/classes.txt. Gia dinh class 0=phone, 1=walkie_talkie")
        return {0: "phone", 1: "walkie_talkie"}, None

    raw_names = [
        line.strip()
        for line in names_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]

    old_id_to_name = {}
    for i, name in enumerate(raw_names):
        old_id_to_name[i] = normalize_name(name)

    return old_id_to_name, names_file


def build_class_id_map(old_id_to_name):
    new_name_to_id = {name: i for i, name in enumerate(CANONICAL_NAMES)}
    old_to_new = {}

    for old_id, old_name in old_id_to_name.items():
        if old_name in new_name_to_id:
            old_to_new[old_id] = new_name_to_id[old_name]
        else:
            print(f"[WARN] Bo qua class khong dung: old_id={old_id}, name={old_name}")

    return old_to_new


def find_images(root: Path):
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda x: str(x).lower()
    )


def find_label_for_image(img_path: Path):
    # Label YOLO thường cùng stem với ảnh
    same_dir = img_path.with_suffix(".txt")
    if same_dir.exists():
        return same_dir

    # Tìm trong toàn bộ extract theo stem nếu CVAT tách folder khác
    candidates = list(img_path.parents[-1].rglob(img_path.stem + ".txt")) if False else []
    return None


def convert_label_file(src_label: Path, dst_label: Path, old_to_new):
    box_count = 0
    phone_count = 0
    walkie_count = 0
    skipped_count = 0

    out_lines = []

    if src_label is not None and src_label.exists():
        lines = src_label.read_text(encoding="utf-8", errors="replace").splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                skipped_count += 1
                continue

            try:
                old_id = int(float(parts[0]))
            except Exception:
                skipped_count += 1
                continue

            if old_id not in old_to_new:
                skipped_count += 1
                continue

            new_id = old_to_new[old_id]
            coords = parts[1:5]

            try:
                vals = [float(x) for x in coords]
            except Exception:
                skipped_count += 1
                continue

            # YOLO normalized: x y w h, thường 0..1
            x, y, w, h = vals
            if w <= 0 or h <= 0:
                skipped_count += 1
                continue

            out_lines.append(
                f"{new_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}"
            )

            box_count += 1
            if new_id == 0:
                phone_count += 1
            elif new_id == 1:
                walkie_count += 1

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    return box_count, phone_count, walkie_count, skipped_count


def main():
    print("==================================================")
    print("STEP 11/20 - BUILD YOLO DATASET FROM CVAT EXPORT")
    print("==================================================")
    print(f"EXPORT_DIR  : {EXPORT_DIR}")
    print(f"EXTRACT_DIR : {EXTRACT_DIR}")
    print(f"YOLO_OUT    : {YOLO_OUT}")
    print("==================================================")

    INV_DIR.mkdir(parents=True, exist_ok=True)

    zips = sorted(EXPORT_DIR.glob("*.zip"), key=lambda x: x.name.lower())

    if not zips:
        print(f"[ERROR] Khong tim thay file zip export trong: {EXPORT_DIR}")
        raise SystemExit(1)

    print(f"[OK] Found export zip files: {len(zips)}")
    for z in zips:
        print(f"- {z.name}")

    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    print("")
    print("[1/5] Extract CVAT exports...")
    for i, z in enumerate(zips, start=1):
        out_sub = EXTRACT_DIR / f"export_{i:03d}_{z.stem}"
        out_sub.mkdir(parents=True, exist_ok=True)

        print(f"Extracting: {z.name}")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(out_sub)

    print("")
    print("[2/5] Read class names...")
    old_id_to_name, names_file = read_class_names(EXTRACT_DIR)
    old_to_new = build_class_id_map(old_id_to_name)

    print(f"Names file: {names_file}")
    print(f"Old ID to name: {old_id_to_name}")
    print(f"Old ID to new ID: {old_to_new}")

    if not old_to_new:
        print("[ERROR] Khong map duoc class nao ve phone/walkie_talkie.")
        raise SystemExit(1)

    print("")
    print("[3/5] Prepare output folders...")

    if YOLO_OUT.exists():
        shutil.rmtree(YOLO_OUT)

    for sub in [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]:
        (YOLO_OUT / sub).mkdir(parents=True, exist_ok=True)

    print("")
    print("[4/5] Collect images and labels...")

    images = find_images(EXTRACT_DIR)

    if not images:
        print("[ERROR] Khong tim thay image trong export. Khi export CVAT can bat Save images.")
        raise SystemExit(1)

    print(f"[OK] Images found: {len(images)}")

    rows = []
    total_boxes = 0
    total_phone = 0
    total_walkie = 0
    total_skipped = 0
    train_count = 0
    val_count = 0
    negative_count = 0

    used_names = set()

    for idx, img in enumerate(images, start=1):
        clip_name = get_clip_name(img.name)
        split = "val" if stable_val_split_key(clip_name) else "train"

        if split == "train":
            train_count += 1
        else:
            val_count += 1

        # Tránh trùng tên nếu nhiều export task có ảnh cùng tên
        out_name = img.name
        if out_name in used_names:
            out_name = f"{img.parent.name}_{img.name}"
        used_names.add(out_name)

        dst_img = YOLO_OUT / "images" / split / out_name
        dst_label = YOLO_OUT / "labels" / split / (Path(out_name).stem + ".txt")

        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst_img)

        label = img.with_suffix(".txt")
        if not label.exists():
            label = None

        box_count, phone_count, walkie_count, skipped_count = convert_label_file(
            label,
            dst_label,
            old_to_new,
        )

        if box_count == 0:
            negative_count += 1

        total_boxes += box_count
        total_phone += phone_count
        total_walkie += walkie_count
        total_skipped += skipped_count

        rows.append({
            "Index": idx,
            "ImageName": out_name,
            "Split": split,
            "ClipName": clip_name,
            "SourceImage": str(img),
            "SourceLabel": str(label) if label else "",
            "OutputImage": str(dst_img),
            "OutputLabel": str(dst_label),
            "Boxes": box_count,
            "PhoneBoxes": phone_count,
            "WalkieTalkieBoxes": walkie_count,
            "SkippedLabels": skipped_count,
        })

        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(images)} images...")

    print("")
    print("[5/5] Write dataset.yaml and reports...")

    dataset_yaml = YOLO_OUT / "dataset.yaml"
    dataset_yaml.write_text(
        f"""path: {YOLO_OUT.as_posix()}
train: images/train
val: images/val

names:
  0: phone
  1: walkie_talkie
""",
        encoding="utf-8"
    )

    with REPORT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "Index",
            "ImageName",
            "Split",
            "ClipName",
            "SourceImage",
            "SourceLabel",
            "OutputImage",
            "OutputLabel",
            "Boxes",
            "PhoneBoxes",
            "WalkieTalkieBoxes",
            "SkippedLabels",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("SAFE DRIVE PHONE/RADIO - YOLO DATASET SUMMARY\n")
        f.write("==================================================\n")
        f.write(f"Time              : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Export dir         : {EXPORT_DIR}\n")
        f.write(f"Extract dir        : {EXTRACT_DIR}\n")
        f.write(f"YOLO output        : {YOLO_OUT}\n")
        f.write(f"Dataset YAML       : {dataset_yaml}\n")
        f.write(f"Total images       : {len(images)}\n")
        f.write(f"Train images       : {train_count}\n")
        f.write(f"Val images         : {val_count}\n")
        f.write(f"Negative images    : {negative_count}\n")
        f.write(f"Total boxes        : {total_boxes}\n")
        f.write(f"Phone boxes        : {total_phone}\n")
        f.write(f"Walkie boxes       : {total_walkie}\n")
        f.write(f"Skipped labels     : {total_skipped}\n")
        f.write(f"Class names file   : {names_file}\n")
        f.write(f"Old ID to name     : {old_id_to_name}\n")
        f.write(f"Old ID to new ID   : {old_to_new}\n")
        f.write(f"Report CSV         : {REPORT_CSV}\n")
        f.write("==================================================\n")

    print("")
    print("==================================================")
    print("STEP 11/20 DONE")
    print("==================================================")
    print(f"Total images    : {len(images)}")
    print(f"Train images    : {train_count}")
    print(f"Val images      : {val_count}")
    print(f"Negative images : {negative_count}")
    print(f"Total boxes     : {total_boxes}")
    print(f"Phone boxes     : {total_phone}")
    print(f"Walkie boxes    : {total_walkie}")
    print(f"Skipped labels  : {total_skipped}")
    print(f"YOLO dataset    : {YOLO_OUT}")
    print(f"Dataset YAML    : {dataset_yaml}")
    print(f"Summary         : {SUMMARY_TXT}")
    print("==================================================")


if __name__ == "__main__":
    main()
