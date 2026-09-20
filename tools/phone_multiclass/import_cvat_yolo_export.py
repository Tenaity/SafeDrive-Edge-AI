import argparse
import os
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_files_recursive(root: Path, exts: set[str]):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvat-root", required=True, help="Folder giải nén export CVAT")
    parser.add_argument("--dst-images", required=True, help="images_raw")
    parser.add_argument("--dst-labels", required=True, help="labels_raw")
    args = parser.parse_args()

    cvat_root = Path(args.cvat_root)
    dst_images = Path(args.dst_images)
    dst_labels = Path(args.dst_labels)

    ensure_dir(str(dst_images))
    ensure_dir(str(dst_labels))

    if not cvat_root.exists():
        raise RuntimeError(f"Không tìm thấy folder CVAT export: {cvat_root}")

    image_files = find_files_recursive(cvat_root, IMAGE_EXTS)
    label_files = find_files_recursive(cvat_root, {".txt"})

    if not image_files:
        print("[WARN] Không tìm thấy ảnh trong export CVAT.")
    if not label_files:
        print("[WARN] Không tìm thấy txt label trong export CVAT.")

    copied_images = 0
    copied_labels = 0

    for img_path in image_files:
        dst_path = dst_images / img_path.name
        if not dst_path.exists():
            shutil.copy2(img_path, dst_path)
            copied_images += 1

    for lbl_path in label_files:
        dst_path = dst_labels / lbl_path.name
        shutil.copy2(lbl_path, dst_path)
        copied_labels += 1

    print(f"Copied images: {copied_images}")
    print(f"Copied labels: {copied_labels}")
    print("Done import_cvat_yolo_export.")


if __name__ == "__main__":
    main()