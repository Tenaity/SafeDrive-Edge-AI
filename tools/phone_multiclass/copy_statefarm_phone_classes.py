import argparse
import os
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_images_from_class_dirs(src_root: str, out_dir: str, class_dirs: list[str], max_per_class: int = 0):
    src_root_path = Path(src_root)
    out_dir_path = Path(out_dir)

    ensure_dir(str(out_dir_path))

    total_copied = 0

    for cls_name in class_dirs:
        cls_dir = src_root_path / cls_name
        if not cls_dir.exists():
            print(f"[WARN] Không thấy folder: {cls_dir}")
            continue

        images = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        images = sorted(images)

        if max_per_class > 0:
            images = images[:max_per_class]

        copied_this_class = 0
        for idx, img_path in enumerate(images):
            dst_name = f"statefarm_{cls_name}_{idx:06d}{img_path.suffix.lower()}"
            dst_path = out_dir_path / dst_name
            shutil.copy2(img_path, dst_path)
            copied_this_class += 1
            total_copied += 1

        print(f"[OK] {cls_name}: copied {copied_this_class} ảnh")

    print(f"[DONE] Tổng ảnh đã copy: {total_copied}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True, help="Folder gốc chứa c1,c2... (ví dụ ...\\statefarm\\train)")
    parser.add_argument("--out-dir", required=True, help="Folder đích images_raw")
    parser.add_argument("--max-per-class", type=int, default=0, help="Giới hạn số ảnh mỗi class, 0 = không giới hạn")
    args = parser.parse_args()

    # Lấy thêm c5 theo yêu cầu
    class_dirs = ["c1", "c2", "c3", "c4", "c5"]

    copy_images_from_class_dirs(
        src_root=args.src_root,
        out_dir=args.out_dir,
        class_dirs=class_dirs,
        max_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()