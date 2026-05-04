import argparse
import os
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def collect_images(src_dir: str):
    src = Path(src_dir)
    images = []
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return sorted(images)


def copy_pair(img_path: Path, labels_src_root: Path, dst_img_dir: Path, dst_lbl_dir: Path):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    dst_img_path = dst_img_dir / img_path.name
    shutil.copy2(img_path, dst_img_path)

    label_name = img_path.stem + ".txt"
    src_label_path = labels_src_root / label_name
    dst_label_path = dst_lbl_dir / label_name

    if src_label_path.exists():
        shutil.copy2(src_label_path, dst_label_path)
    else:
        dst_label_path.write_text("", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-src", required=True)
    parser.add_argument("--labels-src", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Tổng train + val + test phải bằng 1.0")

    images = collect_images(args.images_src)
    if not images:
        raise RuntimeError("Không tìm thấy ảnh nguồn")

    rng = random.Random(args.seed)
    rng.shuffle(images)

    n = len(images)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    n_test = n - n_train - n_val

    train_items = images[:n_train]
    val_items = images[n_train:n_train + n_val]
    test_items = images[n_train + n_val:]

    dataset_root = Path(args.dataset_root)
    labels_src_root = Path(args.labels_src)

    for split_name, items in [
        ("train", train_items),
        ("val", val_items),
        ("test", test_items),
    ]:
        dst_img_dir = dataset_root / "images" / split_name
        dst_lbl_dir = dataset_root / "labels" / split_name
        ensure_dir(str(dst_img_dir))
        ensure_dir(str(dst_lbl_dir))

        for img_path in items:
            copy_pair(img_path, labels_src_root, dst_img_dir, dst_lbl_dir)

    print(f"Tổng ảnh: {n}")
    print(f"Train: {len(train_items)}")
    print(f"Val:   {len(val_items)}")
    print(f"Test:  {len(test_items)}")
    print("Chia dataset xong.")


if __name__ == "__main__":
    main()