import argparse
import shutil
from pathlib import Path
from collections import defaultdict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Dataset cũ 7 class:
# 0 phone
# 1 walkie_talkie
# 2 mouse
# 3 cigarette_pack
# 4 wallet
# 5 power_bank
# 6 remote

OLD_TO_NEW = {
    0: 0,  # phone
    2: 2,  # mouse
    6: 4,  # remote
}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def convert_label_file(src_txt: Path, dst_txt: Path):
    kept = []
    present_new_classes = set()

    with open(src_txt, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue

            parts = s.split()
            if len(parts) < 5:
                continue
            if not parts[0].isdigit():
                continue

            old_id = int(parts[0])
            if old_id not in OLD_TO_NEW:
                continue

            new_id = OLD_TO_NEW[old_id]
            kept.append(" ".join([str(new_id)] + parts[1:]))
            present_new_classes.add(new_id)

    with open(dst_txt, "w", encoding="utf-8") as f:
        if kept:
            f.write("\n".join(kept) + "\n")

    return kept, present_new_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-images", required=True)
    parser.add_argument("--src-labels", required=True)
    parser.add_argument("--dst-images-raw", required=True)
    parser.add_argument("--dst-labels-raw", required=True)
    parser.add_argument("--prefix", default="old")
    args = parser.parse_args()

    src_images = Path(args.src_images)
    src_labels = Path(args.src_labels)
    dst_images_raw = Path(args.dst_images_raw)
    dst_labels_raw = Path(args.dst_labels_raw)

    ensure_dir(dst_images_raw)
    ensure_dir(dst_labels_raw)

    stats = defaultdict(int)
    class_stats = defaultdict(int)

    image_files = [p for p in src_images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    image_files = sorted(image_files)

    for img_path in image_files:
        stem = img_path.stem
        src_txt = src_labels / f"{stem}.txt"
        if not src_txt.exists():
            continue

        kept, present_new_classes = convert_label_file(src_txt, Path("__temp__.txt"))
        try:
            Path("__temp__.txt").unlink(missing_ok=True)
        except Exception:
            pass

        if not kept:
            continue

        new_img_name = f"{args.prefix}_{img_path.name}"
        dst_img = dst_images_raw / new_img_name
        dst_txt = dst_labels_raw / f"{Path(new_img_name).stem}.txt"

        shutil.copy2(img_path, dst_img)

        kept, present_new_classes = convert_label_file(src_txt, dst_txt)
        if not kept:
            try:
                dst_img.unlink(missing_ok=True)
                dst_txt.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        stats["images"] += 1
        stats["labels"] += 1

        for cid in present_new_classes:
            class_stats[cid] += 1

    print("[DONE] import_old_multiclass_to_kaggle5")
    print("stats:", dict(stats))
    print("class_stats:", dict(class_stats))


if __name__ == "__main__":
    main()