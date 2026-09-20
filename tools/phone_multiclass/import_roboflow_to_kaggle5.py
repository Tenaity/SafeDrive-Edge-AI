import argparse
import shutil
from pathlib import Path
from collections import defaultdict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Target 5-class mapping
TARGET_CLASSES = {
    "phone": 0,
    "walkie_talkie": 1,
    "mouse": 2,
    "cigarette_pack": 3,
    "remote": 4,
}

# Roboflow class-name normalization
NORMALIZE_MAP = {
    "phone": "phone",
    "cell phone": "phone",
    "mobile phone": "phone",
    "smartphone": "phone",

    "walkie talkie": "walkie_talkie",
    "walkie_talkie": "walkie_talkie",
    "2 way radio": "walkie_talkie",
    "two way radio": "walkie_talkie",
    "walkie talkie-2 way radio": "walkie_talkie",
    "walkie talkie 2 way radio": "walkie_talkie",
    "walkie talkie 2-way radio": "walkie_talkie",
    "radio": "walkie_talkie",

    "mouse": "mouse",
    "computer mouse": "mouse",

    "pack": "cigarette_pack",
    "cigarette_pack": "cigarette_pack",
    "cigarette pack": "cigarette_pack",
    "cigarettes pack": "cigarette_pack",
    "cigarettes packs": "cigarette_pack",

    "remote": "remote",
    "remote control": "remote",
}


def normalize_name(name: str) -> str:
    x = name.strip().lower()
    x = x.replace("-", " ")
    x = x.replace("_", " ")
    x = " ".join(x.split())
    return NORMALIZE_MAP.get(x, "")


def find_data_yaml(root: Path):
    candidates = list(root.rglob("data.yaml"))
    if not candidates:
        raise RuntimeError(f"Không tìm thấy data.yaml trong {root}")
    # ưu tiên file gần root hơn
    candidates = sorted(candidates, key=lambda p: len(p.parts))
    return candidates[0]


def load_class_map_from_yaml_text(yaml_path: Path):
    text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    names_block = False
    id_to_name = {}

    # Hỗ trợ 2 dạng phổ biến:
    # names:
    #   0: phone
    #   1: remote
    #
    # hoặc:
    # names: ['phone', 'remote']
    for idx, line in enumerate(lines):
        s = line.strip()
        if s.startswith("names:"):
            rest = s[len("names:"):].strip()

            if rest.startswith("[") and rest.endswith("]"):
                raw = rest.strip()[1:-1]
                parts = [x.strip().strip("'").strip('"') for x in raw.split(",") if x.strip()]
                for i, name in enumerate(parts):
                    id_to_name[i] = name
                return id_to_name

            names_block = True
            continue

        if names_block:
            if not s:
                continue
            if ":" in s:
                left, right = s.split(":", 1)
                left = left.strip()
                right = right.strip().strip("'").strip('"')
                if left.isdigit():
                    id_to_name[int(left)] = right
                    continue
            # gặp block khác thì dừng
            if not s[0].isdigit():
                break

    if not id_to_name:
        raise RuntimeError(f"Không đọc được names từ {yaml_path}")

    return id_to_name


def find_split_dirs(root: Path):
    # hỗ trợ train/valid/test hoặc train/val/test
    split_map = {}
    for name in ["train", "valid", "val", "test"]:
        for p in root.rglob(name):
            if p.is_dir():
                split_map.setdefault(name, []).append(p)

    result = {}
    if "train" in split_map:
        result["train"] = sorted(split_map["train"], key=lambda p: len(p.parts))[0]
    if "valid" in split_map:
        result["val"] = sorted(split_map["valid"], key=lambda p: len(p.parts))[0]
    elif "val" in split_map:
        result["val"] = sorted(split_map["val"], key=lambda p: len(p.parts))[0]
    if "test" in split_map:
        result["test"] = sorted(split_map["test"], key=lambda p: len(p.parts))[0]

    if "train" not in result:
        raise RuntimeError(f"Không tìm thấy split train trong {root}")

    return result


def resolve_images_labels_dir(split_dir: Path):
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if images_dir.is_dir() and labels_dir.is_dir():
        return images_dir, labels_dir

    # một số export đặt ảnh/txt cùng split root
    imgs = [p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    txts = [p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    if imgs or txts:
        return split_dir, split_dir

    raise RuntimeError(f"Không xác định được images/labels trong {split_dir}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def convert_one_label_file(src_txt: Path, dst_txt: Path, id_to_name: dict):
    kept = []
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
            old_name = id_to_name.get(old_id, "")
            new_name = normalize_name(old_name)
            if not new_name:
                continue
            if new_name not in TARGET_CLASSES:
                continue

            new_id = TARGET_CLASSES[new_name]
            kept.append(" ".join([str(new_id)] + parts[1:]))

    with open(dst_txt, "w", encoding="utf-8") as f:
        if kept:
            f.write("\n".join(kept) + "\n")


def import_roboflow_dataset(src_root: Path, dst_images_raw: Path, dst_labels_raw: Path, prefix: str):
    data_yaml = find_data_yaml(src_root)
    id_to_name = load_class_map_from_yaml_text(data_yaml)
    splits = find_split_dirs(src_root)

    stats = defaultdict(int)

    for split_name, split_dir in splits.items():
        images_dir, labels_dir = resolve_images_labels_dir(split_dir)

        image_files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        image_files = sorted(image_files)

        for img_path in image_files:
            stem = img_path.stem
            src_txt = labels_dir / f"{stem}.txt"

            new_img_name = f"{prefix}_{split_name}_{img_path.name}"
            dst_img = dst_images_raw / new_img_name
            shutil.copy2(img_path, dst_img)
            stats["images"] += 1

            dst_txt = dst_labels_raw / f"{Path(new_img_name).stem}.txt"
            if src_txt.exists():
                convert_one_label_file(src_txt, dst_txt, id_to_name)
                stats["labels"] += 1
            else:
                # background ảnh không có txt
                dst_txt.write_text("", encoding="utf-8")
                stats["empty_labels"] += 1

    return dict(stats), id_to_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True, help="Roboflow dataset root")
    parser.add_argument("--dst-images-raw", required=True)
    parser.add_argument("--dst-labels-raw", required=True)
    parser.add_argument("--prefix", required=True, help="Tên prefix nguồn, ví dụ walkie hoặc cigarette")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_images_raw = Path(args.dst_images_raw)
    dst_labels_raw = Path(args.dst_labels_raw)

    ensure_dir(dst_images_raw)
    ensure_dir(dst_labels_raw)

    stats, id_to_name = import_roboflow_dataset(
        src_root=src_root,
        dst_images_raw=dst_images_raw,
        dst_labels_raw=dst_labels_raw,
        prefix=args.prefix,
    )

    print("[DONE] import_roboflow_dataset")
    print("source:", src_root)
    print("class_map:", id_to_name)
    print("stats:", stats)


if __name__ == "__main__":
    main()