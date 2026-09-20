import argparse
import json
import os
import shutil
from pathlib import Path


COCO_NAME_TO_ID = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "traffic light": 10,
    "fire hydrant": 11,
    "stop sign": 13,
    "parking meter": 14,
    "bench": 15,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "suitcase": 33,
    "frisbee": 34,
    "skis": 35,
    "snowboard": 36,
    "sports ball": 37,
    "kite": 38,
    "baseball bat": 39,
    "baseball glove": 40,
    "skateboard": 41,
    "surfboard": 42,
    "tennis racket": 43,
    "bottle": 44,
    "wine glass": 46,
    "cup": 47,
    "fork": 48,
    "knife": 49,
    "spoon": 50,
    "bowl": 51,
    "banana": 52,
    "apple": 53,
    "sandwich": 54,
    "orange": 55,
    "broccoli": 56,
    "carrot": 57,
    "hot dog": 58,
    "pizza": 59,
    "donut": 60,
    "cake": 61,
    "chair": 62,
    "couch": 63,
    "potted plant": 64,
    "bed": 65,
    "dining table": 67,
    "toilet": 70,
    "tv": 72,
    "laptop": 73,
    "mouse": 74,
    "remote": 75,
    "keyboard": 76,
    "cell phone": 77,
    "microwave": 78,
    "oven": 79,
    "toaster": 80,
    "sink": 81,
    "refrigerator": 82,
    "book": 84,
    "clock": 85,
    "vase": 86,
    "scissors": 87,
    "teddy bear": 88,
    "hair drier": 89,
    "toothbrush": 90,
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def yolo_bbox_from_coco_xywh(x, y, w, h, img_w, img_h):
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="Ví dụ: .../train2017")
    parser.add_argument("--ann-json", required=True, help="Ví dụ: .../annotations/instances_train2017.json")
    parser.add_argument("--out-images", required=True)
    parser.add_argument("--out-labels", required=True)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    ann_json = Path(args.ann_json)
    out_images = Path(args.out_images)
    out_labels = Path(args.out_labels)

    ensure_dir(str(out_images))
    ensure_dir(str(out_labels))

    data = load_json(str(ann_json))

    target_names = ["cell phone", "mouse", "remote"]
    target_coco_ids = {COCO_NAME_TO_ID[name] for name in target_names}

    # map COCO -> class mới của ta
    coco_to_new = {
        COCO_NAME_TO_ID["cell phone"]: 0,  # phone
        COCO_NAME_TO_ID["mouse"]: 2,       # mouse
        COCO_NAME_TO_ID["remote"]: 6,      # remote
    }

    image_id_to_info = {}
    for img in data["images"]:
        image_id_to_info[img["id"]] = img

    ann_by_image = {}
    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in target_coco_ids:
            continue
        if ann.get("iscrowd", 0) == 1:
            continue

        img_id = ann["image_id"]
        ann_by_image.setdefault(img_id, []).append(ann)

    copied = 0
    labeled = 0

    for img_id, anns in ann_by_image.items():
        info = image_id_to_info.get(img_id)
        if info is None:
            continue

        file_name = info["file_name"]
        img_w = info["width"]
        img_h = info["height"]

        src_img = images_dir / file_name
        if not src_img.exists():
            continue

        dst_img = out_images / file_name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
            copied += 1

        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue

            new_cls = coco_to_new[ann["category_id"]]
            cx, cy, nw, nh = yolo_bbox_from_coco_xywh(x, y, w, h, img_w, img_h)
            lines.append(f"{new_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path = out_labels / (Path(file_name).stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        labeled += 1

    print(f"Copied images: {copied}")
    print(f"Labeled files: {labeled}")
    print("Done extract_coco_subset.")


if __name__ == "__main__":
    main()