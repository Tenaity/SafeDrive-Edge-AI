from pathlib import Path
import cv2
import random


ROOT = Path(r"D:\project_detectfaceandphone\SafeDrive-Edge-AI")
DATASET = ROOT / "datasets" / "phone_radio_cabin" / "05_yolo_dataset_unique80"
OUT_DIR = ROOT / "datasets" / "phone_radio_cabin" / "06_label_preview_unique80"

OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "phone",
    1: "walkie_talkie",
}

MAX_PREVIEW = 200


def yolo_to_xyxy(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    cls = int(float(parts[0]))
    x = float(parts[1])
    y = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])

    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return cls, x1, y1, x2, y2


def draw_label(img, text, x1, y1):
    y_text = max(20, y1 - 8)
    cv2.putText(
        img,
        text,
        (x1, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def process_split(split):
    img_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = sorted(img_dir.glob("*.jpg"))
    labeled = []

    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]

        if lines:
            labeled.append((img_path, label_path, lines))

    random.seed(42)
    random.shuffle(labeled)

    picked = labeled[:MAX_PREVIEW]

    out_split = OUT_DIR / split
    out_split.mkdir(parents=True, exist_ok=True)

    phone_count = 0
    walkie_count = 0
    saved = 0

    for img_path, label_path, lines in picked:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        for line in lines:
            item = yolo_to_xyxy(line, w, h)
            if item is None:
                continue

            cls, x1, y1, x2, y2 = item
            name = CLASS_NAMES.get(cls, f"class_{cls}")

            if cls == 0:
                phone_count += 1
                box_color = (0, 255, 255)
            elif cls == 1:
                walkie_count += 1
                box_color = (255, 0, 255)
            else:
                box_color = (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
            draw_label(img, name, x1, y1)

        out_path = out_split / img_path.name
        cv2.imwrite(str(out_path), img)
        saved += 1

    return {
        "split": split,
        "labeled_images": len(labeled),
        "preview_saved": saved,
        "phone_boxes_in_preview": phone_count,
        "walkie_boxes_in_preview": walkie_count,
        "out_dir": str(out_split),
    }


def main():
    print("==================================================")
    print("STEP 12/20 - PREVIEW YOLO LABELS")
    print("==================================================")
    print(f"Dataset : {DATASET}")
    print(f"Output  : {OUT_DIR}")
    print("==================================================")

    results = []
    for split in ["train", "val"]:
        results.append(process_split(split))

    print("")
    print("==================================================")
    print("STEP 12/20 DONE")
    print("==================================================")

    for r in results:
        print(f"Split                 : {r['split']}")
        print(f"Labeled images         : {r['labeled_images']}")
        print(f"Preview saved          : {r['preview_saved']}")
        print(f"Phone boxes in preview : {r['phone_boxes_in_preview']}")
        print(f"Walkie boxes in preview: {r['walkie_boxes_in_preview']}")
        print(f"Output                 : {r['out_dir']}")
        print("--------------------------------------------------")

    print(f"Open preview folder: {OUT_DIR}")
    print("==================================================")


if __name__ == "__main__":
    main()
