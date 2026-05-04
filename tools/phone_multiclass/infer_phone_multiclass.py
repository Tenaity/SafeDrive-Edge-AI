from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO


CLASS_NAMES = {
    0: "phone",
    1: "walkie_talkie",
    2: "mouse",
    3: "cigarette_pack",
    4: "wallet",
    5: "power_bank",
    6: "remote",
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def draw_result(frame, boxes, confs, clss):
    for box, conf, cls_id in zip(boxes, confs, clss):
        x1, y1, x2, y2 = map(int, box)
        label = CLASS_NAMES.get(int(cls_id), str(int(cls_id)))
        text = f"{label} {float(conf):.2f}"

        color = (0, 255, 0)
        if int(cls_id) == 0:
            color = (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            text,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def infer_images(model, source_dir: Path, out_dir: Path, conf: float):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_exts]
    files = sorted(files)

    if not files:
        raise RuntimeError(f"Không tìm thấy ảnh trong: {source_dir}")

    ensure_dir(out_dir)

    for img_path in files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        results = model.predict(source=str(img_path), conf=conf, verbose=False)
        if not results:
            continue

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and r.boxes.xyxy is not None else []
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None and r.boxes.conf is not None else []
        clss = r.boxes.cls.cpu().numpy() if r.boxes is not None and r.boxes.cls is not None else []

        draw_result(frame, boxes, confs, clss)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), frame)

    print(f"[DONE] Inference images xong. Output: {out_dir}")


def infer_video(model, video_path: Path, out_path: Path, conf: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    ensure_dir(out_path.parent)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        results = model.predict(source=frame, conf=conf, verbose=False)
        if results:
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and r.boxes.xyxy is not None else []
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None and r.boxes.conf is not None else []
            clss = r.boxes.cls.cpu().numpy() if r.boxes is not None and r.boxes.cls is not None else []
            draw_result(frame, boxes, confs, clss)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[DONE] Inference video xong. Output: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["images", "video"], required=True)
    parser.add_argument("--source", required=True, help="Folder ảnh hoặc file video")
    parser.add_argument("--out", required=True, help="Folder output hoặc file video output")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "runs" / "phone_multiclass_yolo11s_cpu_test" / "weights" / "best.pt"

    if not model_path.exists():
        raise RuntimeError(f"Không tìm thấy model: {model_path}")

    print(f"Using model: {model_path}")
    model = YOLO(str(model_path))

    source = Path(args.source)
    out = Path(args.out)

    if args.mode == "images":
        infer_images(model, source, out, args.conf)
    else:
        infer_video(model, source, out, args.conf)


if __name__ == "__main__":
    main()