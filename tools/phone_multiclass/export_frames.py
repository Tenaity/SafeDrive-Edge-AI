import argparse
import os
from pathlib import Path

import cv2


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_frames(
    video_path: str,
    output_dir: str,
    every_n_frames: int = 10,
    prefix: str = "frame",
    max_frames: int = 0,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    ensure_dir(output_dir)

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % every_n_frames == 0:
            out_name = f"{prefix}_{saved_idx:06d}.jpg"
            out_path = os.path.join(output_dir, out_name)
            ok = cv2.imwrite(out_path, frame)
            if not ok:
                raise RuntimeError(f"Ghi ảnh thất bại: {out_path}")
            saved_idx += 1

            if max_frames > 0 and saved_idx >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"Xong. Đã lưu {saved_idx} ảnh vào: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Đường dẫn video đầu vào")
    parser.add_argument("--out", required=True, help="Thư mục ảnh đầu ra")
    parser.add_argument("--step", type=int, default=10, help="Lấy 1 frame mỗi N frame")
    parser.add_argument("--prefix", default="frame", help="Tiền tố tên file ảnh")
    parser.add_argument("--max", type=int, default=0, help="Số ảnh tối đa, 0 = không giới hạn")
    args = parser.parse_args()

    export_frames(
        video_path=args.video,
        output_dir=args.out,
        every_n_frames=args.step,
        prefix=args.prefix,
        max_frames=args.max,
    )


if __name__ == "__main__":
    main()