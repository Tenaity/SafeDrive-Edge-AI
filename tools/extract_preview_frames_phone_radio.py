import os
import csv
import cv2
import math
from pathlib import Path
from datetime import datetime
from typing import Any, cast

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"

try:
    cast(Any, cv2).setLogLevel(0)
except Exception:
    pass


ROOT = Path(r"D:\project_detectfaceandphone\SafeDrive-Edge-AI")
VIDEO_SRC = ROOT / "raw_data" / "realdata"
DATASET = ROOT / "datasets" / "phone_radio_cabin"

OUT_DIR = DATASET / "01_frames_preview"
INV_DIR = DATASET / "00_inventory"

OUT_DIR.mkdir(parents=True, exist_ok=True)
INV_DIR.mkdir(parents=True, exist_ok=True)

REPORT_CSV = INV_DIR / "preview_frames_report.csv"
SUMMARY_TXT = INV_DIR / "preview_frames_summary.txt"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

MAX_FRAMES_PER_VIDEO = 30
JPEG_QUALITY = 92


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def make_sample_indices(frame_count: int, max_frames: int):
    if frame_count <= 0:
        return []

    if frame_count <= max_frames:
        return list(range(frame_count))

    start = int(frame_count * 0.05)
    end = int(frame_count * 0.95)

    if end <= start:
        start = 0
        end = frame_count - 1

    if max_frames <= 1:
        return [start]

    step = (end - start) / float(max_frames - 1)

    indices = []
    for i in range(max_frames):
        idx = int(round(start + i * step))
        idx = max(0, min(idx, frame_count - 1))
        indices.append(idx)

    # remove duplicates but preserve order
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            unique.append(idx)
            seen.add(idx)

    return unique


def extract_video(video_path: Path, video_index: int, total_videos: int):
    cap = cv2.VideoCapture(str(video_path))

    row = {
        "VideoIndex": video_index,
        "FileName": video_path.name,
        "FullPath": str(video_path),
        "OutputFolder": "",
        "FrameCount": 0,
        "FPS": 0.0,
        "Width": 0,
        "Height": 0,
        "RequestedFrames": 0,
        "SavedFrames": 0,
        "Status": "BAD",
        "Error": "",
    }

    if not cap.isOpened():
        row["Error"] = "cap_not_opened"
        print(f"[{video_index:03d}/{total_videos}] BAD | {video_path.name} | cap_not_opened")
        return row

    try:
        frame_count = safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0)
        fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 0.0)
        width = safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 0)
        height = safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 0)

        row["FrameCount"] = frame_count
        row["FPS"] = round(fps, 3)
        row["Width"] = width
        row["Height"] = height

        sample_indices = make_sample_indices(frame_count, MAX_FRAMES_PER_VIDEO)

        out_sub = OUT_DIR / video_path.stem
        out_sub.mkdir(parents=True, exist_ok=True)

        row["OutputFolder"] = str(out_sub)
        row["RequestedFrames"] = len(sample_indices)

        saved = 0

        for n, frame_idx in enumerate(sample_indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()

            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                continue

            out_name = f"{video_path.stem}_preview_{n:03d}_f{frame_idx:08d}.jpg"
            out_path = out_sub / out_name

            cv2.imwrite(
                str(out_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )

            saved += 1

        row["SavedFrames"] = saved
        row["Status"] = "OK" if saved > 0 else "BAD"
        row["Error"] = "" if saved > 0 else "no_frame_saved"

        print(
            f"[{video_index:03d}/{total_videos}] {row['Status']} | "
            f"{video_path.name} | {width}x{height} | fps={round(fps, 3)} | "
            f"saved={saved}/{len(sample_indices)}"
        )

        return row

    except Exception as e:
        row["Error"] = str(e)
        print(f"[{video_index:03d}/{total_videos}] BAD | {video_path.name} | {e}")
        return row

    finally:
        try:
            cap.release()
        except Exception:
            pass


def main():
    videos = sorted(
        [p for p in VIDEO_SRC.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
        key=lambda x: str(x).lower()
    )

    print("==================================================")
    print("STEP 3/20 - EXTRACT PREVIEW FRAMES")
    print("==================================================")
    print(f"Video source : {VIDEO_SRC}")
    print(f"Output       : {OUT_DIR}")
    print(f"Total videos : {len(videos)}")
    print(f"Max/video    : {MAX_FRAMES_PER_VIDEO}")
    print("==================================================")

    rows = []

    for i, video in enumerate(videos, start=1):
        rows.append(extract_video(video, i, len(videos)))

    fieldnames = [
        "VideoIndex",
        "FileName",
        "FullPath",
        "OutputFolder",
        "FrameCount",
        "FPS",
        "Width",
        "Height",
        "RequestedFrames",
        "SavedFrames",
        "Status",
        "Error",
    ]

    with REPORT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    ok = sum(1 for r in rows if r["Status"] == "OK")
    bad = total - ok
    saved_total = sum(int(r["SavedFrames"]) for r in rows)

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("SAFE DRIVE PHONE/RADIO - PREVIEW FRAME EXTRACTION\n")
        f.write("==================================================\n")
        f.write(f"Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video source: {VIDEO_SRC}\n")
        f.write(f"Output      : {OUT_DIR}\n")
        f.write(f"Total videos: {total}\n")
        f.write(f"OK videos   : {ok}\n")
        f.write(f"Bad videos  : {bad}\n")
        f.write(f"Total frames: {saved_total}\n")
        f.write(f"CSV report  : {REPORT_CSV}\n")
        f.write("==================================================\n")

    print("")
    print("==================================================")
    print("STEP 3/20 DONE")
    print("==================================================")
    print(f"Total videos : {total}")
    print(f"OK videos    : {ok}")
    print(f"Bad videos   : {bad}")
    print(f"Total frames : {saved_total}")
    print(f"Output       : {OUT_DIR}")
    print(f"CSV report   : {REPORT_CSV}")
    print(f"Summary      : {SUMMARY_TXT}")
    print("==================================================")


if __name__ == "__main__":
    main()
