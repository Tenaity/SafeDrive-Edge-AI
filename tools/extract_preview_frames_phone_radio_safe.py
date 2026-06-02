import os
import csv
import cv2
import time
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

OUT_DIR = DATASET / "01_frames_preview_safe"
INV_DIR = DATASET / "00_inventory"

OUT_DIR.mkdir(parents=True, exist_ok=True)
INV_DIR.mkdir(parents=True, exist_ok=True)

REPORT_CSV = INV_DIR / "preview_frames_safe_report.csv"
SUMMARY_TXT = INV_DIR / "preview_frames_safe_summary.txt"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

MAX_FRAMES_PER_VIDEO = 20
SAMPLE_EVERY_SEC = 10.0
MAX_READ_SEC_PER_VIDEO = 900.0
MAX_WALL_TIME_PER_VIDEO = 60.0
JPEG_QUALITY = 92


def safe_float(x, default=0.0):
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def extract_video(video_path: Path, video_index: int, total_videos: int):
    row = {
        "VideoIndex": video_index,
        "FileName": video_path.name,
        "FullPath": str(video_path),
        "OutputFolder": "",
        "FPS": 0.0,
        "Width": 0,
        "Height": 0,
        "SavedFrames": 0,
        "Status": "BAD",
        "Error": "",
    }

    out_sub = OUT_DIR / video_path.stem
    out_sub.mkdir(parents=True, exist_ok=True)
    row["OutputFolder"] = str(out_sub)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        row["Error"] = "cap_not_opened"
        print(f"[{video_index:03d}/{total_videos}] BAD | {video_path.name} | cap_not_opened")
        return row

    try:
        fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 20.0)
        if fps <= 1:
            fps = 20.0

        width = int(safe_float(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 0))
        height = int(safe_float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 0))

        row["FPS"] = round(fps, 3)
        row["Width"] = width
        row["Height"] = height

        sample_interval_frames = max(int(fps * SAMPLE_EVERY_SEC), 1)
        max_read_frames = int(fps * MAX_READ_SEC_PER_VIDEO)

        frame_idx = 0
        saved = 0
        start_time = time.time()

        while saved < MAX_FRAMES_PER_VIDEO and frame_idx < max_read_frames:
            if time.time() - start_time > MAX_WALL_TIME_PER_VIDEO:
                row["Error"] = "timeout_per_video"
                break

            ok, frame = cap.read()

            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                break

            if frame_idx % sample_interval_frames == 0:
                saved += 1
                out_name = f"{video_path.stem}_safe_{saved:03d}_f{frame_idx:08d}.jpg"
                out_path = out_sub / out_name

                cv2.imwrite(
                    str(out_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )

            frame_idx += 1

        row["SavedFrames"] = saved
        row["Status"] = "OK" if saved > 0 else "BAD"

        if saved == 0 and not row["Error"]:
            row["Error"] = "no_frame_saved"

        print(
            f"[{video_index:03d}/{total_videos}] {row['Status']} | "
            f"{video_path.name} | {width}x{height} | fps={round(fps, 3)} | "
            f"saved={saved} | read_frames={frame_idx} | {row['Error']}"
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
    print("STEP 3/20 - SAFE PREVIEW FRAMES")
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
        "FPS",
        "Width",
        "Height",
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
        f.write("SAFE DRIVE PHONE/RADIO - SAFE PREVIEW EXTRACTION\n")
        f.write("==================================================\n")
        f.write(f"Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video source: {VIDEO_SRC}\n")
        f.write(f"Output      : {OUT_DIR}\n")
        f.write(f"Total videos: {total}\n")
        f.write(f"OK videos   : {ok}\n")
        f.write(f"Bad videos  : {bad}\n")
        f.write(f"Total frames: {saved_total}\n")
        f.write(f"CSV report  : {REPORT_CSV}\n")
        f.write("\nBad files:\n")
        for r in rows:
            if r["Status"] != "OK":
                f.write(f"- {r['FileName']} | {r['Error']}\n")
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
