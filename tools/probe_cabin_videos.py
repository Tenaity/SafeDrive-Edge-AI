import csv
import cv2
from pathlib import Path
from datetime import datetime


ROOT = Path(r"D:\project_detectfaceandphone\SafeDrive-Edge-AI")
VIDEO_SRC = ROOT / "raw_data" / "realdata"
DATASET = ROOT / "datasets" / "phone_radio_cabin"
OUT_DIR = DATASET / "00_inventory"

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "video_probe.csv"
OUT_SUMMARY = OUT_DIR / "video_probe_summary.txt"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def safe_float(x, default=0.0):
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def probe_video(path: Path):
    row = {
        "FileName": path.name,
        "FullPath": str(path),
        "Extension": path.suffix.lower(),
        "SizeMB": round(path.stat().st_size / (1024 * 1024), 2),
        "Readable": False,
        "Width": 0,
        "Height": 0,
        "FPS": 0.0,
        "FrameCount": 0,
        "DurationSec": 0.0,
        "FirstFrameOK": False,
        "Error": "",
    }

    cap = None

    try:
        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            row["Error"] = "cap_not_opened"
            return row

        fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 0.0)
        frame_count = int(safe_float(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0.0))
        width = int(safe_float(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 0.0))
        height = int(safe_float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 0.0))

        duration = 0.0
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps

        ok, frame = cap.read()
        first_ok = bool(ok and frame is not None and getattr(frame, "size", 0) > 0)

        row.update({
            "Readable": bool(width > 0 and height > 0 and fps > 0 and frame_count > 0),
            "Width": width,
            "Height": height,
            "FPS": round(fps, 3),
            "FrameCount": frame_count,
            "DurationSec": round(duration, 2),
            "FirstFrameOK": first_ok,
            "Error": "" if first_ok else "first_frame_read_failed",
        })

        return row

    except Exception as e:
        row["Error"] = str(e)
        return row

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


def main():
    videos = []
    for p in VIDEO_SRC.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)

    videos = sorted(videos, key=lambda x: str(x).lower())

    print("==================================================")
    print("SAFE DRIVE VIDEO PROBE")
    print("==================================================")
    print(f"Video source : {VIDEO_SRC}")
    print(f"Total videos : {len(videos)}")
    print(f"Output CSV   : {OUT_CSV}")
    print("==================================================")

    rows = []

    for i, video in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] {video.name}")
        rows.append(probe_video(video))

    fieldnames = [
        "FileName",
        "FullPath",
        "Extension",
        "SizeMB",
        "Readable",
        "Width",
        "Height",
        "FPS",
        "FrameCount",
        "DurationSec",
        "FirstFrameOK",
        "Error",
    ]

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    readable = sum(1 for r in rows if r["Readable"])
    first_ok = sum(1 for r in rows if r["FirstFrameOK"])
    bad = total - readable

    total_duration = sum(float(r["DurationSec"]) for r in rows if r["DurationSec"])
    total_size = sum(float(r["SizeMB"]) for r in rows if r["SizeMB"])

    resolutions = {}
    fps_groups = {}

    for r in rows:
        res = f'{r["Width"]}x{r["Height"]}'
        resolutions[res] = resolutions.get(res, 0) + 1

        fps = str(r["FPS"])
        fps_groups[fps] = fps_groups.get(fps, 0) + 1

    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("SAFE DRIVE PHONE/RADIO DATASET - VIDEO PROBE\n")
        f.write("==================================================\n")
        f.write(f"Time          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video source  : {VIDEO_SRC}\n")
        f.write(f"Total videos  : {total}\n")
        f.write(f"Readable      : {readable}\n")
        f.write(f"Bad/unreadable: {bad}\n")
        f.write(f"First frame OK: {first_ok}\n")
        f.write(f"Total size MB : {round(total_size, 2)}\n")
        f.write(f"Total duration: {round(total_duration / 60, 2)} minutes\n")
        f.write("\nResolutions:\n")
        for k, v in sorted(resolutions.items()):
            f.write(f"- {k}: {v}\n")
        f.write("\nFPS groups:\n")
        for k, v in sorted(fps_groups.items()):
            f.write(f"- {k}: {v}\n")
        f.write("\nBad files:\n")
        for r in rows:
            if not r["Readable"] or not r["FirstFrameOK"]:
                f.write(f'- {r["FileName"]} | error={r["Error"]}\n')
        f.write("==================================================\n")

    print("")
    print("==================================================")
    print("STEP 2/20 DONE")
    print("==================================================")
    print(f"Total videos   : {total}")
    print(f"Readable       : {readable}")
    print(f"Bad/unreadable : {bad}")
    print(f"Total duration : {round(total_duration / 60, 2)} minutes")
    print(f"CSV            : {OUT_CSV}")
    print(f"Summary        : {OUT_SUMMARY}")
    print("==================================================")


if __name__ == "__main__":
    main()
