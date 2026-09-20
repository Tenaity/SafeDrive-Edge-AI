import csv
import shutil
import subprocess
from pathlib import Path
from datetime import datetime


ROOT = Path(r"D:\project_detectfaceandphone\SafeDrive-Edge-AI")
VIDEO_SRC = ROOT / "raw_data" / "realdata"
DATASET = ROOT / "datasets" / "phone_radio_cabin"

FFMPEG_EXE = ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"

OUT_DIR = DATASET / "01_frames_raw"
INV_DIR = DATASET / "00_inventory"

OUT_DIR.mkdir(parents=True, exist_ok=True)
INV_DIR.mkdir(parents=True, exist_ok=True)

REPORT_CSV = INV_DIR / "extract_all_frames_clean_report.csv"
SUMMARY_TXT = INV_DIR / "extract_all_frames_clean_summary.txt"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

# 1 FPS = moi giay lay 1 anh.
# Neu sau nay can day hon thi doi thanh 2.
FPS_SAMPLE = 1

JPEG_QUALITY = 2
SKIP_EXISTING = True


def count_jpg(folder: Path):
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.glob("*.jpg"))


def find_ffmpeg():
    if FFMPEG_EXE.exists():
        return str(FFMPEG_EXE)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    return None


def extract_video(ffmpeg: str, video_path: Path, index: int, total: int):
    out_sub = OUT_DIR / video_path.stem
    out_sub.mkdir(parents=True, exist_ok=True)

    before = count_jpg(out_sub)

    row = {
        "Index": index,
        "FileName": video_path.name,
        "FullPath": str(video_path),
        "OutputFolder": str(out_sub),
        "FPS_SAMPLE": FPS_SAMPLE,
        "BeforeFrames": before,
        "AfterFrames": before,
        "NewFrames": 0,
        "Status": "PENDING",
        "Error": "",
    }

    if SKIP_EXISTING and before > 0:
        row["Status"] = "SKIP_EXISTING"
        print(f"[{index:03d}/{total}] SKIP | {video_path.name} | existing={before}")
        return row

    # Ten file ngan, tranh loi Windows Explorer/path dai
    out_pattern = str(out_sub / f"{video_path.stem}_%06d.jpg")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(video_path),
        "-an",
        "-map", "0:v:0",
        "-vf", f"fps={FPS_SAMPLE}",
        "-q:v", str(JPEG_QUALITY),
        out_pattern,
    ]

    print(f"[{index:03d}/{total}] START | {video_path.name}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        after = count_jpg(out_sub)
        row["AfterFrames"] = after
        row["NewFrames"] = max(after - before, 0)

        if result.returncode == 0 and after > 0:
            row["Status"] = "OK"
            print(f"[{index:03d}/{total}] OK   | {video_path.name} | frames={after}")
        elif after > 0:
            row["Status"] = "PARTIAL"
            row["Error"] = result.stderr[-1000:]
            print(f"[{index:03d}/{total}] PART | {video_path.name} | frames={after}")
        else:
            row["Status"] = "BAD"
            row["Error"] = result.stderr[-2000:]
            print(f"[{index:03d}/{total}] BAD  | {video_path.name} | no frame")

        return row

    except Exception as e:
        row["Status"] = "ERROR"
        row["Error"] = str(e)
        print(f"[{index:03d}/{total}] ERR  | {video_path.name} | {e}")
        return row


def main():
    ffmpeg = find_ffmpeg()

    print("==================================================")
    print("STEP 3/20 - CLEAN ALL FRAME EXTRACTION BY FFMPEG")
    print("==================================================")
    print(f"VIDEO_SRC  : {VIDEO_SRC}")
    print(f"OUT_DIR    : {OUT_DIR}")
    print(f"FPS_SAMPLE : {FPS_SAMPLE}")
    print("==================================================")

    if ffmpeg is None:
        print("[ERROR] Khong tim thay ffmpeg.exe")
        raise SystemExit(1)

    print(f"[OK] FFmpeg: {ffmpeg}")

    videos = sorted(
        [p for p in VIDEO_SRC.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
        key=lambda x: str(x).lower()
    )

    print(f"[OK] Total videos: {len(videos)}")
    print("==================================================")

    rows = []

    for i, video in enumerate(videos, start=1):
        rows.append(extract_video(ffmpeg, video, i, len(videos)))

    fieldnames = [
        "Index",
        "FileName",
        "FullPath",
        "OutputFolder",
        "FPS_SAMPLE",
        "BeforeFrames",
        "AfterFrames",
        "NewFrames",
        "Status",
        "Error",
    ]

    with REPORT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    ok = sum(1 for r in rows if r["Status"] == "OK")
    partial = sum(1 for r in rows if r["Status"] == "PARTIAL")
    skipped = sum(1 for r in rows if r["Status"] == "SKIP_EXISTING")
    bad = sum(1 for r in rows if r["Status"] in ("BAD", "ERROR"))
    total_frames = sum(int(r["AfterFrames"]) for r in rows)

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("SAFE DRIVE PHONE/RADIO - CLEAN FRAME EXTRACTION\n")
        f.write("==================================================\n")
        f.write(f"Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video source: {VIDEO_SRC}\n")
        f.write(f"Output      : {OUT_DIR}\n")
        f.write(f"FPS sample  : {FPS_SAMPLE}\n")
        f.write(f"Total videos: {total}\n")
        f.write(f"OK          : {ok}\n")
        f.write(f"Partial     : {partial}\n")
        f.write(f"Skipped     : {skipped}\n")
        f.write(f"Bad/Error   : {bad}\n")
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"CSV report  : {REPORT_CSV}\n")
        f.write("\nBad/Error files:\n")
        for r in rows:
            if r["Status"] in ("BAD", "ERROR"):
                f.write(f"- {r['FileName']} | {r['Error']}\n")
        f.write("==================================================\n")

    print("")
    print("==================================================")
    print("STEP 3/20 DONE")
    print("==================================================")
    print(f"Total videos : {total}")
    print(f"OK           : {ok}")
    print(f"Partial      : {partial}")
    print(f"Skipped      : {skipped}")
    print(f"Bad/Error    : {bad}")
    print(f"Total frames : {total_frames}")
    print(f"Output       : {OUT_DIR}")
    print(f"CSV report   : {REPORT_CSV}")
    print(f"Summary      : {SUMMARY_TXT}")
    print("==================================================")


if __name__ == "__main__":
    main()
