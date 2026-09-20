import csv
import math
import cv2
from pathlib import Path
from datetime import datetime


ROOT = Path(r"D:\project_detectfaceandphone\SafeDrive-Edge-AI")
DATASET = ROOT / "datasets" / "phone_radio_cabin"

FRAMES_ROOT = DATASET / "01_frames_raw"
OUT_DIR = DATASET / "02_contact_sheets"
INV_DIR = DATASET / "00_inventory"

OUT_DIR.mkdir(parents=True, exist_ok=True)
INV_DIR.mkdir(parents=True, exist_ok=True)

REPORT_CSV = INV_DIR / "contact_sheets_report.csv"
SUMMARY_TXT = INV_DIR / "contact_sheets_summary.txt"

# Moi clip lay toi da 25 frame dai dien de ghep vao 1 sheet
SAMPLES_PER_CLIP = 25

# Kich thuoc moi anh con trong contact sheet
THUMB_W = 320
THUMB_H = 180

# Luoi 5x5 = 25 frame
GRID_COLS = 5
GRID_ROWS = 5

HEADER_H = 60
FOOTER_H = 30

JPEG_QUALITY = 92


def list_images(folder: Path):
    return sorted(folder.glob("*.jpg"), key=lambda x: x.name.lower())


def sample_images(images, max_count):
    if len(images) <= max_count:
        return images

    if max_count <= 1:
        return [images[0]]

    step = (len(images) - 1) / float(max_count - 1)
    picked = []

    for i in range(max_count):
        idx = int(round(i * step))
        idx = max(0, min(idx, len(images) - 1))
        picked.append(images[idx])

    # remove duplicate while preserving order
    seen = set()
    unique = []
    for p in picked:
        if p not in seen:
            unique.append(p)
            seen.add(p)

    return unique


def make_blank(width, height):
    return 255 * (cv2.UMat(height, width, cv2.CV_8UC3).get())


def fit_image(img, w, h):
    ih, iw = img.shape[:2]
    if iw <= 0 or ih <= 0:
        return make_blank(w, h)

    scale = min(w / iw, h / ih)
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = make_blank(w, h)
    x = (w - nw) // 2
    y = (h - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized

    return canvas


def put_text(img, text, x, y, scale=0.6, thickness=1):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def create_sheet(clip_dir: Path, index: int, total: int):
    images = list_images(clip_dir)

    row = {
        "Index": index,
        "ClipName": clip_dir.name,
        "FrameFolder": str(clip_dir),
        "TotalFrames": len(images),
        "SampledFrames": 0,
        "SheetPath": "",
        "Status": "BAD",
        "Error": "",
    }

    if not images:
        row["Error"] = "no_images"
        print(f"[{index:03d}/{total}] BAD | {clip_dir.name} | no images")
        return row

    sampled = sample_images(images, SAMPLES_PER_CLIP)
    row["SampledFrames"] = len(sampled)

    sheet_w = GRID_COLS * THUMB_W
    sheet_h = HEADER_H + GRID_ROWS * THUMB_H + FOOTER_H

    sheet = make_blank(sheet_w, sheet_h)

    put_text(
        sheet,
        f"{clip_dir.name} | total_frames={len(images)} | sampled={len(sampled)}",
        20,
        38,
        scale=0.9,
        thickness=2,
    )

    ok_count = 0

    for i, img_path in enumerate(sampled):
        r = i // GRID_COLS
        c = i % GRID_COLS

        if r >= GRID_ROWS:
            break

        x = c * THUMB_W
        y = HEADER_H + r * THUMB_H

        img = cv2.imread(str(img_path))

        if img is None:
            thumb = make_blank(THUMB_W, THUMB_H)
            put_text(thumb, "BAD IMAGE", 20, 90, scale=0.7, thickness=2)
        else:
            thumb = fit_image(img, THUMB_W, THUMB_H)
            ok_count += 1

        # border
        cv2.rectangle(thumb, (0, 0), (THUMB_W - 1, THUMB_H - 1), (0, 0, 0), 1)

        label = img_path.name[-18:]
        put_text(thumb, label, 8, THUMB_H - 10, scale=0.45, thickness=1)

        sheet[y:y+THUMB_H, x:x+THUMB_W] = thumb

    put_text(
        sheet,
        "Review note: phone / walkie_talkie / normal / bad_angle",
        20,
        sheet_h - 10,
        scale=0.6,
        thickness=1,
    )

    out_path = OUT_DIR / f"{clip_dir.name}_sheet.jpg"

    ok = cv2.imwrite(
        str(out_path),
        sheet,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )

    if not ok:
        row["Error"] = "imwrite_failed"
        print(f"[{index:03d}/{total}] BAD | {clip_dir.name} | imwrite failed")
        return row

    row["SheetPath"] = str(out_path)
    row["Status"] = "OK" if ok_count > 0 else "BAD"
    row["Error"] = "" if ok_count > 0 else "all_sampled_bad"

    print(
        f"[{index:03d}/{total}] {row['Status']} | "
        f"{clip_dir.name} | frames={len(images)} | sheet={out_path.name}"
    )

    return row


def main():
    clip_dirs = sorted(
        [p for p in FRAMES_ROOT.iterdir() if p.is_dir()],
        key=lambda x: x.name.lower()
    )

    print("==================================================")
    print("STEP 4/20 - MAKE CONTACT SHEETS")
    print("==================================================")
    print(f"Frames root : {FRAMES_ROOT}")
    print(f"Output      : {OUT_DIR}")
    print(f"Clip folders: {len(clip_dirs)}")
    print("==================================================")

    rows = []

    for i, clip_dir in enumerate(clip_dirs, start=1):
        rows.append(create_sheet(clip_dir, i, len(clip_dirs)))

    fieldnames = [
        "Index",
        "ClipName",
        "FrameFolder",
        "TotalFrames",
        "SampledFrames",
        "SheetPath",
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

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("SAFE DRIVE PHONE/RADIO - CONTACT SHEETS\n")
        f.write("==================================================\n")
        f.write(f"Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Frames root : {FRAMES_ROOT}\n")
        f.write(f"Output      : {OUT_DIR}\n")
        f.write(f"Total clips : {total}\n")
        f.write(f"OK sheets   : {ok}\n")
        f.write(f"Bad sheets  : {bad}\n")
        f.write(f"CSV report  : {REPORT_CSV}\n")
        f.write("\nBad clips:\n")
        for r in rows:
            if r["Status"] != "OK":
                f.write(f"- {r['ClipName']} | {r['Error']}\n")
        f.write("==================================================\n")

    print("")
    print("==================================================")
    print("STEP 4/20 DONE")
    print("==================================================")
    print(f"Total clips : {total}")
    print(f"OK sheets   : {ok}")
    print(f"Bad sheets  : {bad}")
    print(f"Output      : {OUT_DIR}")
    print(f"CSV report  : {REPORT_CSV}")
    print(f"Summary     : {SUMMARY_TXT}")
    print("==================================================")


if __name__ == "__main__":
    main()
