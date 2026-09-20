import csv
import zipfile
from pathlib import Path
from datetime import datetime


ROOT = Path(r"D:\project_detectfaceandphone\SafeDrive-Edge-AI")
DATASET = ROOT / "datasets" / "phone_radio_cabin"

FRAMES_ROOT = DATASET / "01_frames_unique_80"
CVAT_DIR = DATASET / "03_cvat_upload_unique80"
INV_DIR = DATASET / "00_inventory"

CVAT_DIR.mkdir(parents=True, exist_ok=True)
INV_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = INV_DIR / "cvat_upload_unique80_manifest.csv"
SUMMARY_TXT = INV_DIR / "cvat_upload_unique80_summary.txt"

# 2000 ảnh / ZIP: tương đối an toàn cho CVAT
IMAGES_PER_ZIP = 2000

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    print("==================================================")
    print("STEP 8/20 - MAKE CVAT ZIP PACKAGES FROM UNIQUE_80")
    print("==================================================")
    print(f"Frames root    : {FRAMES_ROOT}")
    print(f"CVAT upload dir: {CVAT_DIR}")
    print(f"Images per zip : {IMAGES_PER_ZIP}")
    print("==================================================")

    if not FRAMES_ROOT.exists():
        print(f"[ERROR] Khong tim thay folder anh da loc: {FRAMES_ROOT}")
        raise SystemExit(1)

    images = sorted(
        [p for p in FRAMES_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda x: str(x).lower()
    )

    total_images = len(images)

    if total_images == 0:
        print("[ERROR] Khong tim thay anh .jpg/.png nao trong unique_80.")
        raise SystemExit(1)

    print(f"[OK] Total images found: {total_images}")

    # Xoa zip cu cua bo unique80 de tranh nham lan
    old_zips = list(CVAT_DIR.glob("cvat_phone_radio_unique80_*.zip"))
    for z in old_zips:
        z.unlink()

    rows = []
    package_index = 0

    for start in range(0, total_images, IMAGES_PER_ZIP):
        package_index += 1
        chunk = images[start:start + IMAGES_PER_ZIP]

        zip_name = f"cvat_phone_radio_unique80_{package_index:03d}.zip"
        zip_path = CVAT_DIR / zip_name

        print(f"[ZIP {package_index:03d}] Creating {zip_name} | images={len(chunk)}")

        # JPG da nen san, ZIP_STORED nhanh hon va khong ton CPU
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for img in chunk:
                # Giu ten file ngan de CVAT de xu ly
                zf.write(img, arcname=img.name)

        size_mb = round(zip_path.stat().st_size / (1024 * 1024), 2)

        rows.append({
            "PackageIndex": package_index,
            "ZipName": zip_name,
            "ZipPath": str(zip_path),
            "ImageCount": len(chunk),
            "SizeMB": size_mb,
            "FirstImage": chunk[0].name if chunk else "",
            "LastImage": chunk[-1].name if chunk else "",
        })

        print(f"[ZIP {package_index:03d}] OK | size={size_mb} MB")

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "PackageIndex",
            "ZipName",
            "ZipPath",
            "ImageCount",
            "SizeMB",
            "FirstImage",
            "LastImage",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_zip_size = round(sum(float(r["SizeMB"]) for r in rows), 2)

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("SAFE DRIVE PHONE/RADIO - CVAT UNIQUE80 PACKAGES\n")
        f.write("==================================================\n")
        f.write(f"Time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Frames root    : {FRAMES_ROOT}\n")
        f.write(f"CVAT upload dir: {CVAT_DIR}\n")
        f.write(f"Total images   : {total_images}\n")
        f.write(f"Images per zip : {IMAGES_PER_ZIP}\n")
        f.write(f"Total packages : {len(rows)}\n")
        f.write(f"Total zip size : {total_zip_size} MB\n")
        f.write(f"Manifest CSV   : {MANIFEST_CSV}\n")
        f.write("\nPackages:\n")
        for r in rows:
            f.write(
                f"- {r['ZipName']} | images={r['ImageCount']} | "
                f"size={r['SizeMB']} MB\n"
            )
        f.write("==================================================\n")

    print("")
    print("==================================================")
    print("STEP 8/20 DONE")
    print("==================================================")
    print(f"Total images   : {total_images}")
    print(f"Total packages : {len(rows)}")
    print(f"Total zip size : {total_zip_size} MB")
    print(f"Output         : {CVAT_DIR}")
    print(f"Manifest CSV   : {MANIFEST_CSV}")
    print(f"Summary        : {SUMMARY_TXT}")
    print("==================================================")


if __name__ == "__main__":
    main()
