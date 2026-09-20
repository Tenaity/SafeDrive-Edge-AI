import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-a", required=True, help="miss_phone folder")
    parser.add_argument("--src-b", required=True, help="wrong_class folder")
    parser.add_argument("--dst", required=True, help="images_cvat_round2")
    args = parser.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src_root in [Path(args.src_a), Path(args.src_b)]:
        for p in src_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                shutil.copy2(p, dst / p.name)
                copied += 1

    print(f"Copied {copied} images for round2 CVAT.")
    

if __name__ == "__main__":
    main()