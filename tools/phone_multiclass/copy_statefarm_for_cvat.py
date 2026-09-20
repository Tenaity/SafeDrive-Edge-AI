import argparse
import os
import shutil
from pathlib import Path


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="images_raw")
    parser.add_argument("--dst", required=True, help="images_cvat_statefarm")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    ensure_dir(str(dst))

    copied = 0
    for p in src.iterdir():
        if p.is_file() and p.name.lower().startswith("statefarm_"):
            shutil.copy2(p, dst / p.name)
            copied += 1

    print(f"Copied {copied} State Farm images to CVAT folder.")


if __name__ == "__main__":
    main()