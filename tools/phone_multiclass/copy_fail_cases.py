import argparse
import shutil
from pathlib import Path


def normalize_line_to_filename(text: str) -> str:
    name = text.strip().strip('"').strip("'")
    if not name:
        return ""
    return Path(name).name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-file", required=True, help="File txt chứa danh sách tên ảnh hoặc full path")
    parser.add_argument("--src-dir", required=True, help="Thư mục ảnh nguồn")
    parser.add_argument("--dst-dir", required=True, help="Thư mục đích")
    args = parser.parse_args()

    list_file = Path(args.list_file)
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not list_file.exists():
        raise RuntimeError(f"Không tìm thấy file danh sách: {list_file}")

    copied = 0
    with open(list_file, "r", encoding="utf-8") as f:
        for raw in f:
            name = normalize_line_to_filename(raw)
            if not name:
                continue

            src_path = src_dir / name
            if not src_path.exists():
                print(f"[WARN] Không thấy ảnh: {src_path}")
                continue

            shutil.copy2(src_path, dst_dir / name)
            copied += 1

    print(f"Copied {copied} files to {dst_dir}")


if __name__ == "__main__":
    main()