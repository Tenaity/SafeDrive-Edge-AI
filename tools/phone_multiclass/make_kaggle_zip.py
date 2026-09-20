from pathlib import Path
import zipfile


def main():
    project_root = Path(__file__).resolve().parents[2]
    src_dir = project_root / "datasets" / "phone_multiclass_kaggle_v1"
    zip_path = project_root / "datasets" / "phone_multiclass_kaggle_v1.zip"

    if not src_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {src_dir}")

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if not path.is_file():
                continue
            # dùng đường dẫn tương đối và ép sang dấu /
            arcname = path.relative_to(src_dir).as_posix()
            zf.write(path, arcname)

    print(f"[DONE] Created zip: {zip_path}")


if __name__ == "__main__":
    main()
    