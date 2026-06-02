from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO


ROOT = Path(r"D:\project_detectfaceandphone\SafeDrive-Edge-AI")

DATA_YAML = ROOT / "datasets" / "phone_radio_cabin" / "05_yolo_dataset_unique80" / "dataset.yaml"
BASE_MODEL = ROOT / "yolo11s.pt"

PROJECT_DIR = ROOT / "datasets" / "phone_radio_cabin" / "07_train_outputs"
RUN_NAME = "phone_radio_yolo11s_unique80_v1"


def main():
    print("==================================================")
    print("STEP 13/20 - TRAIN PHONE / WALKIE_TALKIE YOLO")
    print("==================================================")
    print(f"DATA_YAML  : {DATA_YAML}")
    print(f"BASE_MODEL : {BASE_MODEL}")
    print(f"PROJECT    : {PROJECT_DIR}")
    print(f"RUN_NAME   : {RUN_NAME}")
    print("==================================================")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {DATA_YAML}")

    if not BASE_MODEL.exists():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")

    cuda = torch.cuda.is_available()
    device = 0 if cuda else "cpu"

    if cuda:
        imgsz = 960
        batch = 8
        workers = 2
        epochs = 100
        print(f"[OK] CUDA detected: {torch.cuda.get_device_name(0)}")
    else:
        imgsz = 640
        batch = 2
        workers = 0
        epochs = 80
        print("[WARN] CUDA not detected. Training on CPU.")

    print("")
    print("TRAIN CONFIG")
    print("--------------------------------------------------")
    print(f"device  : {device}")
    print(f"imgsz   : {imgsz}")
    print(f"batch   : {batch}")
    print(f"epochs  : {epochs}")
    print(f"workers : {workers}")
    print("--------------------------------------------------")

    model = YOLO(str(BASE_MODEL))

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        patience=30,
        cache=False,
        plots=True,
        val=True,
        verbose=True,
    )

    run_dir = PROJECT_DIR / RUN_NAME
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    print("")
    print("==================================================")
    print("STEP 13/20 DONE")
    print("==================================================")
    print(f"Run dir : {run_dir}")
    print(f"best.pt : {best_pt}")
    print(f"last.pt : {last_pt}")
    print(f"Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==================================================")


if __name__ == "__main__":
    main()
