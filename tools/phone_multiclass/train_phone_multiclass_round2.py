from pathlib import Path
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_yaml = project_root / "datasets" / "phone_multiclass_round2" / "dataset.yaml"
    model_path = project_root / "runs" / "phone_multiclass_yolo11s_cpu_test" / "weights" / "best.pt"

    print(f"Data YAML: {data_yaml}")
    print(f"Base model: {model_path}")
    print("Train mode: CPU")

    model = YOLO(str(model_path))

    model.train(
        data=str(data_yaml),
        epochs=5,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project=str(project_root / "runs"),
        name="phone_multiclass_yolo11s_round2_cpu",
        pretrained=True,
        single_cls=False,
        patience=5,
        cache=False,
    )


if __name__ == "__main__":
    main()