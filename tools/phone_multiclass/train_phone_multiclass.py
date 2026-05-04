from pathlib import Path
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_yaml = project_root / "datasets" / "phone_multiclass" / "dataset.yaml"
    model_path = project_root / "yolo11s.pt"

    print(f"Data YAML: {data_yaml}")
    print(f"Model:     {model_path}")
    print("Train mode: CPU")

    model = YOLO(str(model_path))

    model.train(
        data=str(data_yaml),
        epochs=3,                 # test ngắn trước để kiểm tra pipeline
        imgsz=640,                # giảm cho CPU đỡ nặng
        batch=4,                  # giảm batch để đỡ RAM/CPU
        device="cpu",             # SỬA CHÍNH Ở ĐÂY
        workers=0,                # Windows + CPU thường ổn hơn với 0
        project=str(project_root / "runs"),
        name="phone_multiclass_yolo11s_cpu_test",
        pretrained=True,
        single_cls=False,
        patience=5,
        cache=False,
    )


if __name__ == "__main__":
    main()