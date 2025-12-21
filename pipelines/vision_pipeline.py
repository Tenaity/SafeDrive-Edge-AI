from ultralytics import YOLO

class VisionPipeline:
    """
    VisionPipeline:
    - Detect objects using YOLO
    - Filter by confidence
    - Only expose safety-relevant objects
    """

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

        # Confidence thresholds (anti-spam)
        self.CONF_PERSON = 0.5
        self.CONF_PHONE = 0.6

    def run(self, frame):
        results = self.model(frame, verbose=False)[0]

        phone_detected = False
        filtered_boxes = []

        for box in results.boxes:
            cls = int(box.cls[0])
            label = self.model.names[cls]
            conf = float(box.conf[0])

            if label == "person" and conf >= self.CONF_PERSON:
                filtered_boxes.append(box)

            elif label == "cell phone" and conf >= self.CONF_PHONE:
                phone_detected = True
                filtered_boxes.append(box)

        # Override boxes with filtered ones (avoid clutter)
        results.boxes = filtered_boxes

        return {
            "phone": phone_detected,
            "yolo_result": results
        }
