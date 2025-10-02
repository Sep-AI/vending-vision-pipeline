from ultralytics import YOLO
import cv2
from typing import Tuple
from pydantic import BaseModel


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]

    def to_dict(self):
        return self.model_dump()


class YoloDetector:
    def __init__(self, model_path: str = None, conf_threshold: float = 0.3, load_model: bool = True):
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path) if (model_path and load_model) else None

        # define colors per class
        self.class_colors = {
            "hand": (0, 0, 255),     # red
            "box": (255, 0, 0),      # blue
            "product": (0, 255, 255) # yellow
        }
        self.default_color = (0, 255, 0)  # green fallback

    def detect(self, frame):
        if not self.model:
            raise RuntimeError("Model not loaded. Use load_model=True or mock detect().")

        results = self.model.predict(frame, verbose=False, conf=self.conf_threshold)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            detections.append(
                Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2))
            )
        return detections

    def annotate(self, frame, detections, door_open: bool):
        annotated = frame.copy()

        # draw YOLO detections
        for det in detections:
            color = self.class_colors.get(det.label.lower(), self.default_color)
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_text = f"{det.label} {det.confidence:.2f}"
            cv2.putText(annotated, label_text, (x1, max(y1 - 5, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # draw door status overlay
        text = "DOOR: OPEN" if door_open else "DOOR: CLOSED"
        color = (0, 0, 255) if door_open else (0, 200, 0)
        cv2.putText(annotated, text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

        return annotated