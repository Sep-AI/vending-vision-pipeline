# tests/test_detector.py
import numpy as np
import cv2
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from detector import Detection, YoloDetector

def test_detection_to_dict():
    d = Detection(label="hand", confidence=0.95, bbox=(10, 20, 30, 40))
    d_dict = d.model_dump()
    assert d_dict["label"] == "hand"
    assert d_dict["confidence"] == 0.95
    assert d_dict["bbox"] == (10, 20, 30, 40)

def test_annotate_with_fake_detections(tmp_path):
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    fake_detections = [
        Detection(label="hand", confidence=0.99, bbox=(50, 50, 100, 100))
    ]

    det = YoloDetector(load_model=False)
    annotated = det.annotate(frame, fake_detections, door_open=True)

    assert annotated.shape == frame.shape
    out_file = tmp_path / "annotated.jpg"
    cv2.imwrite(str(out_file), annotated)
    assert out_file.exists()


def test_door_overlay_closed():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    fake_detections = []
    det = YoloDetector(load_model=False)

    annotated = det.annotate(frame, fake_detections, door_open=False)

    assert not np.all(annotated == frame)
