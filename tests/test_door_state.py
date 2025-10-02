# tests/test_door_state.py
import sys
import pathlib

# Add project root (where detector.py lives) to PYTHONPATH
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from door_state import DoorStateTracker


def test_door_open_detected():
    tracker = DoorStateTracker(threshold=100)
    bright_frame = 255 * np.ones((100, 100, 3), dtype=np.uint8)
    assert tracker.is_open(bright_frame) == True   # not "is"

def test_door_closed_detected():
    tracker = DoorStateTracker(threshold=100)
    dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert tracker.is_open(dark_frame) == False
