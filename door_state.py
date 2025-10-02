import cv2

class DoorStateTracker:
    def __init__(self, threshold=100):
        self.threshold = threshold

    def is_open(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = gray.mean()
        return bool(mean_intensity > self.threshold)
