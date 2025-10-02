import cv2
import logging
from detector import YoloDetector
from door_state import DoorStateTracker
from pathlib import Path
import sys

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="w")
    ],
    force=True
)
logger = logging.getLogger("vending-pipeline")

VIDEO_PATH = "994a237a-3d0c-4792-8407-096ed45a0fd3.mp4"
MODEL_PATH = "runs/detect/boost/weights/best.pt"

video_name = Path(VIDEO_PATH).stem  
out_dir = Path("recordings") / video_name
out_dir.mkdir(parents=True, exist_ok=True)

OUTPUT_TEMPLATE = str(out_dir / "door_open_clip_{:03d}.mp4")

def run_pipeline():
    detector = YoloDetector(MODEL_PATH)
    door_tracker = DoorStateTracker(threshold=80)  # adjust threshold after testing

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = int(1000 / fps)
    # Recorder state
    recording = False
    video_writer = None
    clip_index = 0

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break


        door_open = door_tracker.is_open(frame)
        detections = detector.detect(frame)
        annotated = detector.annotate(frame, detections, door_open=door_open)

        logger.info({
            "frame_id": frame_id,
            "door_open": door_open,
            "detections": [d.to_dict() for d in detections]
        })
        
        # cv2.imshow("Detections", annotated)

        # Event-driven recording
        if door_open and not recording:
            # Start a new recording
            clip_path = OUTPUT_TEMPLATE.format(clip_index)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
            recording = True
            logger.info(f"Started recording: {clip_path}")

        if recording:
            video_writer.write(annotated)

        if not door_open and recording:
            # Stop recording
            video_writer.release()
            recording = False
            clip_index += 1
            logger.info("Stopped recording")

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

        frame_id += 1

    cap.release()
    if recording and video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()
