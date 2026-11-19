
import cv2
import time
from datetime import datetime, timezone


class CameraCapture:

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Unable to open webcam")

        # Request a standard webcam resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def get_frame(self):

        success, frame = self.cap.read()

        if not success:
            return None

        # Flip for mirror-like webcam preview
        frame = cv2.flip(frame, 1)

        timestamp = time.monotonic()
        wall_time = datetime.now(timezone.utc).isoformat()

        return {
            "frame": frame,
            "timestamp": timestamp,
            "wall_time": wall_time
        }

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
