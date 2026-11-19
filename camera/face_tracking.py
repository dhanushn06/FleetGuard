
import cv2
import mediapipe as mp


class FaceTracker:

    def __init__(self, model_path="models/face_landmarker.task"):

        self.mp = mp

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = FaceLandmarker.create_from_options(
            options
        )

        self.last_timestamp_ms = -1

    def detect(self, frame, timestamp):

        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(timestamp * 1000)

        # MediaPipe requires increasing timestamps
        timestamp_ms = max(
            timestamp_ms,
            self.last_timestamp_ms + 1
        )

        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if not result.face_landmarks:
            return {
                "tracking_status": False,
                "landmarks": None
            }

        face = result.face_landmarks[0]

        landmarks = []

        for point in face:

            x = int(point.x * width)
            y = int(point.y * height)

            landmarks.append((x, y))

        return {
            "tracking_status": True,
            "landmarks": landmarks
        }

    def close(self):
        self.landmarker.close()
