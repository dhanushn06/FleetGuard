
import cv2
import numpy as np

from camera.capture import CameraCapture
from camera.face_tracking import FaceTracker
from camera.roi_extraction import ROIExtractor, REGION_NAMES


def test_missing_landmarks():
    print("\nTesting missing landmarks...", flush=True)

    extractor = ROIExtractor()

    result = extractor.extract(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        landmarks=None,
        timestamp=1.0,
        wall_time=None
    )

    assert result["track_ok"] is False

    for name in REGION_NAMES:
        assert result["rgb"][name] is None
        assert result["brightness"][name] is None
        assert result["pixel_count"][name] == 0

    print("PASS: Missing landmarks handled correctly.", flush=True)


def test_live_camera():
    print("\nStarting live camera test...", flush=True)

    camera = CameraCapture()
    tracker = FaceTracker()
    extractor = ROIExtractor()

    frame_count = 0

    try:
        print("Camera is running.", flush=True)
        print("Look at the camera. Press Q to quit.", flush=True)

        while True:
            camera_data = camera.get_frame()

            if camera_data is None:
                continue

            frame = camera_data["frame"]
            timestamp = camera_data["timestamp"]
            wall_time = camera_data["wall_time"]

            tracking = tracker.detect(frame, timestamp)

            result = extractor.extract(
                frame=frame,
                landmarks=tracking["landmarks"],
                timestamp=timestamp,
                wall_time=wall_time
            )

            frame_count += 1

            if frame_count % 30 == 0:
                print("\n" + "=" * 50, flush=True)
                print("FACE TRACKING:", result["track_ok"])
                print("MOTION:", result["motion"])

                print("\nREGIONAL RGB VALUES")

                for name in REGION_NAMES:
                    print(
                        name,
                        "=>",
                        result["rgb"][name],
                        "| Pixels:",
                        result["pixel_count"][name]
                    )

                print("\nREGIONAL BRIGHTNESS")

                for name, value in result["brightness"].items():
                    print(name, "=>", value)

                print("\nALERTNESS INDICATORS")

                for name, value in result["alertness"].items():
                    print(name, "=>", value)

                print("=" * 50, flush=True)

            display = frame.copy()

            if result["track_ok"]:
                status = "FACE TRACKED"
                color = (0, 255, 0)
            else:
                status = "NO FACE"
                color = (0, 0, 255)

            cv2.putText(
                display,
                status,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.imshow("PulseCam - ROI Testing", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        tracker.close()
        camera.release()


if __name__ == "__main__":
    print("\nPULSECAM MEMBER 1 TEST", flush=True)
    test_missing_landmarks()
    test_live_camera()
