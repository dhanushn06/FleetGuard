
import cv2

from camera.capture import CameraCapture
from camera.face_tracking import FaceTracker
from camera.roi_extraction import ROIExtractor


def draw_legend(image, colors):

    height, width = image.shape[:2]

    # Small legend positioned at bottom-left
    entries = [
        ("FOREHEAD", colors["forehead"]),
        ("LEFT CHEEK", colors["left_cheek"]),
        ("RIGHT CHEEK", colors["right_cheek"])
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness = 1

    line_height = 19
    panel_height = 72
    panel_width = 125

    x = 10
    y = height - panel_height - 10

    # Compact dark translucent panel
    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + panel_width, y + panel_height),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay, 0.55,
        image, 0.45,
        0,
        image
    )

    for i, (name, color) in enumerate(entries):

        text_y = y + 19 + i * line_height

        # Small coloured indicator
        cv2.line(
            image,
            (x + 8, text_y - 4),
            (x + 20, text_y - 4),
            color,
            2
        )

        # Small region name
        cv2.putText(
            image,
            name,
            (x + 26, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )


def main():

    camera = CameraCapture()
    tracker = FaceTracker()
    extractor = ROIExtractor()

    frame_count = 0

    colors = {
        "forehead": (0, 255, 255),
        "left_cheek": (0, 255, 0),
        "right_cheek": (255, 0, 255)
    }

    print("PulseCam pipeline started.")
    print("Press Q to stop.")

    try:

        while True:

            packet = camera.get_frame()

            if packet is None:
                print("Unable to read camera frame.")
                break

            frame = packet["frame"]
            timestamp = packet["timestamp"]
            wall_time = packet["wall_time"]

            display = frame.copy()

            # 1. Face tracking
            tracking = tracker.detect(
                frame,
                timestamp
            )

            if tracking["tracking_status"]:

                landmarks = tracking["landmarks"]

                # 2. Extract measurements
                data = extractor.extract(
                    frame,
                    landmarks,
                    timestamp,
                    wall_time
                )

                # 3. Generate ROI masks
                masks = extractor.get_roi_masks(
                    frame,
                    landmarks
                )

                # 4. Draw thin ROI outlines
                for name, mask in masks.items():

                    contours, _ = cv2.findContours(
                        mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    cv2.drawContours(
                        display,
                        contours,
                        -1,
                        colors[name],
                        1,
                        cv2.LINE_AA
                    )

                # 5. Small tracking status
                cv2.putText(
                    display,
                    "FACE TRACKED",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

                # 6. Compact colour legend
                draw_legend(display, colors)

                # 7. Print measurements periodically
                if frame_count % 30 == 0:

                    print("\n--- PulseCam Measurements ---")
                    print("Timestamp:", timestamp)
                    print("Brightness:", data["brightness"])
                    print(
                        "Motion:",
                        data["motion_displacement"]
                    )

                    for region, values in data["regions"].items():
                        print(region, ":", values)

            else:

                cv2.putText(
                    display,
                    "NO FACE DETECTED",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )

            # 8. Display live camera
            cv2.imshow(
                "PulseCam - Live Pipeline",
                display
            )

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.release()
        tracker.close()
        cv2.destroyAllWindows()

        print("PulseCam stopped.")


if __name__ == "__main__":
    main()
