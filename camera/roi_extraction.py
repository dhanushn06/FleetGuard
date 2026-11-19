
import cv2
import numpy as np


# MediaPipe face oval landmarks
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]


# Upper forehead contour landmarks
FOREHEAD_TOP = [
    10, 338, 297, 332, 284, 251, 389,
    356, 127, 162, 21, 54, 103, 67, 109
]


# Eyebrow boundary landmarks
EYEBROW_POINTS = [
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300
]


# Image-left and image-right cheek landmarks
CHEEK_LANDMARKS = {

    "left_cheek": [
        50, 101, 118, 117, 123, 147, 187,
        205, 36, 142, 203, 206, 216, 212
    ],

    "right_cheek": [
        280, 330, 347, 346, 352, 376, 411,
        425, 266, 371, 423, 426, 436, 432
    ]
}


class ROIExtractor:

    def __init__(self):

        self.previous_landmarks = None

    def _create_mask(self, frame_shape, polygon):

        height, width = frame_shape[:2]

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        polygon = np.asarray(
            polygon, dtype=np.int32
        )

        polygon[:, 0] = np.clip(
            polygon[:, 0], 0, width - 1
        )

        polygon[:, 1] = np.clip(
            polygon[:, 1], 0, height - 1
        )

        cv2.fillPoly(mask, [polygon], 255)

        return mask

    def get_roi_masks(self, frame, landmarks):

        height, width = frame.shape[:2]

        if landmarks is None or len(landmarks) < 478:
            return {}

        points = np.asarray(
            landmarks, dtype=np.float32
        )

        # Calculate the detected face dimensions
        face_width = (
            np.max(points[:, 0]) -
            np.min(points[:, 0])
        )

        face_height = (
            np.max(points[:, 1]) -
            np.min(points[:, 1])
        )

        if face_width < 10 or face_height < 10:
            return {}

        masks = {}

        # ------------------------------------
        # 1. FACE BOUNDARY MASK
        # ------------------------------------

        face_polygon = points[FACE_OVAL]

        face_mask = self._create_mask(
            frame.shape,
            face_polygon
        )

        # ------------------------------------
        # 2. ADAPTIVE FOREHEAD MASK
        # ------------------------------------

        top_points = points[FOREHEAD_TOP].copy()

        # Sort upper boundary from image-left to image-right
        top_points = top_points[
            np.argsort(top_points[:, 0])
        ]

        brow_points = points[EYEBROW_POINTS].copy()

        # Sort eyebrow boundary horizontally
        brow_points = brow_points[
            np.argsort(brow_points[:, 0])
        ]

        # Move the lower forehead boundary slightly
        # above the eyebrow landmarks.
        # The adjustment scales with face height.
        upward_offset = max(
            1, int(face_height * 0.012)
        )

        brow_points[:, 1] -= upward_offset

        # Construct forehead polygon:
        # upper forehead boundary + eyebrow boundary
        forehead_polygon = np.vstack([
            top_points,
            brow_points[::-1]
        ])

        forehead_mask = self._create_mask(
            frame.shape,
            forehead_polygon
        )

        # Keep forehead pixels inside detected face
        forehead_mask = cv2.bitwise_and(
            forehead_mask,
            face_mask
        )

        masks["forehead"] = forehead_mask

        # ------------------------------------
        # 3. ADAPTIVE CHEEK MASKS
        # ------------------------------------

        for region_name, indices in CHEEK_LANDMARKS.items():

            cheek_points = points[indices].copy()

            # Create initial cheek polygon
            hull = cv2.convexHull(
                cheek_points.astype(np.int32)
            )

            cheek_mask = np.zeros(
                (height, width),
                dtype=np.uint8
            )

            cv2.fillConvexPoly(
                cheek_mask,
                hull,
                255
            )

            # Expand cheek area according to face size
            # More horizontal than vertical expansion
            kernel_x = max(
                3, int(face_width * 0.045)
            )

            kernel_y = max(
                3, int(face_height * 0.018)
            )

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_x, kernel_y)
            )

            cheek_mask = cv2.dilate(
                cheek_mask,
                kernel,
                iterations=1
            )

            # Restrict expansion to the face boundary
            cheek_mask = cv2.bitwise_and(
                cheek_mask,
                face_mask
            )

            masks[region_name] = cheek_mask

        return masks

    def extract(self, frame, landmarks, timestamp, wall_time=None):

        # Handle missing face
        if landmarks is None or len(landmarks) < 478:

            self.previous_landmarks = None

            return {
                "timestamp": timestamp,
                "wall_time": wall_time,
                "tracking_status": False,
                "motion_displacement": None,
                "brightness": None,
                "regions": None
            }

        points = np.asarray(
            landmarks,
            dtype=np.float32
        )

        # ------------------------------------
        # 4. FACIAL MOVEMENT ESTIMATION
        # ------------------------------------

        if self.previous_landmarks is None:

            motion = 0.0

        else:

            previous = np.asarray(
                self.previous_landmarks,
                dtype=np.float32
            )

            motion = float(
                np.mean(
                    np.linalg.norm(
                        points - previous,
                        axis=1
                    )
                )
            )

        self.previous_landmarks = points.copy()

        # ------------------------------------
        # 5. EXTRACT RGB FROM EACH REGION
        # ------------------------------------

        masks = self.get_roi_masks(
            frame,
            landmarks
        )

        regions = {}
        brightness_values = []

        for region_name, mask in masks.items():

            pixels = frame[mask == 255]

            if len(pixels) == 0:

                regions[region_name] = None
                continue

            # Convert OpenCV BGR to RGB
            rgb_pixels = cv2.cvtColor(
                pixels.reshape(-1, 1, 3),
                cv2.COLOR_BGR2RGB
            ).reshape(-1, 3)

            red, green, blue = np.mean(
                rgb_pixels,
                axis=0
            )

            # Calculate brightness
            gray_pixels = cv2.cvtColor(
                pixels.reshape(-1, 1, 3),
                cv2.COLOR_BGR2GRAY
            )

            region_brightness = float(
                np.mean(gray_pixels)
            )

            brightness_values.append(
                region_brightness
            )

            regions[region_name] = {

                "rgb": {
                    "r": float(red),
                    "g": float(green),
                    "b": float(blue)
                },

                "brightness": region_brightness,

                "pixel_count": int(len(pixels))
            }

        overall_brightness = (
            float(np.mean(brightness_values))
            if brightness_values else None
        )

        return {

            "timestamp": timestamp,

            "wall_time": wall_time,

            "tracking_status": True,

            "motion_displacement": motion,

            "brightness": overall_brightness,

            "regions": regions
        }
