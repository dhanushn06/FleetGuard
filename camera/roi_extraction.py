
import cv2
import numpy as np


# -------------------------------------------------
# MEDIAPIPE FACE LANDMARK INDICES
# -------------------------------------------------

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

FOREHEAD_TOP = [
    10, 338, 297, 332, 284, 251, 389,
    356, 127, 162, 21, 54, 103, 67, 109
]

EYEBROW_POINTS = [
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300
]

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


# Eight required measurement regions.
REGION_NAMES = [
    "forehead_center",
    "forehead_left",
    "forehead_right",
    "left_cheek_upper",
    "left_cheek_lower",
    "right_cheek_upper",
    "right_cheek_lower",
    "nose_bridge"
]


# Eye landmark indices for EAR.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# Mouth landmark indices for MAR.
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308


class ROIExtractor:

    def __init__(
        self,
        min_region_pixels=20,
        min_brightness=20.0,
        ear_threshold=0.20
    ):

        self.previous_landmarks = None

        self.min_region_pixels = min_region_pixels
        self.min_brightness = min_brightness
        self.ear_threshold = ear_threshold

    # -------------------------------------------------
    # EMPTY MASKS
    # -------------------------------------------------

    def _empty_masks(self, frame):

        height, width = frame.shape[:2]

        return {
            name: np.zeros(
                (height, width),
                dtype=np.uint8
            )
            for name in REGION_NAMES
        }

    # -------------------------------------------------
    # CREATE POLYGON MASK
    # -------------------------------------------------

    def _create_mask(self, frame_shape, polygon):

        height, width = frame_shape[:2]

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        polygon = np.asarray(
            polygon,
            dtype=np.int32
        ).copy()

        if polygon.ndim != 2 or polygon.shape[0] < 3:
            return mask

        polygon[:, 0] = np.clip(
            polygon[:, 0],
            0,
            width - 1
        )

        polygon[:, 1] = np.clip(
            polygon[:, 1],
            0,
            height - 1
        )

        cv2.fillPoly(
            mask,
            [polygon],
            255
        )

        return mask

    # -------------------------------------------------
    # SPLIT MASK INTO REGIONS
    # -------------------------------------------------

    def _split_mask(
        self,
        mask,
        axis,
        first_limit,
        second_limit
    ):

        height, width = mask.shape

        first = np.zeros_like(mask)
        second = np.zeros_like(mask)
        third = np.zeros_like(mask)

        # Vertical split: left, center, right.
        if axis == 0:

            first[:, :first_limit] = (
                mask[:, :first_limit]
            )

            second[:, first_limit:second_limit] = (
                mask[:, first_limit:second_limit]
            )

            third[:, second_limit:] = (
                mask[:, second_limit:]
            )

            return first, second, third

        # Horizontal split: upper and lower.
        first[:first_limit, :] = (
            mask[:first_limit, :]
        )

        second[first_limit:, :] = (
            mask[first_limit:, :]
        )

        return first, second, None

    # -------------------------------------------------
    # RESTRICT MASK TO FACE
    # -------------------------------------------------

    def _intersect(self, mask, boundary_mask):

        return cv2.bitwise_and(
            mask,
            boundary_mask
        )

    # -------------------------------------------------
    # EXTRACT EIGHT FACIAL ROI MASKS
    # -------------------------------------------------

    def get_roi_masks(self, frame, landmarks):

        height, width = frame.shape[:2]

        if landmarks is None or len(landmarks) < 478:
            return self._empty_masks(frame)

        points = np.asarray(
            landmarks,
            dtype=np.float32
        )

        if not np.all(np.isfinite(points)):
            return self._empty_masks(frame)

        face_points = points[FACE_OVAL]

        face_width = (
            np.max(face_points[:, 0]) -
            np.min(face_points[:, 0])
        )

        face_height = (
            np.max(face_points[:, 1]) -
            np.min(face_points[:, 1])
        )

        if face_width < 10 or face_height < 10:
            return self._empty_masks(frame)

        masks = self._empty_masks(frame)

        # ---------------------------------------------
        # FACE BOUNDARY
        # ---------------------------------------------

        face_mask = self._create_mask(
            frame.shape,
            face_points
        )

        # ---------------------------------------------
        # FOREHEAD MASK
        # ---------------------------------------------

        top_points = points[FOREHEAD_TOP].copy()

        top_points = top_points[
            np.argsort(top_points[:, 0])
        ]

        brow_points = points[EYEBROW_POINTS].copy()

        brow_points = brow_points[
            np.argsort(brow_points[:, 0])
        ]

        upward_offset = max(
            1,
            int(face_height * 0.012)
        )

        brow_points[:, 1] -= upward_offset

        forehead_polygon = np.vstack([
            top_points,
            brow_points[::-1]
        ])

        forehead_mask = self._create_mask(
            frame.shape,
            forehead_polygon
        )

        forehead_mask = self._intersect(
            forehead_mask,
            face_mask
        )

        # Divide forehead into three sections.
        face_left = int(
            np.min(face_points[:, 0])
        )

        face_right = int(
            np.max(face_points[:, 0])
        )

        first_boundary = int(
            face_left + face_width / 3
        )

        second_boundary = int(
            face_left + 2 * face_width / 3
        )

        forehead_left, forehead_center, forehead_right = (
            self._split_mask(
                forehead_mask,
                axis=0,
                first_limit=first_boundary,
                second_limit=second_boundary
            )
        )

        masks["forehead_left"] = forehead_left
        masks["forehead_center"] = forehead_center
        masks["forehead_right"] = forehead_right

        # ---------------------------------------------
        # CHEEK MASKS
        # ---------------------------------------------

        for region_name, indices in CHEEK_LANDMARKS.items():

            cheek_points = points[indices].copy()

            hull = cv2.convexHull(
                cheek_points.astype(np.int32)
            )

            cheek_mask = self._create_mask(
                frame.shape,
                hull.reshape(-1, 2)
            )

            # Adaptive expansion.
            kernel_x = max(
                3,
                int(face_width * 0.025)
            )

            kernel_y = max(
                3,
                int(face_height * 0.012)
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

            cheek_mask = self._intersect(
                cheek_mask,
                face_mask
            )

            y_min = int(
                np.min(cheek_points[:, 1])
            )

            y_max = int(
                np.max(cheek_points[:, 1])
            )

            middle_y = (y_min + y_max) // 2

            upper, lower, _ = self._split_mask(
                cheek_mask,
                axis=1,
                first_limit=middle_y,
                second_limit=middle_y
            )

            if region_name == "left_cheek":

                masks["left_cheek_upper"] = upper
                masks["left_cheek_lower"] = lower

            else:

                masks["right_cheek_upper"] = upper
                masks["right_cheek_lower"] = lower

        # ---------------------------------------------
        # NOSE BRIDGE
        # ---------------------------------------------

        nose_indices = [
            168, 6, 197, 195, 5, 4, 1, 2
        ]

        nose_points = points[nose_indices].copy()

        nose_hull = cv2.convexHull(
            nose_points.astype(np.int32)
        )

        nose_mask = self._create_mask(
            frame.shape,
            nose_hull.reshape(-1, 2)
        )

        # Expand the narrow nose region so small landmark
        # movements do not collapse it into a thin polygon.
        kernel_x = max(5, int(face_width * 0.04))
        kernel_y = max(7, int(face_height * 0.025))

        if kernel_x % 2 == 0:
            kernel_x += 1

        if kernel_y % 2 == 0:
            kernel_y += 1

        nose_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_x, kernel_y)
        )

        nose_mask = cv2.dilate(
            nose_mask,
            nose_kernel,
            iterations=1
        )

        masks["nose_bridge"] = self._intersect(
            nose_mask,
            face_mask
        )

        return masks

    # -------------------------------------------------
    # RGB AND BRIGHTNESS EXTRACTION
    # -------------------------------------------------

    def _measure_mask(self, frame, mask):

        pixels = frame[mask > 0]

        pixel_count = int(len(pixels))

        if pixel_count < self.min_region_pixels:

            return {
                "rgb": None,
                "brightness": None,
                "pixel_count": pixel_count
            }

        mean_bgr = np.mean(
            pixels.astype(np.float32),
            axis=0
        )

        blue, green, red = mean_bgr

        gray_pixels = cv2.cvtColor(
            pixels.reshape(-1, 1, 3),
            cv2.COLOR_BGR2GRAY
        )

        region_brightness = float(
            np.mean(gray_pixels)
        )

        # Reject measurements under very poor lighting.
        if region_brightness < self.min_brightness:

            return {
                "rgb": None,
                "brightness": None,
                "pixel_count": pixel_count
            }

        return {
            "rgb": {
                "r": float(red),
                "g": float(green),
                "b": float(blue)
            },

            "brightness": region_brightness,

            "pixel_count": pixel_count
        }

    # -------------------------------------------------
    # EYE ASPECT RATIO (EAR)
    # -------------------------------------------------

    def _calculate_ear(self, points, eye_indices):

        eye = points[eye_indices]

        horizontal = np.linalg.norm(
            eye[0] - eye[3]
        )

        if horizontal < 1e-6:
            return None

        vertical_1 = np.linalg.norm(
            eye[1] - eye[5]
        )

        vertical_2 = np.linalg.norm(
            eye[2] - eye[4]
        )

        ear = (
            vertical_1 + vertical_2
        ) / (2.0 * horizontal)

        return float(ear)

    # -------------------------------------------------
    # MOUTH ASPECT RATIO (MAR)
    # -------------------------------------------------

    def _calculate_mar(self, points):

        vertical_1 = np.linalg.norm(
            points[13] - points[14]
        )

        vertical_2 = np.linalg.norm(
            points[82] - points[87]
        )

        vertical_3 = np.linalg.norm(
            points[312] - points[317]
        )

        horizontal = np.linalg.norm(
            points[MOUTH_LEFT] -
            points[MOUTH_RIGHT]
        )

        if horizontal < 1e-6:
            return None

        mar = (
            vertical_1 +
            vertical_2 +
            vertical_3
        ) / (2.0 * horizontal)

        return float(mar)

    # -------------------------------------------------
    # APPROXIMATE HEAD POSE
    # -------------------------------------------------

    def _calculate_head_pose(self, points, frame_shape):

        height, width = frame_shape[:2]

        image_points = np.array([
            points[1],    # Nose tip
            points[152],  # Chin
            points[33],   # Left eye outer corner
            points[263],  # Right eye outer corner
            points[61],   # Left mouth corner
            points[291]   # Right mouth corner
        ], dtype=np.float64)

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ], dtype=np.float64)

        focal_length = float(width)

        camera_matrix = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        distortion = np.zeros((4, 1))

        try:

            success, rotation_vector, translation_vector = (
                cv2.solvePnP(
                    model_points,
                    image_points,
                    camera_matrix,
                    distortion,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            )

            if not success:
                return None, None, None

            rotation_matrix, _ = cv2.Rodrigues(
                rotation_vector
            )

            projection_matrix = np.hstack([
                rotation_matrix,
                translation_vector
            ])

            _, _, _, _, _, _, euler_angles = (
                cv2.decomposeProjectionMatrix(
                    projection_matrix
                )
            )

            pitch = float(euler_angles[0, 0])
            yaw = float(euler_angles[1, 0])
            roll = float(euler_angles[2, 0])

            # Normalize pitch to a front-facing angle range.
            if pitch > 90:
                pitch -= 180

            elif pitch < -90:
                pitch += 180

            return yaw, pitch, roll

        except (
            cv2.error,
            ValueError,
            np.linalg.LinAlgError
        ):

            return None, None, None

    # -------------------------------------------------
    # ALERTNESS MEASUREMENTS
    # -------------------------------------------------

    def _extract_alertness(self, points, frame_shape):

        ear_left = self._calculate_ear(
            points,
            LEFT_EYE
        )

        ear_right = self._calculate_ear(
            points,
            RIGHT_EYE
        )

        mar = self._calculate_mar(points)

        yaw, pitch, roll = self._calculate_head_pose(
            points,
            frame_shape
        )

        if ear_left is None or ear_right is None:

            eye_closed = None

        else:

            eye_closed = bool(
                ear_left < self.ear_threshold
                and ear_right < self.ear_threshold
            )

        return {
            "ear_left": ear_left,
            "ear_right": ear_right,
            "eye_closed": eye_closed,
            "mouth_aspect_ratio": mar,
            "head_yaw": yaw,
            "head_pitch": pitch,
            "head_roll": roll
        }

    # -------------------------------------------------
    # MAIN EXTRACTION FUNCTION
    # -------------------------------------------------

    def extract(
        self,
        frame,
        landmarks,
        timestamp,
        wall_time=None
    ):

        invalid_alertness = {
            "ear_left": None,
            "ear_right": None,
            "eye_closed": None,
            "mouth_aspect_ratio": None,
            "head_yaw": None,
            "head_pitch": None,
            "head_roll": None
        }

        # ---------------------------------------------
        # MISSING LANDMARK HANDLING
        # ---------------------------------------------

        if landmarks is None or len(landmarks) < 478:

            self.previous_landmarks = None

            empty_rgb = {
                name: None for name in REGION_NAMES
            }

            empty_brightness = {
                name: None for name in REGION_NAMES
            }

            empty_counts = {
                name: 0 for name in REGION_NAMES
            }

            return {
                "timestamp": timestamp,
                "wall_time": wall_time,
                "rgb": empty_rgb,
                "motion": None,
                "track_ok": False,
                "brightness": empty_brightness,
                "pixel_count": empty_counts,
                "alertness": invalid_alertness,

                # Backward-compatible fields.
                "tracking_status": False,
                "motion_displacement": None,
                "regions": None,
                "overall_brightness": None
            }

        points = np.asarray(
            landmarks,
            dtype=np.float32
        )

        if not np.all(np.isfinite(points)):

            self.previous_landmarks = None

            return {
                "timestamp": timestamp,
                "wall_time": wall_time,
                "rgb": {name: None for name in REGION_NAMES},
                "motion": None,
                "track_ok": False,
                "brightness": {name: None for name in REGION_NAMES},
                "pixel_count": {name: 0 for name in REGION_NAMES},
                "alertness": invalid_alertness,

                "tracking_status": False,
                "motion_displacement": None,
                "regions": None,
                "overall_brightness": None
            }

        # ---------------------------------------------
        # MOTION DISPLACEMENT
        # ---------------------------------------------

        if self.previous_landmarks is None:

            motion = 0.0

        else:

            previous = self.previous_landmarks

            motion = float(
                np.mean(
                    np.linalg.norm(
                        points - previous,
                        axis=1
                    )
                )
            )

        self.previous_landmarks = points.copy()

        # ---------------------------------------------
        # EXTRACT ROI MASKS
        # ---------------------------------------------

        masks = self.get_roi_masks(
            frame,
            points
        )

        rgb = {}
        brightness = {}
        pixel_count = {}
        detailed_regions = {}

        for region_name in REGION_NAMES:

            measurement = self._measure_mask(
                frame,
                masks[region_name]
            )

            rgb[region_name] = measurement["rgb"]

            brightness[region_name] = (
                measurement["brightness"]
            )

            pixel_count[region_name] = (
                measurement["pixel_count"]
            )

            detailed_regions[region_name] = measurement

        # ---------------------------------------------
        # OVERALL BRIGHTNESS
        # ---------------------------------------------

        valid_brightness = [
            value
            for value in brightness.values()
            if value is not None
        ]

        overall_brightness = (
            float(np.mean(valid_brightness))
            if valid_brightness else None
        )

        # ---------------------------------------------
        # BACKWARD-COMPATIBLE REGION DATA
        # ---------------------------------------------

        def combine_regions(region_names):

            valid = [
                detailed_regions[name]
                for name in region_names
                if detailed_regions[name]["rgb"] is not None
            ]

            if not valid:
                return None

            combined_rgb = {}

            for channel in ("r", "g", "b"):

                combined_rgb[channel] = float(
                    np.mean([
                        item["rgb"][channel]
                        for item in valid
                    ])
                )

            valid_brightness_values = [
                item["brightness"]
                for item in valid
                if item["brightness"] is not None
            ]

            return {
                "rgb": combined_rgb,

                "brightness": (
                    float(np.mean(valid_brightness_values))
                    if valid_brightness_values else None
                ),

                "pixel_count": sum(
                    item["pixel_count"]
                    for item in valid
                )
            }

        legacy_regions = {

            "forehead": combine_regions([
                "forehead_center",
                "forehead_left",
                "forehead_right"
            ]),

            "left_cheek": combine_regions([
                "left_cheek_upper",
                "left_cheek_lower"
            ]),

            "right_cheek": combine_regions([
                "right_cheek_upper",
                "right_cheek_lower"
            ])
        }

        # ---------------------------------------------
        # ALERTNESS EXTRACTION
        # ---------------------------------------------

        alertness = self._extract_alertness(
            points,
            frame.shape
        )

        # ---------------------------------------------
        # FINAL FRAME DATA CONTRACT
        # ---------------------------------------------

        return {

            "timestamp": timestamp,

            "wall_time": wall_time,

            "rgb": rgb,

            "motion": motion,

            "track_ok": True,

            "brightness": brightness,

            "pixel_count": pixel_count,

            "alertness": alertness,

            # Backward-compatible fields.
            "tracking_status": True,

            "motion_displacement": motion,

            "regions": legacy_regions,

            "overall_brightness": overall_brightness
        }
