import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Optional

from src.core.base_estimator import (
    BasePoseEstimator,
    PersonKeypoints,
    KeypointFormat,
    create_person_keypoints,
    filter_keypoints_by_confidence,
)


class BlazePoseAdapter(BasePoseEstimator):
    """
    BlazePose adapter using MediaPipe for pose estimation.

    This adapter integrates Google's BlazePose model through MediaPipe
    with the standardized pose estimation interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize BlazePose adapter.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        self.mp_pose = None
        self.pose_estimator = None
        self.mp_drawing = None
        self.mp_drawing_styles = None

        # Set keypoint format and names for BlazePose (33 landmarks)
        self.keypoint_format = KeypointFormat.CUSTOM
        self.keypoint_names = self._get_blazepose_keypoint_names()

        # BlazePose specific configuration
        self.model_complexity = self.config.get("model", {}).get("complexity", 1)
        self.min_detection_confidence = self.config.get("model", {}).get(
            "min_detection_confidence", 0.5
        )
        self.min_tracking_confidence = self.config.get("model", {}).get(
            "min_tracking_confidence", 0.5
        )
        self.enable_segmentation = self.config.get("model", {}).get(
            "enable_segmentation", False
        )
        self.smooth_landmarks = self.config.get("model", {}).get(
            "smooth_landmarks", True
        )

        self.logger.info("BlazePose adapter initialized")

    def _get_blazepose_keypoint_names(self) -> List[str]:
        """Get BlazePose keypoint names (33 landmarks)"""
        return [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]

    def load_model(self) -> bool:
        """
        Load the BlazePose model using MediaPipe.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

            self.pose_estimator = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                smooth_landmarks=self.smooth_landmarks,
                enable_segmentation=self.enable_segmentation,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

            self.is_loaded = True
            self.logger.info("BlazePose model loaded successfully")
            self.logger.info(f"Model complexity: {self.model_complexity}")
            self.logger.info(f"Detection confidence: {self.min_detection_confidence}")
            self.logger.info(f"Tracking confidence: {self.min_tracking_confidence}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load BlazePose model: {str(e)}")
            self.is_loaded = False
            return False

    def process_frame(self, frame: np.ndarray) -> List[PersonKeypoints]:
        """
        Process a single frame and return detected keypoints.

        Args:
            frame: Input image as numpy array (BGR format)

        Returns:
            List of PersonKeypoints objects for detected people
        """
        if not self.is_loaded:
            self.logger.error("Model not loaded. Call load_model() first.")
            return []

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.pose_estimator.process(rgb_frame)

            detected_people = []

            if results.pose_landmarks:
                keypoints = self._extract_keypoints(results.pose_landmarks, frame.shape)

                filtered_keypoints = filter_keypoints_by_confidence(
                    keypoints, self.confidence_threshold
                )

                person = create_person_keypoints(
                    person_id=0,  # BlazePose detects single person per frame
                    keypoints=filtered_keypoints,
                    keypoint_names=self.keypoint_names,
                    bbox=self._calculate_bounding_box(filtered_keypoints),
                    metadata={
                        "model": "BlazePose",
                        "complexity": self.model_complexity,
                        "segmentation_available": results.segmentation_mask is not None,
                    },
                )

                detected_people.append(person)

            return detected_people

        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return []

    def _extract_keypoints(self, landmarks, frame_shape) -> List[List[float]]:
        """
        Extract keypoints from MediaPipe landmarks.

        Args:
            landmarks: MediaPipe pose landmarks
            frame_shape: Shape of the input frame (height, width, channels)

        Returns:
            List of keypoints in format [x, y, confidence]
        """
        height, width = frame_shape[:2]
        keypoints = []

        for landmark in landmarks.landmark:
            x = landmark.x * width
            y = landmark.y * height
            confidence = (
                landmark.visibility
            )  # wow mediapipe uses visibility as confidence

            keypoints.append([float(x), float(y), float(confidence)])

        return keypoints

    def _calculate_bounding_box(
        self, keypoints: List[List[float]]
    ) -> Optional[List[float]]:
        """
        Calculate bounding box from keypoints.

        Args:
            keypoints: List of keypoints [x, y, confidence]

        Returns:
            Bounding box [x, y, width, height] or None if no valid keypoints
        """
        valid_points = [kp for kp in keypoints if kp[2] > 0]  # confidence > 0

        if not valid_points:
            return None

        x_coords = [kp[0] for kp in valid_points]
        y_coords = [kp[1] for kp in valid_points]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # TODO: correct and add some padding
        padding = 10
        width = max_x - min_x + 2 * padding
        height = max_y - min_y + 2 * padding

        return [max(0, min_x - padding), max(0, min_y - padding), width, height]

    def process_frames_batch(
        self, frames: List[np.ndarray]
    ) -> List[List[PersonKeypoints]]:
        """
        Process a batch of frames.
        Note: MediaPipe doesn't support true batch processing, so we process sequentially.

        Args:
            frames: List of input images as numpy arrays

        Returns:
            List of results, one per frame
        """
        results = []
        for frame in frames:
            results.append(self.process_frame(frame))
        return results

    def draw_keypoints(
        self, image: np.ndarray, person_keypoints: PersonKeypoints
    ) -> np.ndarray:
        """
        Draw keypoints on the image using MediaPipe's drawing utilities.

        Args:
            image: Input image
            person_keypoints: PersonKeypoints object

        Returns:
            Image with drawn keypoints
        """
        if not self.is_loaded or not self.mp_drawing:
            return image

        try:
            # Convert keypoints back to MediaPipe landmark format
            landmarks = self._keypoints_to_landmarks(
                person_keypoints.keypoints, image.shape
            )

            if landmarks:
                # Draw the pose landmarks
                self.mp_drawing.draw_landmarks(
                    image,
                    landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            return image

        except Exception as e:
            self.logger.warning(f"Failed to draw keypoints: {str(e)}")
            return image

    def _keypoints_to_landmarks(self, keypoints: List[List[float]], frame_shape):
        """
        Convert keypoints back to MediaPipe landmark format for visualization.

        Args:
            keypoints: List of keypoints [x, y, confidence]
            frame_shape: Shape of the frame

        Returns:
            MediaPipe landmark list or None
        """
        try:
            height, width = frame_shape[:2]
            landmark_list = []

            for kp in keypoints:
                if len(kp) >= 3 and kp[2] > 0:  # Valid keypoint
                    # Normalize coordinates
                    normalized_landmark = mp.solutions.pose.PoseLandmark()
                    normalized_landmark.x = kp[0] / width
                    normalized_landmark.y = kp[1] / height
                    normalized_landmark.visibility = kp[2]
                    landmark_list.append(normalized_landmark)
                else:
                    # Invalid keypoint
                    normalized_landmark = mp.solutions.pose.PoseLandmark()
                    normalized_landmark.x = 0
                    normalized_landmark.y = 0
                    normalized_landmark.visibility = 0
                    landmark_list.append(normalized_landmark)

            # Create landmark list object
            landmarks = mp.solutions.pose.PoseLandmarkList()
            landmarks.landmark.extend(landmark_list)

            return landmarks

        except Exception as e:
            self.logger.warning(f"Failed to convert keypoints to landmarks: {str(e)}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded BlazePose model.

        Returns:
            Dictionary containing model information
        """
        base_info = super().get_model_info()
        base_info.update(
            {
                "model_name": "BlazePose (MediaPipe)",
                "model_complexity": self.model_complexity,
                "min_detection_confidence": self.min_detection_confidence,
                "min_tracking_confidence": self.min_tracking_confidence,
                "enable_segmentation": self.enable_segmentation,
                "smooth_landmarks": self.smooth_landmarks,
                "num_landmarks": 33,
                "supports_batch_processing": False,
                "supports_single_person": True,
                "supports_multi_person": False,
            }
        )
        return base_info

    def cleanup(self):
        """Clean up MediaPipe resources."""
        if self.pose_estimator:
            self.pose_estimator.close()
            self.pose_estimator = None

        self.mp_pose = None
        self.mp_drawing = None
        self.mp_drawing_styles = None

        super().cleanup()

    def update_model_config(self, **kwargs):
        """
        Update model-specific configuration and reload if necessary.

        Args:
            **kwargs: Configuration parameters to update
        """
        config_changed = False

        if "complexity" in kwargs:
            self.model_complexity = kwargs["complexity"]
            config_changed = True

        if "min_detection_confidence" in kwargs:
            self.min_detection_confidence = kwargs["min_detection_confidence"]
            config_changed = True

        if "min_tracking_confidence" in kwargs:
            self.min_tracking_confidence = kwargs["min_tracking_confidence"]
            config_changed = True

        if "enable_segmentation" in kwargs:
            self.enable_segmentation = kwargs["enable_segmentation"]
            config_changed = True

        if "smooth_landmarks" in kwargs:
            self.smooth_landmarks = kwargs["smooth_landmarks"]
            config_changed = True

        if config_changed and self.is_loaded:
            self.logger.info("Model configuration updated, reloading...")
            self.cleanup()
            self.load_model()


# Example configuration for BlazePose
BLAZEPOSE_DEFAULT_CONFIG = {
    "model": {
        "name": "blazepose",
        "device": "auto",
        "complexity": 1,  # 0, 1, or 2 (higher = more accurate but slower)
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "enable_segmentation": False,
        "smooth_landmarks": True,
    },
    "processing": {
        "batch_size": 1,  # BlazePose processes one frame at a time
        "confidence_threshold": 0.3,
        "frame_processing": {"frame_skip": 1},
    },
    "output": {"format": "json", "json": {"indent": 2, "include_metadata": True}},
    "logging": {"level": "INFO"},
}


def create_blazepose_estimator(
    config: Optional[Dict[str, Any]] = None,
) -> BlazePoseAdapter:
    """
    Factory function to create a BlazePose estimator with default configuration.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured BlazePose adapter instance
    """
    if config is None:
        config = BLAZEPOSE_DEFAULT_CONFIG.copy()

    return BlazePoseAdapter(config)


# exmaple usage
if __name__ == "__main__":
    estimator = create_blazepose_estimator()

    if estimator.load_model():
        print("BlazePose model loaded successfully!")
        print("Model info:", estimator.get_model_info())

        # success = estimator.process_video_batch(
        #     video_path="/home/tanishq/Videos/Webcam/2025-06-03-175130.webm",
        #     output_dir="output/",
        #     frame_skip=1,
        # )

        estimator.cleanup()
    else:
        print("Failed to load BlazePose model!")
