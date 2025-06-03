import os
import json
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

try:
    from utils.config_manager import ConfigManager

    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False


class KeypointFormat(Enum):
    """Supported keypoint formats"""

    COCO17 = "coco17"
    COCO133 = "coco133"
    MPII = "mpii"
    CUSTOM = "custom"


class ProcessingStatus(Enum):
    """Status codes for processing operations"""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass
class PersonKeypoints:
    """Data structure for a single person's keypoints"""

    person_id: int
    keypoints: List[List[float]]
    keypoint_names: List[str]
    bbox: Optional[List[float]] = None
    tracking_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FrameResult:
    """Data structure for frame processing results"""

    frame_id: int
    timestamp: float
    people: List[PersonKeypoints]
    status: ProcessingStatus = ProcessingStatus.SUCCESS
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


@dataclass
class VideoMetadata:
    """Metadata for video processing"""

    model_name: str
    keypoint_format: KeypointFormat
    keypoint_names: List[str]
    confidence_threshold: float
    device: str
    total_frames: int
    fps: float
    video_path: str
    resolution: Tuple[int, int] = (0, 0)
    processing_config: Optional[Dict[str, Any]] = None


class BasePoseEstimator(ABC):
    """
    Abstract base class for pose estimators.

    This class defines the standard interface that all pose estimation
    implementations should follow. It provides common functionality
    and enforces implementation of required methods.
    """

    def __init__(self, config: Union[Dict[str, Any], str, None] = None):
        """
        Initialize the pose estimator with configuration.

        Args:
            config: Configuration dictionary, path to config file, or None for default config
        """
        # Initialize configuration
        if CONFIG_MANAGER_AVAILABLE:
            self.config_manager = ConfigManager()
            if isinstance(config, str):
                # Load from config file
                self.config = self.config_manager.load_config(config_path=config)
            elif isinstance(config, dict):
                # Load from dictionary
                self.config = self.config_manager.load_config(config_dict=config)
            else:
                # Load default config
                self.config = self.config_manager.load_config()
        else:
            # Fallback without ConfigManager
            if isinstance(config, dict):
                self.config = config
            elif config is None:
                self.config = self._get_fallback_config()
            else:
                raise ValueError(
                    "ConfigManager not available. Please provide config as dictionary."
                )
            self.config_manager = None

        # Initialize core attributes
        self.device = None
        self.model = None
        self.logger = self._setup_logger()
        self.is_loaded = False

        # Extract configuration values
        self.confidence_threshold = self.config.get("processing", {}).get(
            "confidence_threshold", 0.3
        )
        self.batch_size = self.config.get("processing", {}).get("batch_size", 1)

        # Initialize format and keypoint information
        self.keypoint_format = KeypointFormat.COCO17
        self.keypoint_names = []

        # Initialize device
        self._initialize_device()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the estimator"""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _get_fallback_config(self) -> Dict[str, Any]:
        """Fallback configuration when ConfigManager is not available"""
        return {
            "model": {"name": "mmpose", "device": "auto"},
            "processing": {
                "batch_size": 1,
                "confidence_threshold": 0.3,
                "frame_processing": {"frame_skip": 1},
            },
            "output": {
                "format": "json",
                "json": {"indent": 2, "include_metadata": True},
            },
            "logging": {"level": "INFO"},
        }

    def _initialize_device(self):
        """Initialize processing device based on configuration"""
        device_config = self.config.get("model", {}).get("device", "auto")

        # to select the best available device
        # TODO: to connect to a cloud device
        if device_config == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    self.device = "cuda"
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    self.device = "mps"  # Special case
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device_config

        self.logger.info(f"Initialized device: {self.device}")

    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the pose estimation model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> List[PersonKeypoints]:
        """
        Process a single frame and return detected keypoints.

        Args:
            frame: Input image as numpy array (BGR format)

        Returns:
            List of PersonKeypoints objects for detected people
        """
        pass

    def process_frames_batch(
        self, frames: List[np.ndarray]
    ) -> List[List[PersonKeypoints]]:
        """
        Process a batch of frames. Default implementation processes each frame individually.
        Subclasses can override for true batch processing.

        Args:
            frames: List of input images as numpy arrays

        Returns:
            List of results, one per frame
        """
        results = []
        for frame in frames:
            results.append(self.process_frame(frame))
        return results

    def process_video_batch(
        self,
        video_path: str,
        output_dir: str,
        frame_skip: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> bool:
        """
        Process entire video and save keypoints.

        Args:
            video_path: Path to input video
            output_dir: Directory to save output files
            frame_skip: Process every Nth frame (default: 1, process all frames)
            start_frame: Frame to start processing from
            end_frame: Frame to stop processing at (None for entire video)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_loaded:
            self.logger.error("Model not loaded. Call load_model() first.")
            return False

        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return False

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return False

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if end_frame is None:
                end_frame = total_frames
            end_frame = min(end_frame, total_frames)

            self.logger.info(f"Processing video: {video_path}")
            self.logger.info(f"Total frames: {total_frames}, FPS: {fps:.2f}")
            self.logger.info(
                f"Processing frames {start_frame} to {end_frame} (skip: {frame_skip})"
            )

            # create video metadata
            # TODO: To store it somewhere else to offload the further procession data on cloud
            metadata = VideoMetadata(
                model_name=self.__class__.__name__,
                keypoint_format=self.keypoint_format,
                keypoint_names=self.keypoint_names,
                confidence_threshold=self.confidence_threshold,
                device=str(self.device),
                total_frames=total_frames,
                fps=fps,
                video_path=os.path.basename(video_path),
                resolution=(width, height),
                processing_config={
                    "frame_skip": frame_skip,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "batch_size": self.batch_size,
                },
            )

            frame_results = self._process_video_frames(
                cap, start_frame, end_frame, frame_skip
            )

            output_path = os.path.join(
                output_dir, f"{Path(video_path).stem}_keypoints.json"
            )
            success = self.export_keypoints(metadata, frame_results, output_path)

            if success:
                self.logger.info(f"Successfully processed {len(frame_results)} frames")
                self.logger.info(f"Output saved to: {output_path}")

            return success

        finally:
            cap.release()

    def _process_video_frames(
        self, cap: cv2.VideoCapture, start_frame: int, end_frame: int, frame_skip: int
    ) -> List[FrameResult]:
        """
        Internal method to process video frames.

        Args:
            cap: OpenCV VideoCapture object
            start_frame: Starting frame index
            end_frame: Ending frame index
            frame_skip: Frame skip interval

        Returns:
            List of FrameResult objects
        """
        frame_results = []
        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        try:
            from tqdm import (
                tqdm,
            )  # if available, if anything better available we can use that

            pbar = tqdm(
                total=(end_frame - start_frame) // frame_skip, desc="Processing frames"
            )
        except ImportError:
            pbar = None

        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (current_frame - start_frame) % frame_skip == 0:
                timestamp = current_frame / fps if fps > 0 else 0

                try:
                    import time

                    start_time = time.time()

                    # Process the frame
                    people = self.process_frame(frame)

                    processing_time = time.time() - start_time

                    result = FrameResult(
                        frame_id=current_frame,
                        timestamp=timestamp,
                        people=people,
                        status=ProcessingStatus.SUCCESS,
                        processing_time=processing_time,
                    )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to process frame {current_frame}: {str(e)}"
                    )
                    result = FrameResult(
                        frame_id=current_frame,
                        timestamp=timestamp,
                        people=[],
                        status=ProcessingStatus.FAILED,
                        error_message=str(e),
                    )

                frame_results.append(result)

                if pbar:
                    pbar.update(1)

            current_frame += 1

        if pbar:
            pbar.close()

        return frame_results

    def export_keypoints(
        self,
        metadata: VideoMetadata,
        frame_results: List[FrameResult],
        output_path: str,
    ) -> bool:
        """
        Export keypoints to JSON file with standardized format.

        Args:
            metadata: Video processing metadata
            frame_results: List of frame processing results
            output_path: Path to save the JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            output_data = {
                "metadata": {
                    "model_name": metadata.model_name,
                    "keypoint_format": metadata.keypoint_format.value,
                    "keypoint_names": metadata.keypoint_names,
                    "confidence_threshold": metadata.confidence_threshold,
                    "device": metadata.device,
                    "total_frames": metadata.total_frames,
                    "fps": metadata.fps,
                    "video_path": metadata.video_path,
                    "resolution": metadata.resolution,
                    "processing_config": metadata.processing_config,
                    "processed_frames": len(frame_results),
                    "success_rate": self._calculate_success_rate(frame_results),
                },
                "frames": [],
            }

            for result in frame_results:
                frame_data = {
                    "frame_id": result.frame_id,
                    "timestamp": result.timestamp,
                    "status": result.status.value,
                    "people": [],
                }

                if result.error_message:
                    frame_data["error"] = result.error_message

                if result.processing_time:
                    frame_data["processing_time"] = result.processing_time

                for person in result.people:
                    person_data = {
                        "person_id": person.person_id,
                        "keypoints": person.keypoints,
                        "keypoint_names": person.keypoint_names,
                    }

                    if person.bbox:
                        person_data["bbox"] = person.bbox

                    if person.tracking_id is not None:
                        person_data["tracking_id"] = person.tracking_id

                    if person.metadata:
                        person_data["metadata"] = person.metadata

                    frame_data["people"].append(person_data)

                output_data["frames"].append(frame_data)

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

            return True

        except Exception as e:
            self.logger.error(f"Failed to export keypoints: {str(e)}")
            return False

    def _calculate_success_rate(self, frame_results: List[FrameResult]) -> float:
        """Calculate the success rate of frame processing"""
        if not frame_results:
            return 0.0

        successful_frames = sum(
            1 for result in frame_results if result.status == ProcessingStatus.SUCCESS
        )
        return (successful_frames / len(frame_results)) * 100

    def load_keypoints(self, json_path: str) -> Optional[Dict[str, Any]]:
        """
        Load keypoints from a previously saved JSON file.

        Args:
            json_path: Path to the keypoints JSON file

        Returns:
            Dictionary containing loaded keypoints data, or None if failed
        """
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            return data
        except Exception as e:
            self.logger.error(f"Failed to load keypoints from {json_path}: {str(e)}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model and configuration.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_type": self.__class__.__name__,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "confidence_threshold": self.confidence_threshold,
            "keypoint_format": self.keypoint_format.value,
            "keypoint_names": self.keypoint_names,
            "num_keypoints": len(self.keypoint_names),
            "is_loaded": self.is_loaded,
        }

    def validate_config(self) -> List[str]:
        """
        Validate the configuration and return any issues found.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required sections
        if "model" not in self.config:
            errors.append("Missing 'model' section in config")

        if "processing" not in self.config:
            errors.append("Missing 'processing' section in config")

        # Validate processing parameters
        processing = self.config.get("processing", {})

        if "confidence_threshold" in processing:
            threshold = processing["confidence_threshold"]
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                errors.append("confidence_threshold must be a number between 0 and 1")

        if "batch_size" in processing:
            batch_size = processing["batch_size"]
            if not isinstance(batch_size, int) or batch_size < 1:
                errors.append("batch_size must be a positive integer")

        return errors

    def cleanup(self):
        """
        Clean up resources. Override in subclasses for model-specific cleanup.
        """
        self.logger.info("Cleaning up resources...")
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False

    def process_image(self, image_path: str) -> List[PersonKeypoints]:
        """
        Process a single image file and return keypoints.

        Args:
            image_path: Path to the input image

        Returns:
            List of PersonKeypoints objects for detected people
        """
        if not self.is_loaded:
            self.logger.error("Model not loaded. Call load_model() first.")
            return []

        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            return []

        try:
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                self.logger.error(f"Cannot load image: {image_path}")
                return []

            # Process frame
            return self.process_frame(frame)

        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return []

    def update_config(self, config_updates: Dict[str, Any]):
        """
        Update configuration parameters.

        Args:
            config_updates: Dictionary of configuration updates
        """
        if self.config_manager:
            # use config manager for proper merging
            self.config = self.config_manager.load_config(override_dict=config_updates)
        else:
            # simple merge for fallback case, if no config is fetched
            def deep_update(base_dict, update_dict):
                for key, value in update_dict.items():
                    if (
                        key in base_dict
                        and isinstance(base_dict[key], dict)
                        and isinstance(value, dict)
                    ):
                        deep_update(base_dict[key], value)
                    else:
                        base_dict[key] = value

            deep_update(self.config, config_updates)

        self.confidence_threshold = self.config.get("processing", {}).get(
            "confidence_threshold", self.confidence_threshold
        )
        self.batch_size = self.config.get("processing", {}).get(
            "batch_size", self.batch_size
        )

        self.logger.info("Configuration updated")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self):
        # TODO: provide execution type to safely exit and close the processes
        """Context manager exit - cleanup resources"""
        self.cleanup()


def create_person_keypoints(
    person_id: int,
    keypoints: List[List[float]],
    keypoint_names: List[str],
    bbox: Optional[List[float]] = None,
    tracking_id: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PersonKeypoints:
    """
    Utility function to create PersonKeypoints object.

    Args:
        person_id: Unique identifier for the person
        keypoints: List of keypoint coordinates [x, y, confidence]
        keypoint_names: List of keypoint names
        bbox: Bounding box [x, y, width, height]
        tracking_id: Tracking identifier across frames
        metadata: Additional metadata

    Returns:
        PersonKeypoints object
    """
    return PersonKeypoints(
        person_id=person_id,
        keypoints=keypoints,
        keypoint_names=keypoint_names,
        bbox=bbox,
        tracking_id=tracking_id,
        metadata=metadata,
    )


def filter_keypoints_by_confidence(
    keypoints: List[List[float]], threshold: float = 0.3
) -> List[List[float]]:
    """
    Filter keypoints based on confidence threshold.

    Args:
        keypoints: List of keypoint coordinates [x, y, confidence]
        threshold: Minimum confidence threshold

    Returns:
        Filtered keypoints list
    """
    filtered = []
    for kp in keypoints:
        if len(kp) >= 3 and kp[2] >= threshold:
            filtered.append(kp)
        else:
            # Set low confidence keypoints to [0, 0, 0]
            filtered.append([0, 0, 0])
    return filtered
