"""
MediaPipe Face Mesh Processor

This module handles facial landmark detection using MediaPipe Face Mesh.
It provides a clean interface for extracting facial landmarks and eye regions.
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import mediapipe as mp
import cv2
from dataclasses import dataclass
import time

from config.settings import Config, get_config


logger = logging.getLogger(__name__)


@dataclass
class FaceLandmarks:
    """Container for face landmark data"""
    landmarks: np.ndarray  # Shape: (468, 3) - x, y, z coordinates
    confidence: float
    timestamp: float
    
    # Pre-computed eye regions for efficiency
    left_eye_landmarks: np.ndarray
    right_eye_landmarks: np.ndarray
    
    # Face bounding box
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    
    def get_landmark(self, index: int) -> np.ndarray:
        """Get specific landmark by index"""
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        raise IndexError(f"Landmark index {index} out of range")
    
    def get_eye_center(self, eye: str = 'left') -> np.ndarray:
        """Get center point of eye region"""
        landmarks = self.left_eye_landmarks if eye == 'left' else self.right_eye_landmarks
        return np.mean(landmarks, axis=0)
    
    def get_eye_aspect_ratio(self, eye: str = 'left') -> float:
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        landmarks = self.left_eye_landmarks if eye == 'left' else self.right_eye_landmarks
        
        # Approximate vertical distances
        vertical_dist = np.mean([
            np.linalg.norm(landmarks[1] - landmarks[5]),
            np.linalg.norm(landmarks[2] - landmarks[4])
        ])
        
        # Horizontal distance
        horizontal_dist = np.linalg.norm(landmarks[0] - landmarks[3])
        
        # EAR calculation
        if horizontal_dist > 0:
            return vertical_dist / horizontal_dist
        return 0.0


class MediaPipeProcessor:
    """
    MediaPipe Face Mesh processor for facial landmark detection.
    
    This class wraps MediaPipe functionality and provides a clean interface
    for the eye tracking system.
    """
    
    # Key landmark indices
    LEFT_EYE_INDICES = list(range(33, 42)) + [133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = list(range(362, 374)) + [362, 398, 384, 385, 386, 387, 388, 466]
    
    # Iris landmark indices (for MediaPipe with iris tracking)
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize MediaPipe processor.
        
        Args:
            config: Configuration object, uses default if None
        """
        self.config = config or get_config()
        self.mp_config = self.config.mediapipe
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create face mesh instance
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=self.mp_config.static_image_mode,
            max_num_faces=self.mp_config.max_num_faces,
            refine_landmarks=self.mp_config.refine_landmarks,
            min_detection_confidence=self.mp_config.min_detection_confidence,
            min_tracking_confidence=self.mp_config.min_tracking_confidence
        )
        
        # Performance tracking
        self.frame_count = 0
        self.last_process_time = 0.0
        self.avg_process_time = 0.0
        self.detection_success_rate = 0.0
        
        logger.info(f"MediaPipe processor initialized with mode: {self.config.system.performance_mode.value}")
    
    def process_frame(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        """
        Process a single frame and extract facial landmarks.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            FaceLandmarks object if face detected, None otherwise
        """
        start_time = time.time()
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        # Update performance metrics
        process_time = time.time() - start_time
        self._update_performance_metrics(process_time, results is not None)
        
        if not results.multi_face_landmarks:
            logger.debug("No face detected in frame")
            return None
        
        # Get the first face (we're only tracking one person)
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array
        height, width = frame.shape[:2]
        landmarks_array = self._landmarks_to_array(face_landmarks, width, height)
        
        # Extract eye regions
        left_eye = landmarks_array[self.LEFT_EYE_INDICES]
        right_eye = landmarks_array[self.RIGHT_EYE_INDICES]
        
        # Calculate bounding box
        bbox = self._calculate_bbox(landmarks_array)
        
        # Create FaceLandmarks object
        face_data = FaceLandmarks(
            landmarks=landmarks_array,
            confidence=self._estimate_confidence(face_landmarks),
            timestamp=time.time(),
            left_eye_landmarks=left_eye,
            right_eye_landmarks=right_eye,
            bbox=bbox
        )
        
        return face_data
    
    def _landmarks_to_array(self, landmarks, width: int, height: int) -> np.ndarray:
        """Convert MediaPipe landmarks to numpy array with pixel coordinates"""
        landmark_array = []
        
        for landmark in landmarks.landmark:
            x = landmark.x * width
            y = landmark.y * height
            z = landmark.z * width  # Z is in the same scale as X
            landmark_array.append([x, y, z])
        
        return np.array(landmark_array)
    
    def _calculate_bbox(self, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """Calculate bounding box for face landmarks"""
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        
        # Add some padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        width = x_max - x_min + 2 * padding
        height = y_max - y_min + 2 * padding
        
        return (x_min, y_min, width, height)
    
    def _estimate_confidence(self, landmarks) -> float:
        """Estimate detection confidence based on landmark visibility"""
        # MediaPipe doesn't directly provide confidence for face mesh
        # We estimate it based on landmark visibility and consistency
        visibilities = [lm.visibility for lm in landmarks.landmark if hasattr(lm, 'visibility')]
        
        if visibilities:
            return float(np.mean(visibilities))
        return 0.5  # Default confidence
    
    def _update_performance_metrics(self, process_time: float, success: bool):
        """Update internal performance metrics"""
        self.frame_count += 1
        
        # Exponential moving average for process time
        alpha = 0.1
        self.avg_process_time = (1 - alpha) * self.avg_process_time + alpha * process_time
        
        # Success rate
        if self.frame_count == 1:
            self.detection_success_rate = 1.0 if success else 0.0
        else:
            self.detection_success_rate = ((self.frame_count - 1) * self.detection_success_rate + 
                                         (1.0 if success else 0.0)) / self.frame_count
    
    def draw_landmarks(self, frame: np.ndarray, face_landmarks: FaceLandmarks, 
                      draw_all: bool = False) -> np.ndarray:
        """
        Draw landmarks on frame for visualization.
        
        Args:
            frame: Input frame
            face_landmarks: Detected face landmarks
            draw_all: If True, draw all landmarks. Otherwise, only eyes.
            
        Returns:
            Frame with drawn landmarks
        """
        frame_copy = frame.copy()
        
        if draw_all:
            # Draw all face mesh connections
            # Convert back to MediaPipe format for drawing
            mp_landmarks = self._array_to_mp_landmarks(face_landmarks.landmarks, frame.shape)
            self.mp_drawing.draw_landmarks(
                frame_copy,
                mp_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                None,
                self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        
        # Always draw eye regions with special highlighting
        for landmark in face_landmarks.left_eye_landmarks:
            cv2.circle(frame_copy, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0), -1)
        
        for landmark in face_landmarks.right_eye_landmarks:
            cv2.circle(frame_copy, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0), -1)
        
        # Draw eye centers
        left_center = face_landmarks.get_eye_center('left')
        right_center = face_landmarks.get_eye_center('right')
        cv2.circle(frame_copy, (int(left_center[0]), int(left_center[1])), 4, (255, 0, 0), -1)
        cv2.circle(frame_copy, (int(right_center[0]), int(right_center[1])), 4, (255, 0, 0), -1)
        
        # Draw bounding box
        x, y, w, h = face_landmarks.bbox
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
        # Add performance info
        if self.config.system.debug_mode:
            fps = 1.0 / self.avg_process_time if self.avg_process_time > 0 else 0
            cv2.putText(frame_copy, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_copy, f"Detection Rate: {self.detection_success_rate:.1%}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame_copy
    
    def _array_to_mp_landmarks(self, landmarks: np.ndarray, frame_shape: Tuple[int, int, int]):
        """Convert numpy array back to MediaPipe landmark format for drawing"""
        mp_landmarks = type('obj', (object,), {'landmark': []})()
        height, width = frame_shape[:2]
        
        for landmark in landmarks:
            mp_landmark = type('obj', (object,), {
                'x': landmark[0] / width,
                'y': landmark[1] / height,
                'z': landmark[2] / width
            })()
            mp_landmarks.landmark.append(mp_landmark)
        
        return mp_landmarks
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            'avg_process_time_ms': self.avg_process_time * 1000,
            'fps': 1.0 / self.avg_process_time if self.avg_process_time > 0 else 0,
            'detection_success_rate': self.detection_success_rate,
            'frames_processed': self.frame_count
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.frame_count = 0
        self.avg_process_time = 0.0
        self.detection_success_rate = 0.0
    
    def close(self):
        """Clean up resources"""
        self.face_mesh.close()
        logger.info("MediaPipe processor closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Utility functions for standalone use
def detect_face_landmarks(frame: np.ndarray, config: Optional[Config] = None) -> Optional[FaceLandmarks]:
    """
    Detect face landmarks in a single frame.
    
    This is a convenience function for one-off detections.
    
    Args:
        frame: Input frame (BGR format)
        config: Optional configuration
        
    Returns:
        FaceLandmarks if detected, None otherwise
    """
    processor = MediaPipeProcessor(config)
    try:
        return processor.process_frame(frame)
    finally:
        processor.close()


def draw_face_mesh(frame: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
    """
    Draw face mesh on frame.
    
    Convenience function for visualization.
    
    Args:
        frame: Input frame
        landmarks: Face landmarks to draw
        
    Returns:
        Frame with drawn landmarks
    """
    processor = MediaPipeProcessor()
    try:
        return processor.draw_landmarks(frame, landmarks, draw_all=True)
    finally:
        processor.close()