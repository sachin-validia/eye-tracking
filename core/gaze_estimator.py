"""
Gaze Estimation Module

Implements gaze estimation algorithms using facial landmarks.
Provides both 2D and 3D gaze estimation methods.
"""

import numpy as np
import cv2
import logging
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from scipy.signal import butter, filtfilt
import time

from config.settings import Config, get_config
from core.mediapipe_processor import FaceLandmarks


logger = logging.getLogger(__name__)


@dataclass
class GazeVector:
    """3D gaze direction vector"""
    origin: np.ndarray  # Eye center position (x, y, z)
    direction: np.ndarray  # Normalized direction vector
    confidence: float
    
    def to_angles(self) -> Tuple[float, float]:
        """Convert to yaw and pitch angles in degrees"""
        # Yaw: horizontal angle (left-right)
        yaw = np.arctan2(self.direction[0], -self.direction[2]) * 180 / np.pi
        
        # Pitch: vertical angle (up-down)
        pitch = np.arcsin(self.direction[1]) * 180 / np.pi
        
        return yaw, pitch


@dataclass
class GazeEstimation:
    """Complete gaze estimation result"""
    timestamp: float
    
    # 2D gaze point on screen (normalized 0-1)
    screen_point: Tuple[float, float]
    
    # 3D gaze vectors for each eye
    left_gaze: Optional[GazeVector]
    right_gaze: Optional[GazeVector]
    combined_gaze: GazeVector
    
    # Head pose angles (degrees)
    head_pitch: float
    head_yaw: float
    head_roll: float
    
    # Eye states
    left_eye_openness: float  # 0-1
    right_eye_openness: float  # 0-1
    is_blinking: bool
    
    # Quality metrics
    tracking_confidence: float


class KalmanFilter:
    """Simple Kalman filter for smoothing gaze data"""
    
    def __init__(self, process_variance: float = 1e-3, measurement_variance: float = 0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        self.covariance = np.eye(4) * 0.1
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise
        self.Q = np.eye(4) * process_variance
        
        # Measurement noise
        self.R = np.eye(2) * measurement_variance
        
        self.initialized = False
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update filter with new measurement"""
        if not self.initialized:
            self.state[:2] = measurement
            self.initialized = True
            return measurement
        
        # Predict
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        # Update
        y = measurement - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance
        
        return self.state[:2]


class GazeEstimator:
    """
    Estimates gaze direction from facial landmarks.
    
    Implements multiple gaze estimation methods:
    - Geometric eye center method
    - Pupil-based estimation (when integrated with GazeTracking)
    - 3D model-based estimation
    """
    
    # Camera matrix (will be calibrated or use defaults)
    DEFAULT_CAMERA_MATRIX = np.array([
        [640, 0, 320],
        [0, 640, 240],
        [0, 0, 1]
    ], dtype=float)
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize gaze estimator"""
        self.config = config or get_config()
        self.gaze_config = self.config.gaze_estimation
        
        # Initialize filters if enabled
        self.kalman_filter = None
        if self.gaze_config.use_kalman_filter:
            self.kalman_filter = KalmanFilter(
                self.gaze_config.kalman_q,
                self.gaze_config.kalman_r
            )
        
        # Smoothing buffer
        self.gaze_history: List[Tuple[float, float]] = []
        
        # Camera parameters (to be calibrated)
        self.camera_matrix = self.DEFAULT_CAMERA_MATRIX.copy()
        self.dist_coeffs = np.zeros(5)
        
        # 3D face model points for head pose estimation
        self.face_model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye corner
            (225.0, 170.0, -135.0),    # Right eye corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ], dtype=float) / 4.5
        
        logger.info("Gaze estimator initialized")
    
    def estimate_gaze(self, face_landmarks: FaceLandmarks, 
                     frame_shape: Tuple[int, int]) -> Optional[GazeEstimation]:
        """
        Estimate gaze from facial landmarks.
        
        Args:
            face_landmarks: Detected facial landmarks
            frame_shape: Shape of the input frame (height, width)
            
        Returns:
            GazeEstimation object or None if estimation fails
        """
        height, width = frame_shape
        
        # Update camera matrix for current resolution
        self.camera_matrix[0, 0] = width
        self.camera_matrix[1, 1] = width
        self.camera_matrix[0, 2] = width / 2
        self.camera_matrix[1, 2] = height / 2
        
        # Estimate head pose
        head_pose = self._estimate_head_pose(face_landmarks)
        if head_pose is None:
            return None
        
        # Calculate eye openness
        left_eye_openness = self._calculate_eye_openness(face_landmarks, 'left')
        right_eye_openness = self._calculate_eye_openness(face_landmarks, 'right')
        is_blinking = (left_eye_openness < self.gaze_config.blink_threshold and 
                      right_eye_openness < self.gaze_config.blink_threshold)
        
        # Estimate gaze for each eye
        left_gaze = None
        right_gaze = None
        
        if left_eye_openness > self.gaze_config.blink_threshold:
            left_gaze = self._estimate_eye_gaze(face_landmarks, 'left', head_pose)
        
        if right_eye_openness > self.gaze_config.blink_threshold:
            right_gaze = self._estimate_eye_gaze(face_landmarks, 'right', head_pose)
        
        # Combine gaze vectors
        if left_gaze and right_gaze:
            combined_gaze = self._combine_gaze_vectors(left_gaze, right_gaze)
        elif left_gaze:
            combined_gaze = left_gaze
        elif right_gaze:
            combined_gaze = right_gaze
        else:
            # Both eyes closed or tracking failed
            return None
        
        # Convert to screen coordinates
        screen_point = self._gaze_to_screen_point(combined_gaze, frame_shape)
        
        # Apply smoothing
        if self.gaze_config.use_kalman_filter and self.kalman_filter:
            smoothed_point = self.kalman_filter.update(np.array(screen_point))
            screen_point = tuple(smoothed_point)
        
        # Create result
        estimation = GazeEstimation(
            timestamp=time.time(),
            screen_point=screen_point,
            left_gaze=left_gaze,
            right_gaze=right_gaze,
            combined_gaze=combined_gaze,
            head_pitch=head_pose['pitch'],
            head_yaw=head_pose['yaw'],
            head_roll=head_pose['roll'],
            left_eye_openness=left_eye_openness,
            right_eye_openness=right_eye_openness,
            is_blinking=is_blinking,
            tracking_confidence=face_landmarks.confidence
        )
        
        # Update history
        self._update_gaze_history(screen_point)
        
        return estimation
    
    def _estimate_head_pose(self, face_landmarks: FaceLandmarks) -> Optional[Dict[str, float]]:
        """Estimate head pose using PnP algorithm"""
        # Select key facial points
        image_points = np.array([
            face_landmarks.landmarks[1],    # Nose tip
            face_landmarks.landmarks[152],  # Chin
            face_landmarks.landmarks[33],   # Left eye corner
            face_landmarks.landmarks[263],  # Right eye corner
            face_landmarks.landmarks[61],   # Left mouth corner
            face_landmarks.landmarks[291]   # Right mouth corner
        ], dtype=float)[:, :2]  # Use only x, y
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.face_model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
        
        # Convert rotation vector to Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            cv2.vconcat((pose_matrix, np.array([[0, 0, 0, 1]])))
        )
        
        return {
            'pitch': euler_angles[0][0],
            'yaw': euler_angles[1][0],
            'roll': euler_angles[2][0],
            'rotation_vector': rotation_vector,
            'translation_vector': translation_vector
        }
    
    def _estimate_eye_gaze(self, face_landmarks: FaceLandmarks, 
                          eye: str, head_pose: Dict[str, Any]) -> Optional[GazeVector]:
        """Estimate gaze vector for a single eye"""
        # Get eye landmarks
        eye_landmarks = (face_landmarks.left_eye_landmarks if eye == 'left' 
                        else face_landmarks.right_eye_landmarks)
        
        # Calculate eye center
        eye_center = np.mean(eye_landmarks, axis=0)
        
        # Simple geometric method for Phase 1
        # More sophisticated methods will be added in Phase 2
        
        # Get iris approximation (using inner eye corners)
        if eye == 'left':
            inner_corner = face_landmarks.landmarks[133]
            outer_corner = face_landmarks.landmarks[33]
        else:
            inner_corner = face_landmarks.landmarks[362]
            outer_corner = face_landmarks.landmarks[263]
        
        # Estimate pupil position (simplified for Phase 1)
        # In Phase 2, this will use actual pupil detection
        eye_width = np.linalg.norm(outer_corner - inner_corner)
        
        # Create gaze vector based on eye center offset and head pose
        # This is a simplified version - Phase 2 will use proper 3D modeling
        gaze_direction = np.array([0, 0, -1])  # Default forward gaze
        
        # Apply head rotation
        rotation_matrix, _ = cv2.Rodrigues(head_pose['rotation_vector'])
        gaze_direction = rotation_matrix @ gaze_direction
        
        # Normalize
        gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)
        
        return GazeVector(
            origin=eye_center,
            direction=gaze_direction,
            confidence=0.7  # Simplified confidence for Phase 1
        )
    
    def _combine_gaze_vectors(self, left_gaze: GazeVector, 
                            right_gaze: GazeVector) -> GazeVector:
        """Combine left and right eye gaze vectors"""
        # Average the directions
        combined_direction = (left_gaze.direction + right_gaze.direction) / 2
        combined_direction = combined_direction / np.linalg.norm(combined_direction)
        
        # Average the origins
        combined_origin = (left_gaze.origin + right_gaze.origin) / 2
        
        # Average confidence
        combined_confidence = (left_gaze.confidence + right_gaze.confidence) / 2
        
        return GazeVector(
            origin=combined_origin,
            direction=combined_direction,
            confidence=combined_confidence
        )
    
    def _gaze_to_screen_point(self, gaze: GazeVector, 
                             frame_shape: Tuple[int, int]) -> Tuple[float, float]:
        """Convert 3D gaze vector to 2D screen coordinates"""
        height, width = frame_shape
        
        # Simple ray-plane intersection
        # Assume screen is at z=0
        if abs(gaze.direction[2]) < 0.001:
            # Gaze parallel to screen
            return (0.5, 0.5)
        
        # Calculate intersection
        t = -gaze.origin[2] / gaze.direction[2]
        intersection = gaze.origin + t * gaze.direction
        
        # Normalize to 0-1
        x = intersection[0] / width
        y = intersection[1] / height
        
        # Clamp to valid range
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        return (float(x), float(y))
    
    def _calculate_eye_openness(self, face_landmarks: FaceLandmarks, eye: str) -> float:
        """Calculate eye openness (0=closed, 1=fully open)"""
        return face_landmarks.get_eye_aspect_ratio(eye)
    
    def _update_gaze_history(self, point: Tuple[float, float]):
        """Update gaze history for smoothing and analysis"""
        self.gaze_history.append(point)
        
        # Keep only recent history
        max_history = int(self.gaze_config.smoothing_window * 30)  # Assume 30 FPS
        if len(self.gaze_history) > max_history:
            self.gaze_history = self.gaze_history[-max_history:]
    
    def get_smoothed_gaze(self) -> Optional[Tuple[float, float]]:
        """Get smoothed gaze point from history"""
        if len(self.gaze_history) < 3:
            return None
        
        # Simple moving average
        window = min(len(self.gaze_history), self.gaze_config.smoothing_window)
        recent_points = self.gaze_history[-window:]
        
        x_values = [p[0] for p in recent_points]
        y_values = [p[1] for p in recent_points]
        
        return (np.mean(x_values), np.mean(y_values))
    
    def detect_saccade(self) -> bool:
        """Detect rapid eye movement (saccade)"""
        if len(self.gaze_history) < 3:
            return False
        
        # Calculate recent movement speed
        p1 = self.gaze_history[-3]
        p2 = self.gaze_history[-1]
        
        distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        time_diff = 2 / 30.0  # Assume 30 FPS
        
        speed = distance / time_diff
        
        return speed > 0.5  # Threshold for saccade detection
    
    def calibrate_camera(self, calibration_points: List[Tuple[np.ndarray, Tuple[float, float]]]):
        """
        Calibrate camera parameters using known gaze points.
        
        Args:
            calibration_points: List of (face_landmarks, known_screen_point) pairs
        """
        # This is a placeholder for Phase 1
        # Full calibration will be implemented in later phases
        logger.info("Camera calibration not yet implemented")
    
    def reset(self):
        """Reset estimator state"""
        self.gaze_history.clear()
        if self.kalman_filter:
            self.kalman_filter.initialized = False
    
    def draw_gaze(self, frame: np.ndarray, estimation: GazeEstimation) -> np.ndarray:
        """Draw gaze visualization on frame"""
        frame_copy = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw gaze point
        x = int(estimation.screen_point[0] * width)
        y = int(estimation.screen_point[1] * height)
        
        # Draw crosshair
        cv2.line(frame_copy, (x - 20, y), (x + 20, y), (0, 255, 0), 2)
        cv2.line(frame_copy, (x, y - 20), (x, y + 20), (0, 255, 0), 2)
        cv2.circle(frame_copy, (x, y), 10, (0, 255, 0), 2)
        
        # Draw eye states
        info_y = 30
        cv2.putText(frame_copy, f"Left Eye: {estimation.left_eye_openness:.2f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_copy, f"Right Eye: {estimation.right_eye_openness:.2f}", 
                   (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if estimation.is_blinking:
            cv2.putText(frame_copy, "BLINKING", (10, info_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw head pose
        cv2.putText(frame_copy, f"Head Pose: P:{estimation.head_pitch:.1f} Y:{estimation.head_yaw:.1f} R:{estimation.head_roll:.1f}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_copy