"""
Camera Manager Module

Handles webcam capture, resolution management, and frame buffering.
Provides a robust interface for cross-platform camera access.
"""

import cv2
import numpy as np
import logging
import threading
import queue
import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import platform

from config.settings import Config, get_config


logger = logging.getLogger(__name__)


@dataclass
class CameraInfo:
    """Camera device information"""
    device_id: int
    name: str
    resolution: Tuple[int, int]
    fps: float
    backend: str
    is_available: bool
    supported_resolutions: List[Tuple[int, int]]


class CameraManager:
    """
    Manages camera capture with automatic fallbacks and optimization.
    
    Features:
    - Automatic resolution fallback
    - Frame buffering for smooth capture
    - Cross-platform compatibility
    - Performance monitoring
    - Thread-safe operation
    """
    
    # Platform-specific backends
    BACKENDS = {
        'Windows': cv2.CAP_DSHOW,
        'Linux': cv2.CAP_V4L2,
        'Darwin': cv2.CAP_AVFOUNDATION  # macOS
    }
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize camera manager.
        
        Args:
            config: Configuration object, uses default if None
        """
        self.config = config or get_config()
        self.camera_config = self.config.camera
        
        # Camera capture object
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_device_id = self.camera_config.device_id
        
        # Camera properties
        self.actual_resolution: Optional[Tuple[int, int]] = None
        self.actual_fps: float = 0.0
        
        # Frame buffer for smooth capture
        self.frame_buffer = queue.Queue(maxsize=self.camera_config.buffer_size)
        self.capture_thread: Optional[threading.Thread] = None
        self.is_capturing = False
        
        # Performance tracking
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_frame_time = 0.0
        self.avg_frame_interval = 0.0
        
        # Initialize camera
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera with appropriate backend and settings"""
        system = platform.system()
        backend = self.BACKENDS.get(system, cv2.CAP_ANY)
        
        logger.info(f"Initializing camera on {system} with backend {backend}")
        
        # Try to open camera
        self.cap = cv2.VideoCapture(self.current_device_id, backend)
        
        if not self.cap.isOpened():
            # Try without specific backend
            logger.warning(f"Failed with backend {backend}, trying default")
            self.cap = cv2.VideoCapture(self.current_device_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.current_device_id}")
        
        # Set camera properties
        self._configure_camera()
        
        # Get actual properties
        self.actual_resolution = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera initialized: {self.actual_resolution[0]}x{self.actual_resolution[1]} @ {self.actual_fps:.1f} FPS")
    
    def _configure_camera(self):
        """Configure camera settings"""
        # Try to set target resolution
        target_width = self.camera_config.width
        target_height = self.camera_config.height
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        
        # Check if resolution was set successfully
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if (actual_width != target_width or actual_height != target_height):
            logger.warning(f"Target resolution {target_width}x{target_height} not supported, "
                         f"using {actual_width}x{actual_height}")
            
            # Try fallback resolutions
            for res in self.camera_config.fallback_resolutions:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
                
                if (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == res[0] and
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == res[1]):
                    logger.info(f"Using fallback resolution: {res[0]}x{res[1]}")
                    break
        
        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_config.fps)
        
        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set other properties if specified
        if self.camera_config.brightness is not None:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.camera_config.brightness)
        
        if self.camera_config.contrast is not None:
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.camera_config.contrast)
        
        # Disable auto-exposure if requested (camera-dependent)
        if not self.camera_config.auto_exposure:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
    
    def start_capture(self):
        """Start asynchronous frame capture"""
        if self.is_capturing:
            logger.warning("Capture already started")
            return
        
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("Camera capture started")
    
    def stop_capture(self):
        """Stop asynchronous frame capture"""
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        # Clear buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Camera capture stopped")
    
    def _capture_loop(self):
        """Background thread for capturing frames"""
        while self.is_capturing:
            ret, frame = self.cap.read()
            
            if ret:
                # Update timing
                current_time = time.time()
                if self.last_frame_time > 0:
                    interval = current_time - self.last_frame_time
                    self.avg_frame_interval = 0.9 * self.avg_frame_interval + 0.1 * interval
                self.last_frame_time = current_time
                
                # Add to buffer
                try:
                    # Non-blocking put to avoid delays
                    self.frame_buffer.put_nowait((frame, current_time))
                    self.frame_count += 1
                except queue.Full:
                    # Buffer full, drop oldest frame
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put_nowait((frame, current_time))
                        self.dropped_frames += 1
                    except queue.Empty:
                        pass
            else:
                logger.error("Failed to read frame from camera")
                time.sleep(0.1)  # Brief pause before retry
    
    def get_frame(self, timeout: Optional[float] = None) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the latest frame from the camera.
        
        Args:
            timeout: Maximum time to wait for frame (None for blocking)
            
        Returns:
            Tuple of (frame, timestamp) or None if no frame available
        """
        if not self.is_capturing:
            # Synchronous capture
            ret, frame = self.cap.read()
            if ret:
                return frame, time.time()
            return None
        
        # Get from buffer
        try:
            frame, timestamp = self.frame_buffer.get(timeout=timeout)
            return frame, timestamp
        except queue.Empty:
            return None
    
    def get_frame_sync(self) -> Optional[np.ndarray]:
        """
        Get a frame synchronously (bypasses buffer).
        
        Returns:
            Frame or None if capture fails
        """
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                return frame
        return None
    
    def list_available_cameras(self, max_test: int = 10) -> List[CameraInfo]:
        """
        List available camera devices.
        
        Args:
            max_test: Maximum number of devices to test
            
        Returns:
            List of available cameras
        """
        cameras = []
        
        for i in range(max_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Test common resolutions
                supported_res = []
                for res in [(1920, 1080), (1280, 720), (640, 480), (320, 240)]:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
                    if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == res[0] and
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == res[1]):
                        supported_res.append(res)
                
                cameras.append(CameraInfo(
                    device_id=i,
                    name=f"Camera {i}",
                    resolution=(width, height),
                    fps=fps,
                    backend="Unknown",
                    is_available=True,
                    supported_resolutions=supported_res
                ))
                
                cap.release()
        
        return cameras
    
    def switch_camera(self, device_id: int):
        """Switch to a different camera device"""
        logger.info(f"Switching to camera {device_id}")
        
        # Stop current capture
        was_capturing = self.is_capturing
        if was_capturing:
            self.stop_capture()
        
        # Release current camera
        if self.cap:
            self.cap.release()
        
        # Switch device
        self.current_device_id = device_id
        self._initialize_camera()
        
        # Restart capture if it was running
        if was_capturing:
            self.start_capture()
    
    def save_frame(self, frame: np.ndarray, filepath: Path):
        """Save a frame to file"""
        cv2.imwrite(str(filepath), frame)
        logger.debug(f"Frame saved to {filepath}")
    
    def get_camera_properties(self) -> Dict[str, Any]:
        """Get current camera properties"""
        if not self.cap or not self.cap.isOpened():
            return {}
        
        return {
            'device_id': self.current_device_id,
            'resolution': self.actual_resolution,
            'fps': self.actual_fps,
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
            'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE),
            'backend': self.cap.get(cv2.CAP_PROP_BACKEND)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get capture performance statistics"""
        actual_fps = 1.0 / self.avg_frame_interval if self.avg_frame_interval > 0 else 0
        
        return {
            'frames_captured': self.frame_count,
            'frames_dropped': self.dropped_frames,
            'drop_rate': self.dropped_frames / max(1, self.frame_count),
            'actual_fps': actual_fps,
            'buffer_size': self.frame_buffer.qsize()
        }
    
    def release(self):
        """Release camera resources"""
        self.stop_capture()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Camera released")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


# Utility functions
def test_camera(device_id: int = 0, duration: int = 5) -> bool:
    """
    Test if a camera is working.
    
    Args:
        device_id: Camera device ID
        duration: Test duration in seconds
        
    Returns:
        True if camera works, False otherwise
    """
    try:
        with CameraManager() as cam:
            cam.start_capture()
            
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < duration:
                frame_data = cam.get_frame(timeout=1.0)
                if frame_data:
                    frame_count += 1
            
            fps = frame_count / duration
            logger.info(f"Camera test successful: {frame_count} frames, {fps:.1f} FPS")
            return True
            
    except Exception as e:
        logger.error(f"Camera test failed: {e}")
        return False


def capture_calibration_images(output_dir: Path, count: int = 10) -> List[Path]:
    """
    Capture images for calibration.
    
    Args:
        output_dir: Directory to save images
        count: Number of images to capture
        
    Returns:
        List of saved image paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    
    with CameraManager() as cam:
        logger.info(f"Capturing {count} calibration images...")
        
        for i in range(count):
            # Wait for user to position
            input(f"Position for image {i+1}/{count} and press Enter...")
            
            frame = cam.get_frame_sync()
            if frame is not None:
                filepath = output_dir / f"calibration_{i:03d}.jpg"
                cam.save_frame(frame, filepath)
                saved_paths.append(filepath)
                logger.info(f"Saved {filepath}")
        
    return saved_paths