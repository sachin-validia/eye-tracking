"""
Interview Monitor API

Main API interface for the eye tracking system.
Provides high-level methods for interview monitoring and analysis.
"""

import logging
import time
import threading
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import csv
from datetime import datetime
from enum import Enum
import numpy as np

from config.settings import Config, get_config
from core.mediapipe_processor import MediaPipeProcessor
from core.gaze_estimator import GazeEstimator, GazeEstimation
from utils.camera_manager import CameraManager


logger = logging.getLogger(__name__)


class MonitoringState(Enum):
    """System monitoring states"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class MonitoringSession:
    """Container for a monitoring session data"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    
    # Metrics
    total_frames: int = 0
    successful_detections: int = 0
    total_blinks: int = 0
    attention_score: float = 1.0
    
    # Behavioral flags
    suspicious_behaviors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Gaze data
    gaze_data: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_duration(self) -> float:
        """Get session duration in seconds"""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def get_detection_rate(self) -> float:
        """Get face detection success rate"""
        if self.total_frames == 0:
            return 0.0
        return self.successful_detections / self.total_frames
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.get_duration(),
            'total_frames': self.total_frames,
            'successful_detections': self.successful_detections,
            'detection_rate': self.get_detection_rate(),
            'total_blinks': self.total_blinks,
            'attention_score': self.attention_score,
            'suspicious_behaviors': self.suspicious_behaviors,
            'gaze_data_points': len(self.gaze_data)
        }


class InterviewMonitor:
    """
    Main API class for interview monitoring.
    
    Provides high-level interface for:
    - Starting/stopping monitoring sessions
    - Real-time gaze tracking
    - Behavioral analysis
    - Data logging and export
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize interview monitor"""
        self.config = config or get_config()
        
        # Core components
        self.camera_manager = CameraManager(self.config)
        self.mediapipe_processor = MediaPipeProcessor(self.config)
        self.gaze_estimator = GazeEstimator(self.config)
        
        # State management
        self.state = MonitoringState.IDLE
        self.current_session: Optional[MonitoringSession] = None
        
        # Threading
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'on_frame': [],
            'on_gaze': [],
            'on_blink': [],
            'on_suspicious_behavior': [],
            'on_error': []
        }
        
        # Data logging
        self.log_file: Optional[Any] = None
        self.csv_writer: Optional[csv.DictWriter] = None
        
        logger.info("Interview monitor initialized")
    
    def start_monitoring(self, session_id: Optional[str] = None) -> str:
        """
        Start a new monitoring session.
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            Session ID
        """
        if self.state == MonitoringState.RUNNING:
            raise RuntimeError("Monitoring already in progress")
        
        self.state = MonitoringState.INITIALIZING
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create new session
        self.current_session = MonitoringSession(
            session_id=session_id,
            start_time=time.time()
        )
        
        # Initialize data logging
        if self.config.data_logging.enable_logging:
            self._init_data_logging()
        
        # Start camera capture
        self.camera_manager.start_capture()
        
        # Start monitoring thread
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.state = MonitoringState.RUNNING
        logger.info(f"Started monitoring session: {session_id}")
        
        return session_id
    
    def stop_monitoring(self) -> Optional[MonitoringSession]:
        """
        Stop the current monitoring session.
        
        Returns:
            Completed session data
        """
        if self.state != MonitoringState.RUNNING:
            logger.warning("No active monitoring session to stop")
            return None
        
        # Signal stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # Stop camera
        self.camera_manager.stop_capture()
        
        # Close logging
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        
        # Finalize session
        if self.current_session:
            self.current_session.end_time = time.time()
            session = self.current_session
            self.current_session = None
            
            # Save session summary
            self._save_session_summary(session)
            
            logger.info(f"Stopped monitoring session: {session.session_id}")
            return session
        
        self.state = MonitoringState.STOPPED
        return None
    
    def pause_monitoring(self):
        """Pause monitoring (keeps session active)"""
        if self.state == MonitoringState.RUNNING:
            self.state = MonitoringState.PAUSED
            logger.info("Monitoring paused")
    
    def resume_monitoring(self):
        """Resume paused monitoring"""
        if self.state == MonitoringState.PAUSED:
            self.state = MonitoringState.RUNNING
            logger.info("Monitoring resumed")
    
    def _monitoring_loop(self):
        """Main monitoring loop (runs in separate thread)"""
        logger.debug("Monitoring loop started")
        
        try:
            while not self.stop_event.is_set():
                if self.state != MonitoringState.RUNNING:
                    time.sleep(0.1)
                    continue
                
                # Get frame
                frame_data = self.camera_manager.get_frame(timeout=0.1)
                if frame_data is None:
                    continue
                
                frame, timestamp = frame_data
                self.current_session.total_frames += 1
                
                # Process frame
                self._process_frame(frame, timestamp)
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}", exc_info=True)
            self.state = MonitoringState.ERROR
            self._trigger_callbacks('on_error', error=str(e))
        
        logger.debug("Monitoring loop ended")
    
    def _process_frame(self, frame: np.ndarray, timestamp: float):
        """Process a single frame"""
        # Trigger frame callback
        self._trigger_callbacks('on_frame', frame=frame, timestamp=timestamp)
        
        # Detect face landmarks
        face_landmarks = self.mediapipe_processor.process_frame(frame)
        
        if face_landmarks is None:
            # No face detected
            return
        
        self.current_session.successful_detections += 1
        
        # Estimate gaze
        gaze_estimation = self.gaze_estimator.estimate_gaze(
            face_landmarks, 
            frame.shape[:2]
        )
        
        if gaze_estimation is None:
            return
        
        # Update blink count
        if gaze_estimation.is_blinking:
            self.current_session.total_blinks += 1
            self._trigger_callbacks('on_blink', timestamp=timestamp)
        
        # Trigger gaze callback
        self._trigger_callbacks('on_gaze', 
                              gaze=gaze_estimation, 
                              frame=frame)
        
        # Log data
        if self.config.data_logging.enable_logging:
            self._log_gaze_data(gaze_estimation)
        
        # Store in session (sample to avoid memory issues)
        if len(self.current_session.gaze_data) < 1000 or \
           len(self.current_session.gaze_data) % 10 == 0:
            self.current_session.gaze_data.append({
                'timestamp': gaze_estimation.timestamp,
                'screen_x': gaze_estimation.screen_point[0],
                'screen_y': gaze_estimation.screen_point[1],
                'confidence': gaze_estimation.tracking_confidence
            })
        
        # Basic attention analysis (Phase 1)
        self._analyze_attention(gaze_estimation)
    
    def _analyze_attention(self, gaze: GazeEstimation):
        """Basic attention analysis"""
        # Simple attention scoring based on gaze position
        # In Phase 3, this will include sophisticated behavioral analysis
        
        x, y = gaze.screen_point
        
        # Check if looking at screen center (simplified)
        center_distance = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        
        if center_distance > 0.4:  # Looking away from center
            # Update attention score
            self.current_session.attention_score *= 0.99
            
            # Check for suspicious behavior
            if center_distance > 0.6:
                behavior = {
                    'timestamp': gaze.timestamp,
                    'type': 'looking_away',
                    'severity': 'low',
                    'details': {
                        'gaze_position': gaze.screen_point,
                        'distance_from_center': center_distance
                    }
                }
                
                self.current_session.suspicious_behaviors.append(behavior)
                self._trigger_callbacks('on_suspicious_behavior', 
                                      behavior=behavior)
    
    def _init_data_logging(self):
        """Initialize data logging"""
        log_dir = self.config.data_logging.log_directory
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        extension = self.config.data_logging.log_format
        filename = self.config.data_logging.log_filename_pattern.format(
            timestamp=timestamp,
            extension=extension
        )
        
        filepath = log_dir / filename
        
        if self.config.data_logging.log_format == 'csv':
            self.log_file = open(filepath, 'w', newline='')
            self.csv_writer = csv.DictWriter(
                self.log_file,
                fieldnames=self.config.data_logging.log_fields
            )
            self.csv_writer.writeheader()
        else:
            # JSON logging
            self.log_file = open(filepath, 'w')
        
        logger.info(f"Data logging initialized: {filepath}")
    
    def _log_gaze_data(self, gaze: GazeEstimation):
        """Log gaze data to file"""
        if not self.log_file:
            return
        
        data = {
            'timestamp': gaze.timestamp,
            'frame_number': self.current_session.total_frames,
            'gaze_x': gaze.screen_point[0],
            'gaze_y': gaze.screen_point[1],
            'left_pupil_x': gaze.left_gaze.origin[0] if gaze.left_gaze else None,
            'left_pupil_y': gaze.left_gaze.origin[1] if gaze.left_gaze else None,
            'right_pupil_x': gaze.right_gaze.origin[0] if gaze.right_gaze else None,
            'right_pupil_y': gaze.right_gaze.origin[1] if gaze.right_gaze else None,
            'head_pitch': gaze.head_pitch,
            'head_yaw': gaze.head_yaw,
            'head_roll': gaze.head_roll,
            'blink_detected': gaze.is_blinking,
            'attention_score': self.current_session.attention_score,
            'anomaly_flags': ''
        }
        
        if self.config.data_logging.log_format == 'csv':
            self.csv_writer.writerow(data)
        else:
            # JSON format
            json.dump(data, self.log_file)
            self.log_file.write('\n')
    
    def _save_session_summary(self, session: MonitoringSession):
        """Save session summary to file"""
        summary_dir = self.config.data_logging.log_directory / 'summaries'
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = summary_dir / f"{session.session_id}_summary.json"
        
        with open(filepath, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)
        
        logger.info(f"Session summary saved: {filepath}")
    
    def add_callback(self, event: str, callback: Callable):
        """
        Add a callback for specific events.
        
        Events:
        - on_frame: Called for each processed frame
        - on_gaze: Called when gaze is detected
        - on_blink: Called when blink is detected
        - on_suspicious_behavior: Called when suspicious behavior detected
        - on_error: Called on errors
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event: {event}")
    
    def remove_callback(self, event: str, callback: Callable):
        """Remove a callback"""
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
    
    def _trigger_callbacks(self, event: str, **kwargs):
        """Trigger callbacks for an event"""
        for callback in self.callbacks[event]:
            try:
                callback(**kwargs)
            except Exception as e:
                logger.error(f"Error in callback: {e}", exc_info=True)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics"""
        if not self.current_session:
            return {}
        
        return {
            'session_id': self.current_session.session_id,
            'duration': self.current_session.get_duration(),
            'fps': self.current_session.total_frames / max(1, self.current_session.get_duration()),
            'detection_rate': self.current_session.get_detection_rate(),
            'blink_rate': self.current_session.total_blinks / max(1, self.current_session.get_duration() / 60),
            'attention_score': self.current_session.attention_score,
            'suspicious_behaviors': len(self.current_session.suspicious_behaviors),
            'camera_stats': self.camera_manager.get_performance_stats(),
            'mediapipe_stats': self.mediapipe_processor.get_performance_stats()
        }
    
    def calibrate(self, points: int = 9) -> bool:
        """
        Run calibration procedure.
        
        Args:
            points: Number of calibration points
            
        Returns:
            True if calibration successful
        """
        # Placeholder for Phase 1
        # Full calibration will be implemented in later phases
        logger.info("Calibration not yet implemented")
        return True
    
    def close(self):
        """Clean up resources"""
        # Stop monitoring if active
        if self.state == MonitoringState.RUNNING:
            self.stop_monitoring()
        
        # Close components
        self.camera_manager.release()
        self.mediapipe_processor.close()
        
        logger.info("Interview monitor closed")