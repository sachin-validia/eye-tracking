"""
Interview Monitor API

Main API interface for the eye tracking system.
Supports both real-time and video file processing.
"""

import logging
import time
import threading
from typing import Optional, Dict, Any, List, Callable, Union
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
from utils.video_processor import VideoProcessor, VideoInfo


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
    video_info: Optional[VideoInfo] = None  # For video processing
    
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
    
    Supports both video file processing and real-time monitoring.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize interview monitor"""
        self.config = config or get_config()
        
        # Core components
        self.video_processor = VideoProcessor(self.config)
        self.mediapipe_processor = MediaPipeProcessor(self.config)
        self.gaze_estimator = GazeEstimator(self.config)
        
        # State management
        self.state = MonitoringState.IDLE
        self.current_session: Optional[MonitoringSession] = None
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'on_state_change': [],
            'on_detection': [],
            'on_blink': [],
            'on_attention_change': [],
            'on_suspicious_behavior': []
        }
        
        # Data logging
        self.log_file: Optional[Any] = None
        self.csv_writer: Optional[csv.DictWriter] = None
        
        logger.info("Interview monitor initialized for video processing")
    
    def process_video(self, video_path: Union[str, Path], 
                     output_path: Optional[Path] = None,
                     skip_frames: int = 0,
                     session_id: Optional[str] = None) -> MonitoringSession:
        """
        Process a video file for gaze tracking analysis.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path for output video with annotations
            skip_frames: Process every Nth frame (0 = process all)
            session_id: Optional session identifier
            
        Returns:
            MonitoringSession with results
        """
        video_path = Path(video_path)
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"video_{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create session
        self.current_session = MonitoringSession(
            session_id=session_id,
            start_time=time.time()
        )
        
        # Initialize data logging
        if self.config.data_logging.enable_logging:
            self._init_data_logging()
        
        try:
            # Load video
            video_info = self.video_processor.load_video(video_path)
            self.current_session.video_info = video_info
            
            logger.info(f"Processing video: {video_info}")
            
            # Create output video writer if requested
            video_writer = None
            if output_path:
                video_writer = self.video_processor.create_output_video(output_path)
            
            # Process frames
            for frame, frame_idx, timestamp in self.video_processor.process_frames(skip_frames):
                self.current_session.total_frames += 1
                
                # Process frame
                processed_frame = self._process_frame(frame, timestamp)
                
                # Write annotated frame if output requested
                if video_writer and processed_frame is not None:
                    video_writer.write(processed_frame)
                
                # Log progress periodically
                if frame_idx % 100 == 0:
                    progress = self.video_processor.get_progress()
                    logger.info(f"Progress: {progress:.1%} ({frame_idx}/{video_info.total_frames})")
            
            # Cleanup
            if video_writer:
                video_writer.release()
            
        finally:
            # Finalize session
            self.current_session.end_time = time.time()
            self._save_session_summary(self.current_session)
            
            # Close logging
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            
            self.video_processor.release()
        
        logger.info(f"Video processing complete: {self.current_session.session_id}")
        return self.current_session
    
    def process_video_batch(self, video_paths: List[Union[str, Path]], 
                          output_dir: Optional[Path] = None,
                          skip_frames: int = 0) -> Dict[str, MonitoringSession]:
        """
        Process multiple videos in batch.
        
        Args:
            video_paths: List of video file paths
            output_dir: Directory for output videos (optional)
            skip_frames: Process every Nth frame
            
        Returns:
            Dictionary mapping video names to sessions
        """
        results = {}
        
        for video_path in video_paths:
            video_path = Path(video_path)
            output_path = None
            
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{video_path.stem}_tracked{video_path.suffix}"
            
            try:
                session = self.process_video(video_path, output_path, skip_frames)
                results[video_path.name] = session
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                results[video_path.name] = None
        
        return results
    
    def _process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        """Process a single frame and return annotated version"""
        # Detect face landmarks
        face_landmarks = self.mediapipe_processor.process_frame(frame)
        
        if face_landmarks is None:
            return frame
        
        self.current_session.successful_detections += 1
        
        # Estimate gaze
        gaze_estimation = self.gaze_estimator.estimate_gaze(
            face_landmarks, 
            frame.shape[:2]
        )
        
        if gaze_estimation is None:
            return frame
        
        # Update blink count
        if gaze_estimation.is_blinking:
            self.current_session.total_blinks += 1
        
        # Log data
        if self.config.data_logging.enable_logging:
            self._log_gaze_data(gaze_estimation)
        
        # Store sample data
        if len(self.current_session.gaze_data) < 10000:  # Limit memory usage
            self.current_session.gaze_data.append({
                'timestamp': timestamp,
                'screen_x': gaze_estimation.screen_point[0],
                'screen_y': gaze_estimation.screen_point[1],
                'confidence': gaze_estimation.tracking_confidence
            })
        
        # Analyze attention
        self._analyze_attention(gaze_estimation)
        
        # Draw visualization if enabled
        if self.config.system.show_visualization:
            annotated_frame = self.gaze_estimator.draw_gaze(frame, gaze_estimation)
            return annotated_frame
        
        return frame
    
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
            self.state = MonitoringState.STOPPED
        
        # Close components
        self.mediapipe_processor.close()
        
        logger.info("Interview monitor closed")