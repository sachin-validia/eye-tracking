"""
Eye Tracking System Configuration Settings

This module contains all configurable parameters for the eye tracking system.
Settings can be overridden via environment variables or config files.

Configuration Hierarchy:
1. Default values (defined here)
2. Environment variables (prefixed with ETS_)
3. Config file (config.yaml if exists)
4. Runtime parameters (passed to constructors)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging


class PerformanceMode(Enum):
    """Performance mode presets for different use cases"""
    HIGH_ACCURACY = "high_accuracy"  # Slower but more accurate
    BALANCED = "balanced"            # Default mode
    HIGH_SPEED = "high_speed"       # Faster processing, reduced accuracy
    ULTRA_LOW_LATENCY = "ultra_low_latency"  # Minimum latency for real-time


class LogLevel(Enum):
    """Logging verbosity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class CameraConfig:
    """Camera capture configuration"""
    device_id: int = 0  # 0 for default webcam
    width: int = 640    # Target resolution width
    height: int = 480   # Target resolution height
    fps: int = 30       # Target FPS
    buffer_size: int = 1  # Frame buffer size
    auto_exposure: bool = True
    brightness: Optional[float] = None  # 0.0-1.0, None for auto
    contrast: Optional[float] = None    # 0.0-1.0, None for auto
    
    # Fallback resolutions if target not supported
    fallback_resolutions: list = field(default_factory=lambda: [
        (1280, 720),
        (640, 480),
        (320, 240)
    ])


@dataclass
class MediaPipeConfig:
    """MediaPipe Face Mesh configuration"""
    static_image_mode: bool = False
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Performance mode specific overrides
    performance_overrides: Dict[PerformanceMode, Dict[str, Any]] = field(default_factory=lambda: {
        PerformanceMode.HIGH_ACCURACY: {
            "min_detection_confidence": 0.7,
            "min_tracking_confidence": 0.7,
            "refine_landmarks": True
        },
        PerformanceMode.HIGH_SPEED: {
            "min_detection_confidence": 0.3,
            "min_tracking_confidence": 0.3,
            "refine_landmarks": False
        },
        PerformanceMode.ULTRA_LOW_LATENCY: {
            "min_detection_confidence": 0.2,
            "min_tracking_confidence": 0.2,
            "refine_landmarks": False,
            "static_image_mode": False
        }
    })


@dataclass
class GazeEstimationConfig:
    """Gaze estimation algorithm configuration"""
    use_3d_model: bool = True
    use_kalman_filter: bool = True
    kalman_q: float = 0.004  # Process noise
    kalman_r: float = 0.5    # Measurement noise
    
    # Eye region landmarks (MediaPipe indices)
    left_eye_indices: list = field(default_factory=lambda: list(range(33, 42)))
    right_eye_indices: list = field(default_factory=lambda: list(range(362, 374)))
    
    # Smoothing parameters
    smoothing_window: int = 5
    outlier_threshold: float = 2.0  # Standard deviations
    
    # Blink detection
    blink_threshold: float = 0.2
    min_blink_duration: float = 0.1  # seconds
    max_blink_duration: float = 0.4  # seconds


@dataclass
class BehaviorAnalysisConfig:
    """Behavior analysis and anomaly detection configuration"""
    enable_analysis: bool = True
    
    # Attention metrics
    attention_timeout: float = 2.0  # seconds looking away
    focus_region_size: float = 0.2  # normalized screen coordinates
    
    # Suspicious behavior thresholds
    rapid_movement_threshold: float = 0.5  # normalized units/second
    excessive_blinking_rate: float = 30  # blinks per minute
    minimal_blinking_rate: float = 5     # blinks per minute
    
    # Pattern detection windows
    short_term_window: float = 5.0   # seconds
    long_term_window: float = 60.0   # seconds


@dataclass
class DataLoggingConfig:
    """Data logging and storage configuration"""
    enable_logging: bool = True
    log_format: str = "csv"  # csv, json, hdf5
    
    # File paths
    log_directory: Path = Path("./logs")
    log_filename_pattern: str = "gaze_tracking_{timestamp}.{extension}"
    
    # Logging frequency
    log_frequency: float = 10.0  # Hz (logs per second)
    buffer_size: int = 100       # Buffer before writing to disk
    
    # Data fields to log
    log_fields: list = field(default_factory=lambda: [
        "timestamp",
        "frame_number",
        "gaze_x",
        "gaze_y",
        "left_pupil_x",
        "left_pupil_y",
        "right_pupil_x", 
        "right_pupil_y",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "blink_detected",
        "attention_score",
        "anomaly_flags"
    ])
    
    # Compression
    enable_compression: bool = True
    compression_level: int = 6  # 1-9


@dataclass
class SystemConfig:
    """Overall system configuration"""
    # Performance
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    num_threads: int = 0  # 0 for auto-detect
    enable_gpu: bool = True
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    console_log: bool = True
    file_log: bool = True
    log_file_path: Path = Path("./logs/system.log")
    
    # Monitoring
    enable_performance_monitoring: bool = True
    performance_log_interval: float = 5.0  # seconds
    
    # Development
    debug_mode: bool = False
    show_visualization: bool = True
    save_debug_frames: bool = False
    debug_frame_directory: Path = Path("./debug_frames")


class Config:
    """Main configuration class that combines all config sections"""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.camera = CameraConfig()
        self.mediapipe = MediaPipeConfig()
        self.gaze_estimation = GazeEstimationConfig()
        self.behavior_analysis = BehaviorAnalysisConfig()
        self.data_logging = DataLoggingConfig()
        self.system = SystemConfig()
        
        # Load from file if provided
        if config_file and config_file.exists():
            self.load_from_file(config_file)
        
        # Override with environment variables
        self.load_from_env()
        
        # Apply performance mode presets
        self.apply_performance_mode()
        
        # Create necessary directories
        self._create_directories()
    
    def load_from_file(self, config_file: Path):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
                
            # Update each section
            for section_name, section_data in data.items():
                if hasattr(self, section_name) and isinstance(section_data, dict):
                    section = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            # Handle enums
                            if key == 'performance_mode':
                                value = PerformanceMode(value)
                            elif key == 'log_level':
                                value = LogLevel(value)
                            setattr(section, key, value)
                            
        except Exception as e:
            logging.warning(f"Failed to load config file: {e}")
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        # Format: ETS_SECTION_PARAMETER
        for key, value in os.environ.items():
            if key.startswith('ETS_'):
                parts = key[4:].lower().split('_', 2)
                if len(parts) >= 2:
                    section_name = parts[0]
                    param_name = '_'.join(parts[1:])
                    
                    if hasattr(self, section_name):
                        section = getattr(self, section_name)
                        if hasattr(section, param_name):
                            # Convert string to appropriate type
                            current_value = getattr(section, param_name)
                            if isinstance(current_value, bool):
                                value = value.lower() in ('true', '1', 'yes')
                            elif isinstance(current_value, int):
                                value = int(value)
                            elif isinstance(current_value, float):
                                value = float(value)
                            elif isinstance(current_value, Path):
                                value = Path(value)
                            
                            setattr(section, param_name, value)
    
    def apply_performance_mode(self):
        """Apply performance mode presets to MediaPipe config"""
        mode = self.system.performance_mode
        if mode in self.mediapipe.performance_overrides:
            overrides = self.mediapipe.performance_overrides[mode]
            for key, value in overrides.items():
                setattr(self.mediapipe, key, value)
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.data_logging.log_directory,
            self.system.log_file_path.parent,
        ]
        
        if self.system.save_debug_frames:
            directories.append(self.system.debug_frame_directory)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'camera': asdict(self.camera),
            'mediapipe': {k: v for k, v in asdict(self.mediapipe).items() 
                         if k != 'performance_overrides'},
            'gaze_estimation': asdict(self.gaze_estimation),
            'behavior_analysis': asdict(self.behavior_analysis),
            'data_logging': {k: str(v) if isinstance(v, Path) else v 
                           for k, v in asdict(self.data_logging).items()},
            'system': {k: str(v) if isinstance(v, Path) else v.value if isinstance(v, Enum) else v
                      for k, v in asdict(self.system).items()}
        }
    
    def save_to_file(self, config_file: Path):
        """Save current configuration to YAML file"""
        with open(config_file, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get expected performance characteristics for current mode"""
        mode = self.system.performance_mode
        return {
            'mode': mode.value,
            'expected_fps': {
                PerformanceMode.HIGH_ACCURACY: 15-20,
                PerformanceMode.BALANCED: 25-30,
                PerformanceMode.HIGH_SPEED: 30-40,
                PerformanceMode.ULTRA_LOW_LATENCY: 40-60
            }.get(mode, 30),
            'accuracy_level': {
                PerformanceMode.HIGH_ACCURACY: 'high',
                PerformanceMode.BALANCED: 'medium',
                PerformanceMode.HIGH_SPEED: 'low',
                PerformanceMode.ULTRA_LOW_LATENCY: 'minimal'
            }.get(mode, 'medium')
        }


# Singleton instance
_config: Optional[Config] = None


def get_config(config_file: Optional[Path] = None) -> Config:
    """Get or create configuration instance"""
    global _config
    if _config is None:
        _config = Config(config_file)
    return _config


def reset_config():
    """Reset configuration (useful for testing)"""
    global _config
    _config = None