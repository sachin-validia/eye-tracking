"""
Video Processor Module

Handles video file input, frame extraction, and batch processing.
Replaces real-time camera capture with video file processing.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Generator, Tuple, List, Dict, Any
from dataclasses import dataclass
import time

from config.settings import Config, get_config


logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Video file information"""
    filepath: Path
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str
    
    def __str__(self):
        return (f"Video: {self.filepath.name}\n"
                f"Resolution: {self.width}x{self.height}\n"
                f"FPS: {self.fps:.2f}\n"
                f"Duration: {self.duration:.2f}s\n"
                f"Frames: {self.total_frames}")


class VideoProcessor:
    """
    Processes video files for eye tracking analysis.
    
    Features:
    - Frame extraction from video files
    - Batch processing support
    - Progress tracking
    - Frame skipping for performance
    - Multiple video format support
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize video processor"""
        self.config = config or get_config()
        self.current_video: Optional[cv2.VideoCapture] = None
        self.video_info: Optional[VideoInfo] = None
        self.current_frame_index = 0
        
        # Supported formats
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v']
        
        logger.info("Video processor initialized")
    
    def load_video(self, filepath: Path) -> VideoInfo:
        """
        Load a video file for processing.
        
        Args:
            filepath: Path to video file
            
        Returns:
            VideoInfo object with video metadata
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Video file not found: {filepath}")
        
        if filepath.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {filepath.suffix}. "
                           f"Supported: {self.supported_formats}")
        
        # Release previous video if any
        if self.current_video:
            self.current_video.release()
        
        # Open video
        self.current_video = cv2.VideoCapture(str(filepath))
        if not self.current_video.isOpened():
            raise RuntimeError(f"Cannot open video: {filepath}")
        
        # Get video properties
        width = int(self.current_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.current_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.current_video.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.current_video.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = int(self.current_video.get(cv2.CAP_PROP_FOURCC))
        
        # Calculate duration
        duration = total_frames / fps if fps > 0 else 0
        
        # Convert codec to string
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        
        self.video_info = VideoInfo(
            filepath=filepath,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            codec=codec_str
        )
        
        self.current_frame_index = 0
        
        logger.info(f"Loaded video: {filepath.name} "
                   f"({width}x{height}, {fps:.1f} FPS, {total_frames} frames)")
        
        return self.video_info
    
    def process_frames(self, skip_frames: int = 0, 
                      max_frames: Optional[int] = None) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Generator that yields frames from the video.
        
        Args:
            skip_frames: Process every (skip_frames + 1)th frame
            max_frames: Maximum number of frames to process
            
        Yields:
            Tuple of (frame, frame_index, timestamp)
        """
        if not self.current_video:
            raise RuntimeError("No video loaded")
        
        frames_processed = 0
        
        while True:
            ret, frame = self.current_video.read()
            if not ret:
                break
            
            # Skip frames if requested
            if skip_frames > 0 and self.current_frame_index % (skip_frames + 1) != 0:
                self.current_frame_index += 1
                continue
            
            # Calculate timestamp
            timestamp = self.current_frame_index / self.video_info.fps
            
            yield frame, self.current_frame_index, timestamp
            
            self.current_frame_index += 1
            frames_processed += 1
            
            # Check max frames limit
            if max_frames and frames_processed >= max_frames:
                break
        
        logger.info(f"Processed {frames_processed} frames")
    
    def seek_to_frame(self, frame_index: int):
        """Seek to specific frame index"""
        if not self.current_video:
            raise RuntimeError("No video loaded")
        
        self.current_video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.current_frame_index = frame_index
    
    def seek_to_time(self, timestamp: float):
        """Seek to specific timestamp in seconds"""
        if not self.video_info:
            raise RuntimeError("No video loaded")
        
        frame_index = int(timestamp * self.video_info.fps)
        self.seek_to_frame(frame_index)
    
    def extract_frame_batch(self, frame_indices: List[int]) -> List[Tuple[np.ndarray, int]]:
        """
        Extract specific frames by index.
        
        Args:
            frame_indices: List of frame indices to extract
            
        Returns:
            List of (frame, index) tuples
        """
        frames = []
        
        for idx in frame_indices:
            self.seek_to_frame(idx)
            ret, frame = self.current_video.read()
            if ret:
                frames.append((frame, idx))
            else:
                logger.warning(f"Failed to read frame {idx}")
        
        return frames
    
    def get_progress(self) -> float:
        """Get processing progress (0-1)"""
        if not self.video_info:
            return 0.0
        return self.current_frame_index / self.video_info.total_frames
    
    def create_output_video(self, output_path: Path, fps: Optional[float] = None,
                           codec: str = 'mp4v') -> cv2.VideoWriter:
        """
        Create video writer for output.
        
        Args:
            output_path: Path for output video
            fps: Output FPS (uses input FPS if None)
            codec: Video codec (default: mp4v)
            
        Returns:
            VideoWriter object
        """
        if not self.video_info:
            raise RuntimeError("No video loaded")
        
        fps = fps or self.video_info.fps
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (self.video_info.width, self.video_info.height)
        )
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create output video: {output_path}")
        
        return writer
    
    def release(self):
        """Release video resources"""
        if self.current_video:
            self.current_video.release()
            self.current_video = None
        self.video_info = None
        self.current_frame_index = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def process_video_batch(video_paths: List[Path], 
                       processor_func: callable,
                       output_dir: Path,
                       skip_frames: int = 0) -> Dict[str, Any]:
    """
    Process multiple videos with the same processing function.
    
    Args:
        video_paths: List of video file paths
        processor_func: Function to process each frame
        output_dir: Directory for output files
        skip_frames: Process every Nth frame
        
    Returns:
        Dictionary with processing results
    """
    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for video_path in video_paths:
        logger.info(f"Processing {video_path.name}...")
        
        with VideoProcessor() as vp:
            info = vp.load_video(video_path)
            
            # Process frames
            frame_results = []
            for frame, idx, timestamp in vp.process_frames(skip_frames):
                result = processor_func(frame, idx, timestamp)
                frame_results.append(result)
            
            results[video_path.name] = {
                'info': info,
                'results': frame_results
            }
    
    return results