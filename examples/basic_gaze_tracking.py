"""
Basic Gaze Tracking Example

Demonstrates the basic usage of the eye tracking system.
Shows real-time gaze tracking with visualization.
"""

import cv2
import numpy as np
import logging
import argparse
import sys
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config, PerformanceMode
from api.interview_monitor import InterviewMonitor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GazeTrackingDemo:
    """Simple demo application for gaze tracking"""
    
    def __init__(self, performance_mode: PerformanceMode = PerformanceMode.BALANCED):
        """Initialize demo"""
        # Create config with specified performance mode
        self.config = Config()
        self.config.system.performance_mode = performance_mode
        self.config.system.show_visualization = True
        
        # Initialize monitor
        self.monitor = InterviewMonitor(self.config)
        
        # Visualization window
        self.window_name = "Gaze Tracking Demo"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # State
        self.current_frame = None
        self.current_gaze = None
        self.show_landmarks = False
        self.recording = False
        
        # Performance metrics
        self.fps_timer = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        
        # Register callbacks
        self.monitor.add_callback('on_frame', self.on_frame)
        self.monitor.add_callback('on_gaze', self.on_gaze)
        self.monitor.add_callback('on_blink', self.on_blink)
        self.monitor.add_callback('on_suspicious_behavior', self.on_suspicious_behavior)
    
    def on_frame(self, frame: np.ndarray, timestamp: float):
        """Handle new frame"""
        self.current_frame = frame
        self.fps_counter += 1
        
        # Update FPS
        if time.time() - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = time.time()
    
    def on_gaze(self, gaze, frame: np.ndarray):
        """Handle gaze estimation"""
        self.current_gaze = gaze
    
    def on_blink(self, timestamp: float):
        """Handle blink detection"""
        logger.info(f"Blink detected at {timestamp:.2f}")
    
    def on_suspicious_behavior(self, behavior: dict):
        """Handle suspicious behavior detection"""
        logger.warning(f"Suspicious behavior: {behavior['type']} at {behavior['timestamp']:.2f}")
    
    def draw_interface(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI elements on frame"""
        if frame is None:
            return None
        
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Draw gaze visualization
        if self.current_gaze:
            display_frame = self.monitor.gaze_estimator.draw_gaze(display_frame, self.current_gaze)
        
        # Draw UI elements
        # Title bar
        cv2.rectangle(display_frame, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.putText(display_frame, "Eye Tracking System - Demo", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Performance info
        info_x = width - 250
        cv2.putText(display_frame, f"FPS: {self.current_fps}", (info_x, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Status bar
        status_y = height - 30
        cv2.rectangle(display_frame, (0, status_y - 30), (width, height), (0, 0, 0), -1)
        
        # Current stats
        stats = self.monitor.get_current_stats()
        if stats:
            status_text = f"Detection: {stats.get('detection_rate', 0):.1%} | "
            status_text += f"Blinks/min: {stats.get('blink_rate', 0):.1f} | "
            status_text += f"Attention: {stats.get('attention_score', 1.0):.2f}"
            cv2.putText(display_frame, status_text, (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Recording indicator
        if self.recording:
            cv2.circle(display_frame, (width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (width - 70, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Instructions
        instructions = [
            "Press 'q' to quit",
            "Press 'r' to start/stop recording",
            "Press 'l' to toggle landmarks",
            "Press 's' to save screenshot",
            "Press 'c' to calibrate (not implemented)"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(display_frame, instruction, (10, height - 100 - i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display_frame
    
    def save_screenshot(self):
        """Save current frame as screenshot"""
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, self.current_frame)
            logger.info(f"Screenshot saved: {filename}")
    
    def run(self):
        """Run the demo application"""
        logger.info("Starting gaze tracking demo...")
        logger.info(f"Performance mode: {self.config.system.performance_mode.value}")
        logger.info("Press 'q' to quit")
        
        # Start monitoring
        session_id = self.monitor.start_monitoring()
        logger.info(f"Started session: {session_id}")
        
        try:
            while True:
                # Get and display frame
                if self.current_frame is not None:
                    display_frame = self.draw_interface(self.current_frame)
                    if display_frame is not None:
                        cv2.imshow(self.window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.recording = not self.recording
                    logger.info(f"Recording: {'ON' if self.recording else 'OFF'}")
                elif key == ord('l'):
                    self.show_landmarks = not self.show_landmarks
                    logger.info(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord('s'):
                    self.save_screenshot()
                elif key == ord('c'):
                    logger.info("Calibration not yet implemented")
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            # Stop monitoring
            session = self.monitor.stop_monitoring()
            if session:
                logger.info(f"Session ended: {session.session_id}")
                logger.info(f"Duration: {session.get_duration():.1f} seconds")
                logger.info(f"Total frames: {session.total_frames}")
                logger.info(f"Detection rate: {session.get_detection_rate():.1%}")
                logger.info(f"Total blinks: {session.total_blinks}")
            
            # Clean up
            cv2.destroyAllWindows()
            self.monitor.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Eye Tracking System Demo")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['high_accuracy', 'balanced', 'high_speed', 'ultra_low_latency'],
        default='balanced',
        help='Performance mode'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--resolution',
        type=str,
        default='640x480',
        help='Camera resolution (e.g., 640x480, 1280x720)'
    )
    parser.add_argument(
        '--list-cameras',
        action='store_true',
        help='List available cameras and exit'
    )
    
    args = parser.parse_args()
    
    # List cameras if requested
    if args.list_cameras:
        from utils.camera_manager import CameraManager
        cam = CameraManager()
        cameras = cam.list_available_cameras()
        
        print("\nAvailable cameras:")
        for camera in cameras:
            print(f"  Device {camera.device_id}: {camera.resolution[0]}x{camera.resolution[1]} @ {camera.fps} FPS")
            print(f"    Supported resolutions: {camera.supported_resolutions}")
        
        cam.release()
        return
    
    # Parse resolution
    if 'x' in args.resolution:
        width, height = map(int, args.resolution.split('x'))
    else:
        width, height = 640, 480
    
    # Create and run demo
    mode = PerformanceMode[args.mode.upper()]
    demo = GazeTrackingDemo(performance_mode=mode)
    
    # Override camera settings
    demo.config.camera.device_id = args.camera
    demo.config.camera.width = width
    demo.config.camera.height = height
    
    demo.run()


if __name__ == "__main__":
    main()