"""
Video Processing Example

Process video files for eye tracking and gaze estimation.
"""

import argparse
import sys
from pathlib import Path
import logging
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config, PerformanceMode
from api.interview_monitor import InterviewMonitor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_single_video(video_path: Path, output_path: Path = None, 
                        skip_frames: int = 0, mode: str = 'balanced'):
    """Process a single video file"""
    # Configure system
    config = Config()
    config.system.performance_mode = PerformanceMode[mode.upper()]
    config.system.show_visualization = (output_path is not None)
    
    # Create monitor
    monitor = InterviewMonitor(config)
    
    # Process video
    logger.info(f"Processing video: {video_path}")
    start_time = time.time()
    
    session = monitor.process_video(
        video_path=video_path,
        output_path=output_path,
        skip_frames=skip_frames
    )
    
    processing_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*50)
    print(f"Video Processing Complete")
    print("="*50)
    print(f"Video: {video_path.name}")
    print(f"Session ID: {session.session_id}")
    print(f"Duration: {session.get_duration():.1f}s")
    print(f"Processing Time: {processing_time:.1f}s")
    print(f"Total Frames: {session.total_frames}")
    print(f"Detection Rate: {session.get_detection_rate():.1%}")
    print(f"Total Blinks: {session.total_blinks}")
    print(f"Blinks/min: {session.total_blinks / (session.get_duration() / 60):.1f}")
    print(f"Attention Score: {session.attention_score:.2f}")
    print(f"Suspicious Behaviors: {len(session.suspicious_behaviors)}")
    
    # Save detailed results
    results_path = video_path.parent / f"{video_path.stem}_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(session.to_dict(), f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")
    
    return session


def process_batch(video_dir: Path, output_dir: Path = None, 
                 skip_frames: int = 0, mode: str = 'balanced'):
    """Process all videos in a directory"""
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    if not video_files:
        logger.error(f"No video files found in {video_dir}")
        return
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    # Configure system
    config = Config()
    config.system.performance_mode = PerformanceMode[mode.upper()]
    config.system.show_visualization = (output_dir is not None)
    
    # Create monitor
    monitor = InterviewMonitor(config)
    
    # Process videos
    results = monitor.process_video_batch(
        video_paths=video_files,
        output_dir=output_dir,
        skip_frames=skip_frames
    )
    
    # Summary
    print("\n" + "="*50)
    print("Batch Processing Summary")
    print("="*50)
    
    successful = sum(1 for r in results.values() if r is not None)
    print(f"Videos processed: {successful}/{len(video_files)}")
    
    for video_name, session in results.items():
        if session:
            print(f"\n{video_name}:")
            print(f"  Detection rate: {session.get_detection_rate():.1%}")
            print(f"  Blinks: {session.total_blinks}")
            print(f"  Attention: {session.attention_score:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Process videos for eye tracking")
    parser.add_argument('input', type=str, help='Input video file or directory')
    parser.add_argument('-o', '--output', type=str, help='Output video file or directory')
    parser.add_argument('-s', '--skip', type=int, default=0, 
                       help='Process every Nth frame (0=all frames)')
    parser.add_argument('-m', '--mode', type=str, default='balanced',
                       choices=['high_accuracy', 'balanced', 'high_speed', 'ultra_low_latency'],
                       help='Performance mode')
    parser.add_argument('-b', '--batch', action='store_true',
                       help='Process all videos in directory')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    if args.batch or input_path.is_dir():
        # Batch processing
        if not input_path.is_dir():
            logger.error("Batch mode requires input directory")
            return
        
        process_batch(input_path, output_path, args.skip, args.mode)
    else:
        # Single video
        if not input_path.is_file():
            logger.error(f"Video file not found: {input_path}")
            return
        
        process_single_video(input_path, output_path, args.skip, args.mode)


if __name__ == "__main__":
    main()