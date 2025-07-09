"""
Analyze Interview Videos Script

Main script for analyzing pre-recorded interview videos.
Generates comprehensive reports with gaze patterns and behavioral metrics.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import Config
from api.interview_monitor import InterviewMonitor
import json


def generate_report(session, output_dir: Path):
    """Generate analysis report with visualizations"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert gaze data to DataFrame
    df = pd.DataFrame(session.gaze_data)
    
    if len(df) == 0:
        print("No gaze data available for analysis")
        return
    
    # 1. Gaze heatmap
    plt.figure(figsize=(10, 8))
    plt.hexbin(df['screen_x'], df['screen_y'], gridsize=10, cmap='hot')
    plt.colorbar(label='Fixation Count')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().invert_yaxis()
    plt.title('Gaze Heatmap')
    plt.xlabel('Screen X (normalized)')
    plt.ylabel('Screen Y (normalized)')
    plt.savefig(output_dir / f"{session.session_id}_heatmap.png")
    plt.close()
    
    # 2. Attention over time
    df['timestamp_min'] = df['timestamp'] / 60
    window_size = max(1, len(df) // 100)
    
    # Calculate attention score (distance from center)
    df['center_distance'] = ((df['screen_x'] - 0.5)**2 + (df['screen_y'] - 0.5)**2)**0.5
    df['attention'] = 1 - df['center_distance'].rolling(window_size).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp_min'], df['attention'])
    plt.fill_between(df['timestamp_min'], df['attention'], alpha=0.3)
    plt.ylim(0, 1)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Attention Score')
    plt.title('Attention Score Over Time')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"{session.session_id}_attention.png")
    plt.close()
    
    # 3. Generate text report
    report_path = output_dir / f"{session.session_id}_report.txt"
    with open(report_path, 'w') as f:
        f.write("INTERVIEW VIDEO ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Session ID: {session.session_id}\n")
        f.write(f"Video: {session.video_info.filepath.name if session.video_info else 'N/A'}\n")
        f.write(f"Duration: {session.get_duration():.1f} seconds\n")
        f.write(f"Frames Analyzed: {session.total_frames}\n")
        f.write(f"Detection Rate: {session.get_detection_rate():.1%}\n\n")
        
        f.write("BEHAVIORAL METRICS\n")
        f.write("-"*30 + "\n")
        f.write(f"Total Blinks: {session.total_blinks}\n")
        f.write(f"Blink Rate: {session.total_blinks / (session.get_duration() / 60):.1f} per minute\n")
        f.write(f"Average Attention Score: {df['attention'].mean():.2f}\n")
        f.write(f"Suspicious Behaviors Detected: {len(session.suspicious_behaviors)}\n\n")
        
        if session.suspicious_behaviors:
            f.write("SUSPICIOUS BEHAVIORS\n")
            f.write("-"*30 + "\n")
            for i, behavior in enumerate(session.suspicious_behaviors, 1):
                f.write(f"{i}. {behavior['type']} at {behavior['timestamp']:.1f}s\n")
                f.write(f"   Severity: {behavior['severity']}\n")
    
    print(f"Report generated: {report_path}")
    
    # 4. Save raw data
    df.to_csv(output_dir / f"{session.session_id}_data.csv", index=False)
    
    # 5. Save session summary as JSON
    with open(output_dir / f"{session.session_id}_summary.json", 'w') as f:
        json.dump(session.to_dict(), f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze interview videos for eye tracking patterns"
    )
    parser.add_argument('video', type=str, help='Video file to analyze')
    parser.add_argument('-o', '--output-dir', type=str, default='./analysis_results',
                       help='Output directory for results')
    parser.add_argument('-s', '--skip-frames', type=int, default=2,
                       help='Process every Nth frame (default: 2)')
    parser.add_argument('--annotate', action='store_true',
                       help='Create annotated output video')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip report generation')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    output_dir = Path(args.output_dir)
    
    # Configure for video processing
    config = Config()
    config.system.performance_mode = "balanced"
    config.system.show_visualization = args.annotate
    config.data_logging.enable_logging = True
    
    # Create monitor
    monitor = InterviewMonitor(config)
    
    # Process video
    print(f"Analyzing video: {video_path}")
    print(f"Skip frames: {args.skip_frames}")
    print("-" * 50)
    
    output_video = None
    if args.annotate:
        output_video = output_dir / f"{video_path.stem}_annotated{video_path.suffix}"
    
    session = monitor.process_video(
        video_path=video_path,
        output_path=output_video,
        skip_frames=args.skip_frames
    )
    
    # Generate report
    if not args.no_report:
        generate_report(session, output_dir)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()