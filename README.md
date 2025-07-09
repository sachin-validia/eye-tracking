# Eye Tracking System

A production-ready eye tracking system for interview monitoring and behavioral analysis. Built with MediaPipe, OpenCV, and GazeTracking for accurate gaze estimation. **Now supports both real-time and video file processing.**

## Features (Phase 1 - Current Implementation)

- ✅ Real-time face detection using MediaPipe Face Mesh (468 landmarks)
- ✅ Basic gaze estimation with 3D head pose tracking
- ✅ Configurable performance modes (High Accuracy, Balanced, High Speed, Ultra Low Latency)
- ✅ Cross-platform support (Windows, Linux, macOS)
- ✅ Modular architecture for easy integration
- ✅ CSV/JSON data logging
- ✅ Real-time visualization
- ✅ Blink detection
- ✅ Basic attention monitoring

## System Requirements

- Python 3.8 or higher
- Webcam (built-in or USB)
- 4GB RAM minimum (8GB recommended)
- CPU: Dual-core 2.0GHz minimum (Quad-core recommended)

## Installation

### WSL (Windows Subsystem for Linux) Setup

1. **Install WSL dependencies:**
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3.8 python3-pip python3-venv

# Install system dependencies for OpenCV and dlib
sudo apt install cmake build-essential
sudo apt install libopencv-dev python3-opencv
sudo apt install libboost-all-dev
sudo apt install libx11-dev libgtk-3-dev

# For GUI support in WSL
sudo apt install x11-apps
```

2. **Configure WSL for GUI applications:**
```bash
# Add to ~/.bashrc
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
export LIBGL_ALWAYS_INDIRECT=1
```

3. **Install VcXsrv on Windows:**
   - Download from: https://sourceforge.net/projects/vcxsrv/
   - Run XLaunch with "Disable access control" checked

### General Installation (All Platforms)

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/eye-tracking-system.git
cd eye-tracking-system
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Install dlib (may take time):**
```bash
# For faster installation with pre-built wheel (if available)
pip install dlib

# If the above fails, build from source
pip install cmake
pip install dlib --no-cache-dir
```

5. **Install GazeTracking library:**
```bash
pip install git+https://github.com/antoinelame/GazeTracking.git
```

6. **Install the package:**
```bash
pip install -e .
```

## Quick Start

## Quick Start - Video Processing

### Process Interview Videos
```bash
# Basic video analysis
python analyze_interview_videos.py interview_video.mp4

# With annotated output video
python analyze_interview_videos.py interview_video.mp4 --annotate

# Process every 5th frame (faster)
python analyze_interview_videos.py interview_video.mp4 --skip-frames 4
```

### Batch Process Videos
```bash
# Process all videos in a directory
python examples/video_processing.py ./videos/ --batch -o ./results/
```

See [VIDEO_PROCESSING_GUIDE.md](VIDEO_PROCESSING_GUIDE.md) for detailed usage.

## Configuration

### Performance Modes

Edit `config/settings.py` or set via code:

```python
from eye_tracking_system import Config, PerformanceMode

config = Config()
config.system.performance_mode = PerformanceMode.HIGH_ACCURACY  # Most accurate
# or
config.system.performance_mode = PerformanceMode.HIGH_SPEED     # Fastest
```

### Environment Variables

Override settings using environment variables:

```bash
export ETS_SYSTEM_PERFORMANCE_MODE=high_speed
export ETS_CAMERA_WIDTH=1280
export ETS_CAMERA_HEIGHT=720
export ETS_DATA_LOGGING_ENABLE_LOGGING=true
```

### Configuration File

Create `config.yaml`:

```yaml
system:
  performance_mode: balanced
  debug_mode: false
  
camera:
  width: 640
  height: 480
  fps: 30
  
mediapipe:
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
  
data_logging:
  enable_logging: true
  log_format: csv
```

## Testing Instructions

### Unit Tests (Coming in Phase 2)

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=eye_tracking_system tests/

# Run specific test
pytest tests/test_mediapipe.py
```

### Manual Testing Checklist

1. **Camera Detection:**
   - Run `python examples/basic_gaze_tracking.py --list-cameras`
   - Verify your camera is detected

2. **Basic Tracking:**
   - Run the demo
   - Move your head slowly
   - Verify green dots appear on your eyes
   - Check FPS counter (should be 25-30 for balanced mode)

3. **Blink Detection:**
   - Blink naturally
   - Check console for "Blink detected" messages

4. **Performance Modes:**
   - Test each mode: `--mode high_accuracy`, `--mode high_speed`
   - Compare FPS and tracking quality

5. **Data Logging:**
   - Run demo with recording (press 'r')
   - Check `logs/` directory for CSV files
   - Verify data completeness

## Current Shortcomings (Phase 1)

### Technical Limitations

1. **Basic Gaze Estimation:**
   - Currently uses simplified geometric method
   - No actual pupil detection (uses eye landmarks approximation)
   - Accuracy: ~5-10° error (target: 1-2°)

2. **No Calibration:**
   - System uses default camera parameters
   - No user-specific calibration implemented
   - Screen mapping is approximate

3. **Limited Behavioral Analysis:**
   - Only basic "looking away" detection
   - No sophisticated pattern recognition
   - No ML-based anomaly detection

4. **Performance:**
   - GazeTracking integration not optimized
   - May struggle on low-end hardware
   - No GPU acceleration

5. **Head Pose Limitations:**
   - Works best with frontal face
   - Accuracy degrades at extreme angles (>45°)
   - No compensation for glasses/reflections

### Integration Limitations

1. **No Real Platform Integration:**
   - No Zoom/Teams/WebRTC integration
   - No cloud deployment ready
   - Basic API only

2. **Limited Data Export:**
   - Only CSV/JSON formats
   - No real-time streaming
   - No dashboard/analytics

## Nice-to-Haves for Future Phases

### Phase 2 Enhancements
1. **Improved Accuracy:**
   - Integrate actual GazeTracking pupil detection
   - Implement full 3D eye model
   - Add iris landmark detection
   - User-specific calibration system

2. **Advanced Head Pose:**
   - Full 6DOF head tracking
   - Glasses detection and compensation
   - Multiple face support

3. **Better Performance:**
   - GPU acceleration with CUDA
   - Optimized GazeTracking integration
   - Frame dropping for consistent FPS

### Phase 3 Features
1. **Behavioral Analysis:**
   - ML-based suspicious behavior detection
   - Pattern learning from historical data
   - Attention heatmaps
   - Cognitive load estimation

2. **Advanced Metrics:**
   - Saccade detection and analysis
   - Fixation duration
   - Scan path analysis
   - Pupil dilation tracking

### Phase 4 Integration
1. **Platform Integration:**
   - Zoom SDK integration
   - Browser-based WebRTC support
   - Screen recording correlation
   - Multi-camera support

2. **Production Features:**
   - Cloud deployment ready
   - Real-time dashboard
   - Alert system
   - Compliance/privacy features

## Troubleshooting

### WSL-Specific Issues

1. **Camera not detected:**
```bash
# Install USB/IP support
sudo apt install linux-tools-generic hwdata
sudo update-alternatives --install /usr/local/bin/usbip usbip /usr/lib/linux-tools/*-generic/usbip 20

# On Windows (Admin PowerShell):
usbipd wsl list
usbipd wsl attach --busid <BUSID>
```

2. **GUI not showing:**
- Ensure VcXsrv is running
- Check DISPLAY variable: `echo $DISPLAY`
- Try: `export DISPLAY=:0`

3. **Slow performance:**
- Use native Linux instead of WSL for production
- Reduce camera resolution
- Use `high_speed` mode

### Common Issues

1. **ImportError for dlib:**
```bash
# Install build tools
sudo apt install cmake build-essential
pip install cmake
pip install dlib --no-cache-dir
```

2. **MediaPipe errors:**
```bash
# Ensure correct version
pip install mediapipe==0.10.21
```

3. **Low FPS:**
- Check CPU usage
- Reduce resolution
- Disable debug visualization
- Use performance mode

