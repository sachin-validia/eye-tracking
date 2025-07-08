# Testing Guide - Eye Tracking System

## Quick Installation Test

After installation, run:
```bash
python test_installation.py
```

This will verify all dependencies and hardware are working correctly.

## Performance Testing

### 1. FPS Benchmarking

Test different performance modes:

```bash
# Test each mode for 30 seconds
python examples/basic_gaze_tracking.py --mode high_accuracy
python examples/basic_gaze_tracking.py --mode balanced
python examples/basic_gaze_tracking.py --mode high_speed
python examples/basic_gaze_tracking.py --mode ultra_low_latency
```

Expected FPS ranges:
- High Accuracy: 15-20 FPS
- Balanced: 25-30 FPS  
- High Speed: 30-40 FPS
- Ultra Low Latency: 40-60 FPS

### 2. Resolution Testing

Test different camera resolutions:

```bash
# Low resolution (fast)
python examples/basic_gaze_tracking.py --resolution 320x240

# Medium resolution (balanced)
python examples/basic_gaze_tracking.py --resolution 640x480

# High resolution (slower but more accurate)
python examples/basic_gaze_tracking.py --resolution 1280x720
```

## Functional Testing

### 1. Eye Tracking Accuracy

1. Run the demo: `python examples/basic_gaze_tracking.py`
2. Position yourself 50-70cm from the camera
3. Look at each corner of the screen
4. Verify the green crosshair follows your gaze (approximately)

### 2. Blink Detection

1. Blink naturally while running the demo
2. Check console output for "Blink detected" messages
3. Verify blink rate in status bar (normal: 15-20 blinks/min)

### 3. Head Pose Tracking

1. Keep eyes on center of screen
2. Slowly rotate head left/right (yaw)
3. Tilt head up/down (pitch)
4. Tilt head side to side (roll)
5. Check head pose values in bottom of screen

### 4. Data Logging

1. Start demo and press 'r' to begin recording
2. Perform various movements for 1 minute
3. Press 'r' again to stop
4. Check `logs/` directory for CSV file
5. Verify data completeness:

```python
import pandas as pd
df = pd.read_csv('logs/gaze_tracking_[timestamp].csv')
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values: {df.isnull().sum().sum()}")
```

## WSL-Specific Testing

### 1. Camera Access

```bash
# List cameras
ls -la /dev/video*

# Test with v4l2
sudo apt install v4l-utils
v4l2-ctl --list-devices

# Test camera directly
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL')"
```

### 2. GUI Display

```bash
# Test X11 forwarding
xclock  # Should show a clock

# Check display
echo $DISPLAY  # Should show something like :0 or 192.168.x.x:0

# Test OpenCV window
python -c "import cv2; import numpy as np; cv2.imshow('test', np.zeros((100,100,3), np.uint8)); cv2.waitKey(1000); cv2.destroyAllWindows()"
```

## Integration Testing

### 1. API Test

Create `test_api.py`:

```python
import time
from eye_tracking_system import InterviewMonitor

# Initialize
monitor = InterviewMonitor()

# Add callbacks
def on_blink(timestamp):
    print(f"Blink at {timestamp}")

def on_suspicious(behavior):
    print(f"Suspicious: {behavior}")

monitor.add_callback('on_blink', on_blink)
monitor.add_callback('on_suspicious_behavior', on_suspicious)

# Run for 10 seconds
session_id = monitor.start_monitoring()
time.sleep(10)
session = monitor.stop_monitoring()

# Print results
print(f"Session: {session_id}")
print(f"Duration: {session.get_duration():.1f}s")
print(f"Detection rate: {session.get_detection_rate():.1%}")
print(f"Total blinks: {session.total_blinks}")
```

### 2. Configuration Test

Create custom config and test:

```python
from eye_tracking_system import Config, InterviewMonitor

# Custom config
config = Config()
config.camera.width = 1280
config.camera.height = 720
config.system.performance_mode = "high_accuracy"
config.data_logging.enable_logging = True
config.data_logging.log_format = "json"

# Save config
config.save_to_file("my_config.yaml")

# Load and use
config2 = Config("my_config.yaml")
monitor = InterviewMonitor(config2)
```

## Stress Testing

### 1. Long Duration Test

Run for extended period:

```bash
# Run for 1 hour
timeout 3600 python examples/basic_gaze_tracking.py
```

Monitor:
- Memory usage (should be stable)
- CPU usage (should be consistent)
- FPS degradation (should be minimal)

### 2. Rapid Movement Test

1. Run demo
2. Move head rapidly
3. Blink frequently
4. Verify system doesn't crash
5. Check for error messages

## Known Issues & Workarounds

### WSL Issues

1. **Slow performance**: Use native Linux or Windows
2. **Camera not found**: Use USB passthrough or native Windows
3. **GUI issues**: Ensure VcXsrv is configured correctly

### Performance Issues

1. **Low FPS**: 
   - Reduce resolution
   - Use faster performance mode
   - Close other applications
   
2. **High CPU usage**:
   - Check for memory leaks
   - Reduce logging frequency
   - Disable debug visualization

### Accuracy Issues

1. **Poor gaze tracking**:
   - Ensure good lighting
   - Position 50-70cm from camera
   - Clean camera lens
   - Remove glasses if possible (Phase 1 limitation)

2. **Frequent detection loss**:
   - Improve lighting
   - Ensure face is fully visible
   - Check camera focus

## Debugging Tips

### Enable Debug Mode

```python
config = Config()
config.system.debug_mode = True
config.system.save_debug_frames = True
```

### Check Logs

```bash
# System logs
tail -f logs/system.log

# Python debug
python -u examples/basic_gaze_tracking.py 2>&1 | tee debug.log
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile the code
cProfile.run('monitor.start_monitoring(); time.sleep(10); monitor.stop_monitoring()', 'profile_stats')

# Analyze
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## Reporting Issues

When reporting issues, include:

1. Output of `python test_installation.py`
2. System info: OS, Python version, hardware specs
3. Error messages and stack traces
4. Steps to reproduce
5. Config file (if customized)
6. Sample log files