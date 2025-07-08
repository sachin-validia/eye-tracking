#!/usr/bin/env python3
"""
Test Installation Script

Run this script to verify that all dependencies are correctly installed.
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*50)
    print(f" {text}")
    print("="*50)

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✓ {package_name:<20} {version}")
        return True
    except ImportError as e:
        print(f"✗ {package_name:<20} Failed: {e}")
        return False

def test_camera():
    """Test camera access"""
    print("\nTesting camera access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera working: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("✗ Camera opened but cannot read frames")
            cap.release()
        else:
            print("✗ Cannot open camera")
            return False
        return True
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_mediapipe_face():
    """Test MediaPipe face detection"""
    print("\nTesting MediaPipe face detection...")
    try:
        import cv2
        import mediapipe as mp
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Try to get a frame and detect face
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    print(f"✓ Face detected with {len(results.multi_face_landmarks[0].landmark)} landmarks")
                else:
                    print("! No face detected - please ensure your face is visible to camera")
            cap.release()
        face_mesh.close()
        return True
    except Exception as e:
        print(f"✗ MediaPipe test failed: {e}")
        return False

def test_dlib():
    """Test dlib installation"""
    print("\nTesting dlib...")
    try:
        import dlib
        print(f"✓ dlib version: {dlib.__version__}")
        # Test that we can create a face detector
        detector = dlib.get_frontal_face_detector()
        print("✓ dlib face detector created successfully")
        return True
    except Exception as e:
        print(f"✗ dlib test failed: {e}")
        return False

def test_project_structure():
    """Test project structure"""
    print("\nTesting project structure...")
    required_dirs = ['config', 'core', 'utils', 'api', 'examples']
    required_files = ['requirements.txt', 'setup.py']
    
    all_good = True
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ Directory '{dir_name}' exists")
        else:
            print(f"✗ Directory '{dir_name}' missing")
            all_good = False
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✓ File '{file_name}' exists")
        else:
            print(f"✗ File '{file_name}' missing")
            all_good = False
    
    return all_good

def test_gui_support():
    """Test GUI support (especially for WSL)"""
    print("\nTesting GUI support...")
    try:
        import os
        
        # Check DISPLAY variable
        display = os.environ.get('DISPLAY', '')
        if display:
            print(f"✓ DISPLAY variable set: {display}")
        else:
            print("! DISPLAY variable not set (GUI may not work)")
            print("  For WSL, run: export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0")
        
        # Try to create a window
        import cv2
        import numpy as np
        
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow('Test Window', test_img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        print("✓ GUI window creation successful")
        return True
        
    except Exception as e:
        print(f"! GUI test failed: {e}")
        print("  Make sure X server is running (VcXsrv on Windows)")
        return False

def main():
    """Run all tests"""
    print_header("Eye Tracking System Installation Test")
    
    # Test Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher required")
        return
    else:
        print("✓ Python version OK")
    
    # Test core dependencies
    print_header("Testing Core Dependencies")
    
    dependencies = [
        ('cv2', 'opencv-python'),
        ('mediapipe', 'mediapipe'),
        ('numpy', 'numpy'),
        ('PIL', 'Pillow'),
        ('yaml', 'PyYAML'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas'),
    ]
    
    all_imports_ok = True
    for module, package in dependencies:
        if not test_import(module, package):
            all_imports_ok = False
    
    # Test dlib separately as it's more complex
    if not test_dlib():
        all_imports_ok = False
    
    # Test hardware
    print_header("Testing Hardware")
    camera_ok = test_camera()
    
    # Test MediaPipe
    print_header("Testing MediaPipe Face Detection")
    mediapipe_ok = test_mediapipe_face()
    
    # Test GUI
    print_header("Testing GUI Support")
    gui_ok = test_gui_support()
    
    # Test project structure
    print_header("Testing Project Structure")
    structure_ok = test_project_structure()
    
    # Summary
    print_header("Test Summary")
    
    if all_imports_ok and camera_ok and mediapipe_ok and structure_ok:
        print("✓ All core tests passed!")
        print("\nYou can now run the demo:")
        print("  python examples/basic_gaze_tracking.py")
        
        if not gui_ok:
            print("\n! Warning: GUI support may have issues")
            print("  The system will still work but visualization may not display")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. For dlib issues: sudo apt install cmake build-essential")
        print("3. For camera issues: check permissions and device availability")
        print("4. For WSL: ensure VcXsrv is running and DISPLAY is set")

if __name__ == "__main__":
    main()