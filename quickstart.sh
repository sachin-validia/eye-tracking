#!/bin/bash
# Eye Tracking System Quick Start Script
# Supports Ubuntu/Debian, macOS, and WSL

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if grep -q Microsoft /proc/version; then
            OS="wsl"
        else
            OS="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        print_error "Unsupported OS: $OSTYPE"
        exit 1
    fi
    print_status "Detected OS: $OS"
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    if [[ "$OS" == "linux" ]] || [[ "$OS" == "wsl" ]]; then
        sudo apt update
        sudo apt install -y \
            python3.8 python3-pip python3-venv \
            cmake build-essential \
            libopencv-dev python3-opencv \
            libboost-all-dev \
            libx11-dev libgtk-3-dev \
            libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
            git wget
            
        if [[ "$OS" == "wsl" ]]; then
            print_status "Installing WSL-specific packages..."
            sudo apt install -y x11-apps
            
            # Check if DISPLAY is set
            if [ -z "$DISPLAY" ]; then
                print_warning "DISPLAY not set. Adding to ~/.bashrc"
                echo 'export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '"'"'{print $2}'"'"'):0' >> ~/.bashrc
                echo 'export LIBGL_ALWAYS_INDIRECT=1' >> ~/.bashrc
                export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
                export LIBGL_ALWAYS_INDIRECT=1
            fi
        fi
        
    elif [[ "$OS" == "macos" ]]; then
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            print_error "Homebrew not found. Please install from https://brew.sh"
            exit 1
        fi
        
        print_status "Installing macOS dependencies..."
        brew install cmake boost python@3.8
        brew install opencv
    fi
}

# Create virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Activating..."
    else
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
}

# Install Python packages
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Install numpy first (dependency for many packages)
    pip install numpy
    
    # Install main requirements
    pip install -r requirements.txt
    
    # Install dlib (may take time)
    print_status "Installing dlib (this may take several minutes)..."
    pip install dlib
    
    # Install GazeTracking
    print_status "Installing GazeTracking library..."
    pip install git+https://github.com/antoinelame/GazeTracking.git
    
    # Install package in development mode
    pip install -e .
}

# Create necessary directories
setup_directories() {
    print_status "Creating project directories..."
    
    mkdir -p logs
    mkdir -p logs/summaries
    mkdir -p debug_frames
    
    # Create default config if not exists
    if [ ! -f "config.yaml" ]; then
        cp config.yaml.example config.yaml 2>/dev/null || true
    fi
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    python test_installation.py
    
    if [ $? -eq 0 ]; then
        print_status "Installation test passed!"
    else
        print_error "Installation test failed. Please check the errors above."
        exit 1
    fi
}

# WSL-specific instructions
wsl_instructions() {
    if [[ "$OS" == "wsl" ]]; then
        echo
        print_warning "WSL-specific instructions:"
        echo "1. Install VcXsrv on Windows from: https://sourceforge.net/projects/vcxsrv/"
        echo "2. Run XLaunch with these settings:"
        echo "   - Multiple windows"
        echo "   - Start no client"
        echo "   - Disable access control (important!)"
        echo "3. Restart your terminal to apply DISPLAY settings"
        echo
        echo "For camera access in WSL2:"
        echo "1. Install usbipd-win on Windows"
        echo "2. Run in admin PowerShell: usbipd wsl list"
        echo "3. Attach camera: usbipd wsl attach --busid <BUSID>"
        echo
    fi
}

# Main installation flow
main() {
    echo "========================================"
    echo "Eye Tracking System Quick Start"
    echo "========================================"
    echo
    
    # Check if running in project directory
    if [ ! -f "requirements.txt" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    detect_os
    install_system_deps
    setup_venv
    install_python_deps
    setup_directories
    test_installation
    wsl_instructions
    
    echo
    print_status "Installation complete!"
    echo
    echo "To run the demo:"
    echo "  source venv/bin/activate"
    echo "  python examples/basic_gaze_tracking.py"
    echo
    echo "For more options:"
    echo "  python examples/basic_gaze_tracking.py --help"
    echo
    print_status "Happy tracking!"
}

# Run main function
main