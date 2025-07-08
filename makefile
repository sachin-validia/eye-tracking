# Eye Tracking System Makefile
# Common development and deployment tasks

.PHONY: help install test run clean lint format docker

# Default target
help:
	@echo "Eye Tracking System - Development Commands"
	@echo "========================================="
	@echo "make install    - Install all dependencies"
	@echo "make test       - Run tests"
	@echo "make run        - Run the demo application"
	@echo "make lint       - Run code linting"
	@echo "make format     - Format code with black"
	@echo "make clean      - Clean up generated files"
	@echo "make docker     - Build Docker image"
	@echo "make docs       - Generate documentation"

# Install dependencies
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	@echo "Installing GazeTracking..."
	pip install git+https://github.com/antoinelame/GazeTracking.git

# Run tests
test:
	python test_installation.py
	# pytest tests/ -v  # Uncomment when tests are added

# Run the demo
run:
	python examples/basic_gaze_tracking.py

# Run with different modes
run-fast:
	python examples/basic_gaze_tracking.py --mode high_speed

run-accurate:
	python examples/basic_gaze_tracking.py --mode high_accuracy

# Lint code
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Format code
format:
	black . --line-length 100

# Type checking
typecheck:
	mypy . --ignore-missing-imports

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/
	rm -rf logs/*.csv logs/*.json
	rm -rf debug_frames/*

# Create directories
dirs:
	mkdir -p logs logs/summaries debug_frames

# Docker build (Phase 4)
docker:
	@echo "Docker support will be added in Phase 4"
	# docker build -t eye-tracking-system .

# Generate documentation
docs:
	@echo "Generating API documentation..."
	# sphinx-build -b html docs/ docs/_build/

# Development setup
dev-setup: install
	pip install -r requirements.txt
	pre-commit install

# Quick test camera
test-camera:
	python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL'); cap.release()"

# Profile performance
profile:
	python -m cProfile -o profile_stats examples/basic_gaze_tracking.py
	python -c "import pstats; stats = pstats.Stats('profile_stats'); stats.sort_stats('cumulative'); stats.print_stats(20)"

# WSL specific setup
wsl-setup:
	sudo apt update
	sudo apt install -y x11-apps python3-opencv cmake build-essential
	@echo "Remember to:"
	@echo "1. Install VcXsrv on Windows"
	@echo "2. Set DISPLAY variable"
	@echo "3. Run XLaunch with access control disabled"