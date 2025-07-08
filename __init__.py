"""
Eye Tracking System

A production-ready eye tracking system for interview monitoring and behavioral analysis.
"""

__version__ = "0.1.0"
__author__ = "Eye Tracking System Team"

from .api import InterviewMonitor, MonitoringSession, MonitoringState
from .config import Config, get_config, PerformanceMode, LogLevel

__all__ = [
    'InterviewMonitor', 'MonitoringSession', 'MonitoringState',
    'Config', 'get_config', 'PerformanceMode', 'LogLevel',
    '__version__'
]