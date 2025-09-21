"""
Innomatics OMR System - Core Processing Package
Enterprise-grade OMR processing components
"""

__version__ = "1.0.0"
__author__ = "Code4EdTech Challenge Team"

# Core components
from .core.omr_processor import InnoMaticsOMRProcessor
from .core.coordinate_mapper import InnoMaticsCoordinateMapper
from .core.bubble_detector import InnoMaticsBubbleDetector

__all__ = [
    'InnoMaticsOMRProcessor',
    'InnoMaticsCoordinateMapper', 
    'InnoMaticsBubbleDetector'
]
