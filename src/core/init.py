"""
Innomatics OMR System - Core Processing Components
Classical Computer Vision based OMR evaluation engine
"""

__version__ = "1.0.0"

# Import core components
from .coordinate_mapper import InnoMaticsCoordinateMapper
from .bubble_detector import InnoMaticsBubbleDetector
from .omr_processor import InnoMaticsOMRProcessor

__all__ = [
    'InnoMaticsCoordinateMapper',
    'InnoMaticsBubbleDetector', 
    'InnoMaticsOMRProcessor'
]
