"""
Production Configuration Settings
"""

from pathlib import Path
import logging
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
WEB_DIR = PROJECT_ROOT / "web"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Data directories
SAMPLES_DIR = DATA_DIR / "samples"
ANSWER_KEYS_DIR = DATA_DIR / "answer_keys"

# Create directories
for directory in [DATA_DIR, SAMPLES_DIR, ANSWER_KEYS_DIR, LOGS_DIR, EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# OMR Processing Settings
OMR_SETTINGS = {
    "detection_threshold": 0.3,
    "confidence_threshold": 0.5,
    "high_confidence_threshold": 0.8,
    "bubble_radius": 15,
    "processing_timeout": 30
}

# Coordinate Settings
COORDINATE_SETTINGS = {
    "questions_per_column": 20,
    "total_columns": 5,
    "column_spacing": 140,
    "row_spacing": 32,
    "base_coordinates": {
        "start_x": 184,
        "start_y": 262,
        "option_spacing": 32
    }
}

# System Information
SYSTEM_INFO = {
    "name": "Innomatics OMR Evaluation System",
    "version": "1.0.0",
    "description": "Professional OMR Processing System",
    "author": "Code4EdTech Challenge Team",
    "competition": "Innomatics Research Labs 2025"
}
