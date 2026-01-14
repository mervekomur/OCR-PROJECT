"""
OCR Engines Package
Multiple OCR engine implementations for ensemble processing.
"""

from .base import BaseOCREngine, OCRResult
from .easyocr_engine import EasyOCREngine
from .paddleocr_engine import PaddleOCREngine
from .donut_engine import DonutEngine
from .got_ocr_engine import GOTOCREngine
from .ensemble import EnsembleOCR, compare_engines

__all__ = [
    'BaseOCREngine',
    'OCRResult',
    'EasyOCREngine',
    'PaddleOCREngine',
    'DonutEngine',
    'GOTOCREngine',
    'EnsembleOCR',
    'compare_engines'
]
