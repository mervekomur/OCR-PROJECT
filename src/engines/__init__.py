"""
OCR Engines Package
Multiple OCR engine implementations for ensemble processing.
"""

from .base import BaseOCREngine, OCRResult
from .easyocr_engine import EasyOCREngine
from .paddleocr_engine import PaddleOCREngine
from .donut_engine import DonutEngine
from .got_ocr_engine import GOTOCREngine
from .google_vision_engine import GoogleVisionEngine
from .claude_vision_engine import ClaudeVisionEngine
from .structure_engine import StructureEngine, SAPDocument, extract_receipt_data
from .regex_structure_engine import RegexStructureEngine, RegexStructuredReceipt, extract_with_regex
from .financial_validator import FinancialValidator, ValidationResult, validate_receipt
from .ensemble import EnsembleOCR, compare_engines

__all__ = [
    'BaseOCREngine',
    'OCRResult',
    'EasyOCREngine',
    'PaddleOCREngine',
    'DonutEngine',
    'GOTOCREngine',
    'GoogleVisionEngine',
    'ClaudeVisionEngine',
    'StructureEngine',
    'SAPDocument',
    'extract_receipt_data',
    'RegexStructureEngine',
    'RegexStructuredReceipt',
    'extract_with_regex',
    'FinancialValidator',
    'ValidationResult',
    'validate_receipt',
    'EnsembleOCR',
    'compare_engines'
]
