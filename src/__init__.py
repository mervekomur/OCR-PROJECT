# -*- coding: utf-8 -*-
"""
OCR Project - SAM tabanlı fiş/belge tespiti

Kullanım:
    from src import detect_receipt, detect_and_save

    # Tek satırda tespit
    result = detect_receipt("path/to/image.jpg")

    # Tespit ve kaydet
    result = detect_and_save("path/to/image.jpg", "output/")
"""

from src.sam_detector import (
    ReceiptDetector,
    detect_receipt,
    detect_and_save,
    get_detector
)

__all__ = [
    'ReceiptDetector',
    'detect_receipt',
    'detect_and_save',
    'get_detector'
]

__version__ = '2.0.0'
