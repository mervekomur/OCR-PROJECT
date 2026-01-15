"""
EasyOCR Engine Implementation
"""

import cv2
from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from preprocessing import document_scan_preprocess
from .base import BaseOCREngine, OCRResult

logger = get_logger(__name__)


class EasyOCREngine(BaseOCREngine):
    """EasyOCR-based text extraction engine."""

    name = "easyocr"
    description = "EasyOCR - Multi-language OCR with good Turkish support"

    def __init__(self, languages: List[str] = None, gpu: bool = False, preprocess: bool = True):
        """
        Initialize EasyOCR engine.

        Args:
            languages: Languages to support (default: ['tr', 'en'])
            gpu: Enable GPU acceleration
            preprocess: Apply document scanning preprocessing (default: True)
        """
        super().__init__()
        self.languages = languages or ['tr', 'en']
        self.gpu = gpu
        self.preprocess = preprocess

    @classmethod
    def _check_availability(cls) -> bool:
        """Check if EasyOCR is installed."""
        try:
            import easyocr
            return True
        except ImportError:
            return False

    def _initialize(self) -> None:
        """Initialize EasyOCR reader."""
        import easyocr

        logger.info(f"Loading EasyOCR (languages: {self.languages}, gpu: {self.gpu})...")
        self._model = easyocr.Reader(self.languages, gpu=self.gpu)
        logger.info("EasyOCR ready.")

    def _extract_text(self, image_path: str) -> OCRResult:
        """Extract text using EasyOCR with optional preprocessing."""
        # Always read image with OpenCV for consistency
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        if self.preprocess:
            # Apply document scanning preprocessing
            image = document_scan_preprocess(image)
            logger.debug("Document scanning preprocessing applied")

        results = self._model.readtext(image)

        lines = []
        total_confidence = 0.0

        for bbox, text, confidence in results:
            lines.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
            total_confidence += confidence

        avg_confidence = total_confidence / len(results) if results else 0.0
        raw_text = '\n'.join([item[1] for item in results])

        # Extract fields from raw text
        fields = self._extract_fields_from_text(raw_text, lines)

        return OCRResult(
            raw_text=raw_text,
            lines=lines,
            confidence=avg_confidence,
            fields=fields,
            metadata={
                'languages': self.languages,
                'line_count': len(lines)
            }
        )

    def _extract_fields_from_text(
        self,
        text: str,
        lines: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract structured fields from OCR text."""
        import re

        fields = {
            'date': None,
            'time': None,
            'total': None,
            'merchant': None,
            'date_confidence': 0.0,
            'total_confidence': 0.0,
            'merchant_confidence': 0.0
        }

        text_upper = text.upper()

        # Extract date - multiple formats supported
        date_patterns = [
            r'(\d{2}[/.\-]\d{2}[/.\-]\d{4})',      # DD/MM/YYYY, DD.MM.YYYY, DD-MM-YYYY
            r'(\d{2}[/.\-]\d{2}[/.\-]\d{2})',      # DD/MM/YY, DD.MM.YY, DD-MM-YY
            r'(\d{2}\s+\d{2}\s+\d{4})',            # DD MM YYYY (space separated)
            r'(\d{2}\s+\d{2}\s+\d{2})',            # DD MM YY (space separated)
            r'(\d{4}[/.\-]\d{2}[/.\-]\d{2})',      # YYYY/MM/DD, YYYY.MM.DD
            r'Tarih[:\s]*(\d{2}[/.\-\s]\d{2}[/.\-\s]\d{2,4})',  # Tarih: DD.MM.YY
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(match.lastindex)
                fields['date'] = date_str.strip()
                for line in lines:
                    if date_str in line['text']:
                        fields['date_confidence'] = line['confidence']
                        break
                break

        # Extract time
        time_patterns = [
            r'(\d{2}[:.]\d{2}[:.]\d{2})',
            r'(\d{2}[:.]\d{2})',
        ]
        for pattern in time_patterns:
            match = re.search(pattern, text)
            if match:
                fields['time'] = match.group(1)
                break

        # Extract total - multiple formats supported
        total_patterns = [
            r'TOPLAM[\s\n]*[:.]?\s*\*?(\d+[.,]\d{1,2})',   # TOPLAM 1000.70 or 1000.7
            r'TOPLAM[\s\n]*[:.]?\s*\*?(\d+)[.,]?(?!\d)',   # TOPLAM 1000 (no decimal)
            r'TOTAL[\s\n]*[:.]?\s*[€$£₺]?\s*(\d+[.,]\d{1,2})',
            r'TUTAR[\s\n]*[:.]?\s*(\d+[.,]\d{1,2})',
            r'(\d+[.,]\d{1,2})\s*TL',                       # 1000.70 TL format
        ]
        for pattern in total_patterns:
            match = re.search(pattern, text_upper)
            if match:
                total_str = match.group(1).replace(',', '.')
                total_str = total_str.rstrip('.')
                try:
                    fields['total'] = float(total_str)
                    for line in lines:
                        if 'TOPLAM' in line['text'].upper() or 'TOTAL' in line['text'].upper():
                            fields['total_confidence'] = line['confidence']
                            break
                    break
                except ValueError:
                    continue

        # Extract merchant - skip serial numbers and metadata
        merchant_blacklist = ['SERI', 'SIRA', 'NO:', 'NO.', 'ABS', 'FIS', 'FİŞ',
                              'TARIH', 'TARİH', 'SAAT', 'VKN', 'TCKN', 'PERAKENDE',
                              'SATIS', 'SATIŞ']
        for line in lines[:8]:  # Check more lines
            line_text = line['text'].strip()
            line_upper = line_text.upper()

            # Skip if too short
            if len(line_text) < 3:
                continue

            # Skip if contains blacklisted words
            if any(bl in line_upper for bl in merchant_blacklist):
                continue

            # Skip if looks like a number/code (mostly digits)
            digit_ratio = sum(c.isdigit() for c in line_text) / len(line_text)
            if digit_ratio > 0.5:
                continue

            fields['merchant'] = line_text
            fields['merchant_confidence'] = line['confidence']
            break

        return fields
