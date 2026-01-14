"""
PaddleOCR Engine Implementation
"""

from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from .base import BaseOCREngine, OCRResult

logger = get_logger(__name__)


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR-based text extraction engine."""

    name = "paddleocr"
    description = "PaddleOCR - High accuracy multi-language OCR by Baidu"

    def __init__(self, lang: str = 'tr', use_gpu: bool = False):
        """
        Initialize PaddleOCR engine.

        Args:
            lang: Language code ('tr', 'en', 'fr', etc.)
            use_gpu: Enable GPU acceleration
        """
        super().__init__()
        self.lang = lang
        self.use_gpu = use_gpu

    @classmethod
    def _check_availability(cls) -> bool:
        """Check if PaddleOCR is installed."""
        try:
            from paddleocr import PaddleOCR
            return True
        except ImportError:
            return False

    def _initialize(self) -> None:
        """Initialize PaddleOCR."""
        from paddleocr import PaddleOCR

        logger.info(f"Loading PaddleOCR (lang: {self.lang}, gpu: {self.use_gpu})...")

        # PaddleOCR language mapping
        # Turkish is not directly supported, use 'en' with Turkish-like settings
        paddle_lang = self.lang if self.lang in ['en', 'ch', 'fr', 'german', 'korean', 'japan'] else 'en'

        self._model = PaddleOCR(
            use_angle_cls=True,
            lang=paddle_lang,
            use_gpu=self.use_gpu,
            show_log=False
        )
        logger.info("PaddleOCR ready.")

    def _extract_text(self, image_path: str) -> OCRResult:
        """Extract text using PaddleOCR."""
        results = self._model.ocr(image_path, cls=True)

        lines = []
        total_confidence = 0.0
        text_parts = []

        # PaddleOCR returns nested list structure
        if results and results[0]:
            for line in results[0]:
                bbox = line[0]
                text = line[1][0]
                confidence = line[1][1]

                lines.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
                text_parts.append(text)
                total_confidence += confidence

        avg_confidence = total_confidence / len(lines) if lines else 0.0
        raw_text = '\n'.join(text_parts)

        # Extract fields
        fields = self._extract_fields_from_text(raw_text, lines)

        return OCRResult(
            raw_text=raw_text,
            lines=lines,
            confidence=avg_confidence,
            fields=fields,
            metadata={
                'lang': self.lang,
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

        # Extract date
        date_patterns = [
            r'(\d{2}[/.-]\d{2}[/.-]\d{4})',
            r'(\d{2}[/.-]\d{2}[/.-]\d{2})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                fields['date'] = match.group(1)
                for line in lines:
                    if match.group(1) in line['text']:
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

        # Extract total
        total_patterns = [
            r'TOPLAM\s*[:.]?\s*\*?(\d+[.,]\d{2})',
            r'TOTAL\s*[:.]?\s*[€$£₺]?\s*(\d+[.,]\d{2})',
            r'TUTAR\s*[:.]?\s*(\d+[.,]\d{2})',
        ]
        for pattern in total_patterns:
            match = re.search(pattern, text_upper)
            if match:
                fields['total'] = float(match.group(1).replace(',', '.'))
                for line in lines:
                    if 'TOPLAM' in line['text'].upper() or 'TOTAL' in line['text'].upper():
                        fields['total_confidence'] = line['confidence']
                        break
                break

        # Extract merchant
        for line in lines[:5]:
            line_text = line['text'].strip()
            if len(line_text) > 3:
                fields['merchant'] = line_text
                fields['merchant_confidence'] = line['confidence']
                break

        return fields
