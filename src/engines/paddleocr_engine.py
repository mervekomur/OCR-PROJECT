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

    def __init__(self, lang: str = 'tr'):
        """
        Initialize PaddleOCR engine (CPU mode).

        Args:
            lang: Language code ('tr', 'en', 'fr', etc.)
        """
        super().__init__()
        self.lang = lang
        self._paddle_available = None

    @classmethod
    def _check_availability(cls) -> bool:
        """Check if PaddleOCR is installed."""
        try:
            from paddleocr import PaddleOCR
            import paddleocr
            version = getattr(paddleocr, '__version__', '0.0.0')
            # PaddleOCR 2.x is supported (CPU mode)
            # PaddleOCR 3.x has API issues on Windows
            if version.startswith('3.'):
                logger.warning(f"PaddleOCR 3.x detected ({version}), may have issues. Consider downgrade to 2.7.x")
                return False
            return True
        except ImportError:
            return False

    def _initialize(self) -> None:
        """Initialize PaddleOCR (CPU mode, no GPU required)."""
        from paddleocr import PaddleOCR

        logger.info(f"Loading PaddleOCR (lang: {self.lang}, CPU mode)...")

        # PaddleOCR language mapping
        # Turkish not directly supported, use English model
        paddle_lang = self.lang if self.lang in ['en', 'ch', 'fr', 'german', 'korean', 'japan'] else 'en'

        # Initialize with CPU mode explicitly
        self._model = PaddleOCR(
            use_angle_cls=True,  # Enable text angle classification
            lang=paddle_lang,
            use_gpu=False,       # Force CPU mode
            show_log=False       # Suppress verbose logging
        )
        logger.info("PaddleOCR ready (CPU mode).")

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
                # Get the last group (in case of Tarih: prefix pattern)
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
                # Remove trailing dot if exists
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

        # Extract merchant
        for line in lines[:5]:
            line_text = line['text'].strip()
            if len(line_text) > 3:
                fields['merchant'] = line_text
                fields['merchant_confidence'] = line['confidence']
                break

        return fields
