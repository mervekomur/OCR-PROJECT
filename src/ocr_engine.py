"""
OCR Engine Module
Text extraction from images using EasyOCR.
"""

import easyocr
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils.logger import get_logger
from constants import DEFAULT_LANGUAGES, DEFAULT_GPU

logger = get_logger(__name__)


class OCREngine:
    """EasyOCR-based OCR engine."""

    def __init__(self, languages: List[str] = None, gpu: bool = None):
        """
        Initialize OCR engine.

        Args:
            languages: Languages to support (default: Turkish and English)
            gpu: Enable GPU (default: False)
        """
        self.languages = languages or DEFAULT_LANGUAGES
        self.gpu = gpu if gpu is not None else DEFAULT_GPU
        self.reader = None

    def _ensure_reader(self):
        """Lazy load the OCR reader."""
        if self.reader is None:
            logger.info(f"Loading EasyOCR (languages: {self.languages})...")
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info("EasyOCR ready.")

    def extract_text(self, image: np.ndarray, detail: bool = False) -> Any:
        """
        Extract text from image.

        Args:
            image: Image as numpy array
            detail: Return position info if True

        Returns:
            detail=False: Text string
            detail=True: List [(bbox, text, confidence), ...]
        """
        self._ensure_reader()

        results = self.reader.readtext(image)

        if detail:
            return results

        texts = [item[1] for item in results]
        return '\n'.join(texts)

    def extract_text_from_file(
        self,
        image_path: str,
        detail: bool = False
    ) -> Any:
        """
        Extract text from image file.

        Args:
            image_path: Path to image file
            detail: Return position info if True

        Returns:
            detail=False: Text string
            detail=True: List [(bbox, text, confidence), ...]
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {image_path}")

        self._ensure_reader()

        results = self.reader.readtext(str(path))

        if detail:
            return results

        texts = [item[1] for item in results]
        return '\n'.join(texts)

    def extract_structured(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract structured data from image.

        Args:
            image: Image as numpy array

        Returns:
            Dict: {
                'raw_text': str,
                'lines': List[Dict],
                'confidence_avg': float
            }
        """
        self._ensure_reader()

        results = self.reader.readtext(image)

        lines = []
        total_confidence = 0

        for bbox, text, confidence in results:
            lines.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
            total_confidence += confidence

        avg_confidence = total_confidence / len(results) if results else 0

        return {
            'raw_text': '\n'.join([item[1] for item in results]),
            'lines': lines,
            'confidence_avg': avg_confidence
        }


# Module-level default instance
_default_engine: Optional[OCREngine] = None


def get_engine() -> OCREngine:
    """Get default OCR engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = OCREngine()
    return _default_engine


def extract_text(image_path: str) -> str:
    """
    Simple function for text extraction.

    Args:
        image_path: Path to image file

    Returns:
        Extracted text
    """
    engine = get_engine()
    return engine.extract_text_from_file(image_path)
