"""
GOT-OCR (General OCR Theory) Engine Implementation
"""

from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from .base import BaseOCREngine, OCRResult

logger = get_logger(__name__)


class GOTOCREngine(BaseOCREngine):
    """GOT-OCR based text extraction engine."""

    name = "got-ocr"
    description = "GOT-OCR 2.0 - General OCR Theory (Transformer-based)"

    # Hugging Face model name
    MODEL_NAME = "stepfun-ai/GOT-OCR2_0"

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize GOT-OCR engine.

        Args:
            model_name: Hugging Face model name
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        super().__init__()
        self.model_name = model_name or self.MODEL_NAME
        self.device = device
        self._tokenizer = None

    @classmethod
    def _check_availability(cls) -> bool:
        """
        Check if GOT-OCR is available.

        NOTE: GOT-OCR requires ~4-5GB model download and GPU (CUDA required).
        Disabled for local development. Enable for Colab/Cloud testing.
        """
        # Disabled: Very heavy model (~4-5GB), requires CUDA GPU
        # To enable: return True and ensure torch + transformers are installed
        return False

    def _initialize(self) -> None:
        """Initialize GOT-OCR model and tokenizer."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading GOT-OCR model: {self.model_name}...")

        # Auto-detect device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=self.device if self.device == "cuda" else None,
            use_safetensors=True
        )

        if self.device == "cpu":
            self._model = self._model.float()
        else:
            self._model = self._model.half().to(self.device)

        self._model.eval()

        logger.info(f"GOT-OCR ready on {self.device}.")

    def _extract_text(self, image_path: str) -> OCRResult:
        """Extract text using GOT-OCR model."""
        import torch

        # Use the model's chat method for OCR
        try:
            with torch.no_grad():
                result_text = self._model.chat(
                    self._tokenizer,
                    image_path,
                    ocr_type='ocr'  # Plain OCR mode
                )
        except Exception as e:
            logger.error(f"GOT-OCR extraction failed: {e}")
            result_text = ""

        # Parse the result
        lines = []
        if result_text:
            for line in result_text.split('\n'):
                line = line.strip()
                if line:
                    lines.append({
                        'text': line,
                        'confidence': 0.8  # GOT-OCR doesn't provide confidence
                    })

        # Calculate overall confidence
        confidence = 0.8 if lines else 0.0

        # Extract fields
        fields = self._extract_fields_from_text(result_text, lines)

        return OCRResult(
            raw_text=result_text,
            lines=lines,
            confidence=confidence,
            fields=fields,
            metadata={
                'model': self.model_name,
                'device': self.device,
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

        if not text:
            return fields

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
                fields['date_confidence'] = 0.8
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
                fields['total_confidence'] = 0.8
                break

        # Extract merchant (first meaningful line)
        for line in lines[:5]:
            line_text = line['text'].strip()
            if len(line_text) > 3:
                fields['merchant'] = line_text
                fields['merchant_confidence'] = 0.8
                break

        return fields
