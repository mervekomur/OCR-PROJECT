"""
Google Cloud Vision OCR Engine Implementation
Cloud-based OCR using Google Cloud Vision API.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from .base import BaseOCREngine, OCRResult

# Default credentials file in project root
DEFAULT_CREDENTIALS_FILE = "pebble-451721-bdc8fa59a91d.json"


class GoogleVisionEngine(BaseOCREngine):
    """Google Cloud Vision API-based text extraction engine."""

    name = "google_vision"
    description = "Google Cloud Vision - High accuracy cloud OCR with multi-language support"

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        language_hints: Optional[List[str]] = None
    ):
        """
        Initialize Google Cloud Vision engine.

        Args:
            credentials_path: Path to service account JSON file.
                              If None, looks for default credentials in project root.
            language_hints: Language hints for OCR (e.g., ['tr', 'en']).
                           Improves accuracy for specific languages.
        """
        super().__init__()

        # Resolve credentials path
        if credentials_path:
            self.credentials_path = credentials_path
        else:
            # Look for default credentials in project root
            project_root = Path(__file__).parent.parent.parent
            default_path = project_root / DEFAULT_CREDENTIALS_FILE
            if default_path.exists():
                self.credentials_path = str(default_path)
            else:
                self.credentials_path = None

        self.language_hints = language_hints or ['tr', 'en']
        self._client = None
        self._credentials = None

    @classmethod
    def _check_availability(cls) -> bool:
        """Check if google-cloud-vision and oauth2 are installed."""
        try:
            from google.cloud import vision
            from google.oauth2 import service_account
            return True
        except ImportError:
            return False

    def _initialize(self) -> None:
        """Initialize Google Cloud Vision client with service account credentials."""
        from google.cloud import vision
        from google.oauth2 import service_account

        # Verify credentials path exists
        if not self.credentials_path:
            raise ValueError(
                f"Google Cloud credentials not found. "
                f"Place '{DEFAULT_CREDENTIALS_FILE}' in project root "
                f"or provide credentials_path parameter."
            )

        if not Path(self.credentials_path).exists():
            raise FileNotFoundError(
                f"Credentials file not found: {self.credentials_path}"
            )

        # Load credentials from service account file
        self._credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-vision']
        )

        # Create client with explicit credentials
        self._client = vision.ImageAnnotatorClient(credentials=self._credentials)
        self._model = self._client  # For compatibility with base class

    def _extract_text(self, image_path: str) -> OCRResult:
        """Extract text using Google Cloud Vision API."""
        from google.cloud import vision

        # Read image file
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Configure language hints
        image_context = vision.ImageContext(
            language_hints=self.language_hints
        )

        # Perform text detection (DOCUMENT_TEXT_DETECTION for better accuracy)
        response = self._client.document_text_detection(
            image=image,
            image_context=image_context
        )

        # Check for errors
        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")

        # Extract full text
        full_text_annotation = response.full_text_annotation
        raw_text = full_text_annotation.text if full_text_annotation else ""

        # Process text annotations for line-level data
        lines = []
        total_confidence = 0.0
        word_count = 0

        if full_text_annotation:
            for page in full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        paragraph_text = ""
                        paragraph_confidence = 0.0
                        word_confidences = []

                        for word in paragraph.words:
                            word_text = ''.join([
                                symbol.text for symbol in word.symbols
                            ])
                            paragraph_text += word_text + " "
                            word_confidences.append(word.confidence)
                            total_confidence += word.confidence
                            word_count += 1

                        paragraph_text = paragraph_text.strip()
                        if paragraph_text:
                            avg_word_conf = (
                                sum(word_confidences) / len(word_confidences)
                                if word_confidences else 0.0
                            )

                            # Get bounding box
                            vertices = paragraph.bounding_box.vertices
                            bbox = [
                                [v.x, v.y] for v in vertices
                            ]

                            lines.append({
                                'text': paragraph_text,
                                'confidence': avg_word_conf,
                                'bbox': bbox,
                                'block_type': block.block_type.name
                            })

        avg_confidence = total_confidence / word_count if word_count > 0 else 0.0

        # Extract fields from raw text
        fields = self._extract_fields_from_text(raw_text, lines)

        return OCRResult(
            raw_text=raw_text,
            lines=lines,
            confidence=avg_confidence,
            fields=fields,
            metadata={
                'language_hints': self.language_hints,
                'line_count': len(lines),
                'word_count': word_count,
                'api_version': 'v1'
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
            r'(\d{2}[/.\-]\d{2}[/.\-]\d{4})',      # DD/MM/YYYY
            r'(\d{2}[/.\-]\d{2}[/.\-]\d{2})',      # DD/MM/YY
            r'(\d{2}\s+\d{2}\s+\d{4})',            # DD MM YYYY
            r'(\d{2}\s+\d{2}\s+\d{2})',            # DD MM YY
            r'(\d{4}[/.\-]\d{2}[/.\-]\d{2})',      # YYYY/MM/DD
            r'Tarih[:\s]*(\d{2}[/.\-\s]\d{2}[/.\-\s]\d{2,4})',
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

        # Extract total - multiple formats
        total_patterns = [
            r'TOPLAM[\s\n]*[:.]?\s*\*?(\d+[.,]\d{1,2})',
            r'TOPLAM[\s\n]*[:.]?\s*\*?(\d+)[.,]?(?!\d)',
            r'TOTAL[\s\n]*[:.]?\s*[€$£₺]?\s*(\d+[.,]\d{1,2})',
            r'TUTAR[\s\n]*[:.]?\s*(\d+[.,]\d{1,2})',
            r'(\d+[.,]\d{1,2})\s*TL',
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

        # Extract merchant name
        merchant_blacklist = [
            'SERI', 'SIRA', 'NO:', 'NO.', 'ABS', 'FIS', 'FİŞ',
            'TARIH', 'TARİH', 'SAAT', 'VKN', 'TCKN', 'PERAKENDE',
            'SATIS', 'SATIŞ'
        ]
        for line in lines[:8]:
            line_text = line['text'].strip()
            line_upper = line_text.upper()

            if len(line_text) < 3:
                continue

            if any(bl in line_upper for bl in merchant_blacklist):
                continue

            digit_ratio = sum(c.isdigit() for c in line_text) / len(line_text)
            if digit_ratio > 0.5:
                continue

            fields['merchant'] = line_text
            fields['merchant_confidence'] = line['confidence']
            break

        return fields

    def detect_text_simple(self, image_path: str) -> str:
        """
        Simple text detection (faster, less structured).

        Args:
            image_path: Path to image file

        Returns:
            Extracted text as string
        """
        from google.cloud import vision

        self._ensure_initialized()

        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = self._client.text_detection(image=image)

        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")

        texts = response.text_annotations
        return texts[0].description if texts else ""

    def detect_labels(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect labels/objects in image.

        Args:
            image_path: Path to image file

        Returns:
            List of detected labels with scores
        """
        from google.cloud import vision

        self._ensure_initialized()

        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = self._client.label_detection(image=image)

        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")

        return [
            {
                'description': label.description,
                'score': label.score,
                'topicality': label.topicality
            }
            for label in response.label_annotations
        ]
