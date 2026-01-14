"""
Base OCR Engine Interface
Abstract base class for all OCR engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import time


@dataclass
class OCRResult:
    """Standardized OCR result structure."""

    # Raw extracted text
    raw_text: str = ""

    # List of text lines with metadata
    lines: List[Dict[str, Any]] = field(default_factory=list)

    # Overall confidence score (0.0 - 1.0)
    confidence: float = 0.0

    # Extracted fields (date, total, merchant, etc.)
    fields: Dict[str, Any] = field(default_factory=dict)

    # Processing time in seconds
    processing_time: float = 0.0

    # Engine name that produced this result
    engine_name: str = ""

    # Engine-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'raw_text': self.raw_text,
            'lines': self.lines,
            'confidence': self.confidence,
            'fields': self.fields,
            'processing_time': self.processing_time,
            'engine_name': self.engine_name,
            'metadata': self.metadata
        }


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""

    # Engine name for identification
    name: str = "base"

    # Engine description
    description: str = "Base OCR Engine"

    # Whether the engine is available (dependencies installed)
    _available: Optional[bool] = None

    def __init__(self):
        """Initialize the engine."""
        self._model = None
        self._initialized = False

    @classmethod
    def is_available(cls) -> bool:
        """Check if the engine dependencies are available."""
        if cls._available is None:
            cls._available = cls._check_availability()
        return cls._available

    @classmethod
    @abstractmethod
    def _check_availability(cls) -> bool:
        """Check if required packages are installed."""
        pass

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the model (lazy loading)."""
        pass

    def _ensure_initialized(self) -> None:
        """Ensure the model is initialized."""
        if not self._initialized:
            self._initialize()
            self._initialized = True

    @abstractmethod
    def _extract_text(self, image_path: str) -> OCRResult:
        """
        Extract text from image (internal implementation).

        Args:
            image_path: Path to image file

        Returns:
            OCRResult with extracted data
        """
        pass

    def extract(self, image_path: str) -> OCRResult:
        """
        Extract text from image with timing.

        Args:
            image_path: Path to image file

        Returns:
            OCRResult with extracted data and timing
        """
        # Validate path
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Ensure model is loaded
        self._ensure_initialized()

        # Process with timing
        start_time = time.time()
        result = self._extract_text(str(path))
        result.processing_time = time.time() - start_time
        result.engine_name = self.name

        return result

    def extract_fields(self, image_path: str) -> Dict[str, Any]:
        """
        Extract specific fields from receipt.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with extracted fields
        """
        result = self.extract(image_path)
        return result.fields

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', available={self.is_available()})"
