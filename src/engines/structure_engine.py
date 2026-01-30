"""
Structure Engine - Claude API Integration
Converts raw OCR text to structured JSON using Claude's intelligence.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class StructuredReceipt:
    """Structured receipt data extracted by Claude."""
    merchant_name: Optional[str] = None
    date: Optional[str] = None
    total_amount: Optional[float] = None
    tax_details: Dict[str, Any] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "merchant_name": self.merchant_name,
            "date": self.date,
            "total_amount": self.total_amount,
            "tax_details": self.tax_details
        }


class StructureEngine:
    """
    Uses Claude API to extract structured data from OCR text.

    Handles OCR errors intelligently by understanding context.
    """

    SYSTEM_PROMPT = """Sen bir fiş/fatura analiz uzmanısın. Sana verilen OCR metninden yapılandırılmış veri çıkaracaksın.

ÖNEMLİ KURALLAR:
1. OCR hataları olabilir. Örneğin:
   - "T0PLAM" aslında "TOPLAM" demek
   - "l" harfi "1" rakamı olarak okunmuş olabilir
   - "O" harfi "0" rakamı olarak okunmuş olabilir
   - "TUTAP" aslında "TUTAR" olabilir
   - Türkçe karakterler bozulmuş olabilir (ı->i, ş->s, ğ->g vb.)

2. Bağlamı kullanarak doğru değerleri çıkar:
   - Fiyatlar genellikle "X.XXX,XX" veya "X,XXX.XX" formatında
   - Tarihler "DD.MM.YYYY" veya "DD/MM/YYYY" formatında
   - KDV oranları genellikle %1, %10, %20 gibi standart değerler

3. tax_details içinde şunları çıkar (varsa):
   - kdv_orani: KDV yüzdesi
   - kdv_tutari: KDV miktarı
   - kdv_matrah: KDV matrahı (vergi öncesi tutar)

4. Eğer bir değeri bulamıyorsan null olarak bırak, tahmin etme.

SADECE JSON FORMATINDA YANIT VER, başka hiçbir açıklama ekleme."""

    USER_PROMPT_TEMPLATE = """Aşağıdaki fiş/fatura OCR metnini analiz et ve yapılandırılmış veri çıkar:

---
{ocr_text}
---

Yanıtını SADECE şu JSON formatında ver:
{{
    "merchant_name": "İşletme adı veya null",
    "date": "Tarih (DD.MM.YYYY formatında) veya null",
    "total_amount": Toplam tutar (sayı olarak, örn: 1234.56) veya null,
    "tax_details": {{
        "kdv_orani": KDV yüzdesi veya null,
        "kdv_tutari": KDV tutarı veya null,
        "kdv_matrah": KDV matrahı veya null
    }}
}}"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize Structure Engine.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Claude model to use (default: claude-sonnet-4-20250514)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if anthropic package is installed."""
        try:
            import anthropic
            return True
        except ImportError:
            return False

    def _ensure_client(self):
        """Initialize Anthropic client if needed."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key not found. "
                    "Set ANTHROPIC_API_KEY environment variable "
                    "or provide api_key parameter."
                )
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)

    def extract_structured_data(self, ocr_text: str) -> StructuredReceipt:
        """
        Extract structured data from OCR text using Claude.

        Args:
            ocr_text: Raw text from OCR engine

        Returns:
            StructuredReceipt with extracted fields
        """
        self._ensure_client()

        user_prompt = self.USER_PROMPT_TEMPLATE.format(ocr_text=ocr_text)

        message = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        # Extract response text
        response_text = message.content[0].text.strip()

        # Parse JSON response
        try:
            # Clean up response if needed (remove markdown code blocks)
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return empty result with raw response
            return StructuredReceipt(
                raw_response={"error": str(e), "raw_text": response_text}
            )

        # Build structured receipt
        tax_details = data.get("tax_details", {})
        if tax_details is None:
            tax_details = {}

        return StructuredReceipt(
            merchant_name=data.get("merchant_name"),
            date=data.get("date"),
            total_amount=data.get("total_amount"),
            tax_details=tax_details,
            raw_response=data
        )

    def process_vision_result(self, vision_result) -> StructuredReceipt:
        """
        Process GoogleVisionEngine result directly.

        Args:
            vision_result: OCRResult from GoogleVisionEngine

        Returns:
            StructuredReceipt with extracted fields
        """
        return self.extract_structured_data(vision_result.raw_text)


def extract_receipt_data(ocr_text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to extract structured data from OCR text.

    Args:
        ocr_text: Raw text from OCR
        api_key: Optional Anthropic API key

    Returns:
        Dictionary with extracted fields

    Example:
        >>> from src.engines.structure_engine import extract_receipt_data
        >>> data = extract_receipt_data(ocr_result.raw_text)
        >>> print(data["merchant_name"])
        >>> print(data["total_amount"])
    """
    engine = StructureEngine(api_key=api_key)
    result = engine.extract_structured_data(ocr_text)
    return result.to_dict()
