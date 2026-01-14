"""
Data Extraction Module
Extracts specific fields from OCR text.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from constants import (
    BUSINESS_KEYWORDS,
    ADDRESS_KEYWORDS,
    KNOWN_BANKS,
    CURRENCY_MAP,
    PAYMENT_METHODS
)

logger = get_logger(__name__)


def extract_business_name(texts: List[str]) -> Optional[str]:
    """
    Extract business name (usually in first lines).

    Args:
        texts: List of OCR text lines

    Returns:
        Business name or None
    """
    for text in texts[:3]:
        # Uppercase and meaningful length
        if len(text) > 3 and text.isupper():
            return text

        # Contains business keywords
        for kw in BUSINESS_KEYWORDS:
            if kw in text.upper():
                return text

    return texts[0] if texts else None


def extract_address(texts: List[str]) -> Optional[str]:
    """
    Extract address information.

    Args:
        texts: List of OCR text lines

    Returns:
        Address string or None
    """
    for text in texts:
        for kw in ADDRESS_KEYWORDS:
            if kw in text.upper():
                return text
    return None


def extract_tax_number(text: str) -> Optional[str]:
    """
    Extract tax number (10-11 digit number).

    Args:
        text: OCR text

    Returns:
        Tax number or None
    """
    patterns = [
        r'V\.?D\.?\s*[:.]?\s*(\d{10,11})',
        r'VN\s*[:.]?\s*(\d{10,11})',
        r'(\d{10,11})'
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def extract_date(text: str) -> Optional[str]:
    """
    Extract date (multiple format support).

    Args:
        text: OCR text

    Returns:
        Date string or None
    """
    patterns = [
        # Labeled formats (TR and EN)
        r'(?:TARIH|DATE|DATUM)\s*[:.]?\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})',
        # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
        r'(\d{2}[/.-]\d{2}[/.-]\d{4})',
        # D/M/YYYY (single digit day/month)
        r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})',
        # OCR error formats: 31/0/1207
        r'(\d{2}[/.-]\d{1}[/.-]\d{4})',
        # YYYY-MM-DD (ISO format)
        r'(\d{4}[/.-]\d{2}[/.-]\d{2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_time(text: str) -> Optional[str]:
    """
    Extract time (multiple format support).

    Args:
        text: OCR text

    Returns:
        Time string or None
    """
    patterns = [
        # Labeled formats (TR and EN)
        r'(?:SAAT|TIME|ZEIT)\s*[:.]?\s*(\d{1,2}[:.;,]\d{2}[:.;,]?\d{0,2})',
        # HH:MM:SS or HH,MM,SS
        r'(\d{2}[:.;,]\d{2}[:.;,]\d{2})',
        # HH:MM
        r'(\d{2}[:.;,]\d{2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result = match.group(1)
            result = re.sub(r'[;,.]', ':', result)
            return result
    return None


def extract_receipt_number(text: str) -> Optional[str]:
    """
    Extract receipt number.

    Args:
        text: OCR text

    Returns:
        Receipt number or None
    """
    patterns = [
        r'FIS\s*NO\s*[:.]?\s*(\d+)',
        r'FISNO\s*[:.]?\s*(\d+)',
        r'Z\s*NO\s*[:.]?\s*(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_amount(text: str, keywords: List[str]) -> Optional[float]:
    """
    Extract amount/total (multi-currency support).

    Args:
        text: OCR text
        keywords: Keywords to search for (e.g., ['TOPLAM', 'TOTAL'])

    Returns:
        Amount as float or None
    """
    for keyword in keywords:
        patterns = [
            # TOTAL € 24,90 or TOTAL: €24.90
            rf'{keyword}\s*[:.]?\s*[€$£₺]?\s*(\d+[.,]\d{{2}})',
            # € 24,90 TOTAL
            rf'[€$£₺]\s*(\d+[.,]\d{{2}})\s*{keyword}',
            # TOTAL *242,00 (Turkish receipts)
            rf'{keyword}\s*[:.]?\s*\*?(\d+[.,]\d{{2}})',
            # TOTAL 242
            rf'{keyword}\s*[:.]?\s*\*?(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '.')
                try:
                    return float(amount_str)
                except ValueError:
                    continue

    # Fallback: look for Euro amounts
    euro_pattern = r'[€]\s*(\d+[.,;:]\d{2})'
    match = re.search(euro_pattern, text)
    if match:
        amount_str = re.sub(r'[,:;]', '.', match.group(1))
        try:
            return float(amount_str)
        except ValueError:
            pass

    return None


def extract_currency(text: str) -> Optional[str]:
    """
    Extract currency code.

    Args:
        text: OCR text

    Returns:
        Currency code (EUR, USD, TRY, etc.) or None
    """
    for symbol, code in CURRENCY_MAP.items():
        if symbol in text:
            return code
    return None


def extract_payment_method(text: str) -> Optional[str]:
    """
    Extract payment method.

    Args:
        text: OCR text

    Returns:
        Payment method string or None
    """
    text_upper = text.upper()
    for key, value in PAYMENT_METHODS.items():
        if key in text_upper:
            return value
    return None


def extract_bank(text: str) -> Optional[str]:
    """
    Extract bank name.

    Args:
        text: OCR text

    Returns:
        Bank name or None
    """
    text_upper = text.upper()
    for bank in KNOWN_BANKS:
        if bank in text_upper:
            return bank.title()
    return None


def extract_card_number(text: str) -> Optional[str]:
    """
    Extract masked card number.

    Args:
        text: OCR text

    Returns:
        Masked card number or None
    """
    patterns = [
        r'\*{4}\s*\*{4}\s*\*{4}\s*(\d{4})',
        r'\*+(\d{4})',
        r'[^\d](\d{4})\s*\n.*SATIS',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"****{match.group(1)}"
    return None


def extract_approval_code(text: str) -> Optional[str]:
    """
    Extract approval/authorization code.

    Args:
        text: OCR text

    Returns:
        Approval code or None
    """
    patterns = [
        r'ONAY\s*KODU?\s*[:.]?\s*(\d{5,6})',
        r'KODU\s*[:.]?\s*\d?\s*(\d{5,6})',
        r'PROV\s*[:.]?\s*(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_transaction_type(text: str, texts: List[str] = None) -> Dict[str, Any]:
    """
    Extract transaction type information.

    Turkish receipt transaction types:
    - SATIS: Normal sale
    - IADE: Return/refund
    - IPTAL: Cancelled transaction
    - ARSIV FATURA / E-ARSIV: Archive invoice
    - BILGI FISI: Info receipt

    Args:
        text: Combined OCR text
        texts: List of OCR text lines

    Returns:
        Dict with type, subtype, document_number, is_invoice
    """
    result = {
        'type': 'SATIS',
        'subtype': None,
        'document_number': None,
        'is_invoice': False
    }

    text_upper = text.upper()

    # Transaction type detection (priority order)
    transaction_patterns = [
        (r'IADE|REFUND|RETURN|GERI\s*ODEME', 'IADE'),
        (r'IPTAL|VOID|CANCEL', 'IPTAL'),
        (r'E[-\s]*ARSIV|E[-\s]*FATURA', 'E-ARSIV'),
        (r'ARSIV\s*FATURA|ARSIV\s*BELGE', 'ARSIV_FATURA'),
        (r'BILGI\s*FISI|INFO\s*RECEIPT', 'BILGI_FISI'),
        (r'IRSALIYE|DELIVERY\s*NOTE', 'IRSALIYE'),
        (r'FATURA|INVOICE', 'FATURA'),
    ]

    for pattern, trans_type in transaction_patterns:
        if re.search(pattern, text_upper):
            result['type'] = trans_type
            break

    # Subtype detection
    subtype_match = re.search(r'TUR\s*[:.]?\s*([A-Z\s]+?)(?:\n|$|FATURA)', text_upper)
    if subtype_match:
        subtype = subtype_match.group(1).strip()
        if subtype and len(subtype) > 2:
            result['subtype'] = subtype

    # Document number detection
    doc_patterns = [
        r'(?:FATURA|IRSALIYE).*?(?:SERI|SIRA)\s*[:.]?\s*(\d+)',
        r'BELGE\s*NO\s*[:.]?\s*(\d+)',
        r'FATURA\s*NO\s*[:.]?\s*(\d+)',
        r'E[-\s]*ARSIV\s*NO\s*[:.]?\s*(\d+)',
    ]

    for pattern in doc_patterns:
        match = re.search(pattern, text_upper)
        if match:
            result['document_number'] = match.group(1)
            break

    # Invoice check
    if any(t in result['type'] for t in ['FATURA', 'ARSIV', 'E-ARSIV']):
        result['is_invoice'] = True

    # "IRSALIYE YERINE GECER" check
    if re.search(r'IRSALIYE\s*YERINE\s*GECER', text_upper):
        result['subtype'] = result.get('subtype') or 'IRSALIYE_YERINE_GECER'

    return result
