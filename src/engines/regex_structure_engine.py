"""
Regex Structure Engine
Extracts structured data from OCR text using regular expressions.
Lightweight alternative to Claude API-based StructureEngine.
"""

import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class RegexStructuredReceipt:
    """Structured receipt data extracted via regex."""
    merchant_name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    total_amount: Optional[float] = None
    tax_details: Dict[str, Any] = field(default_factory=dict)
    items: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "merchant_name": self.merchant_name,
            "date": self.date,
            "time": self.time,
            "total_amount": self.total_amount,
            "tax_details": self.tax_details,
            "items": self.items
        }


class RegexStructureEngine:
    """
    Extracts structured data from OCR text using regex patterns.

    Handles common OCR errors like:
    - T0PLAM -> TOPLAM
    - TUTAP -> TUTAR
    - O/0 confusion
    """

    # OCR error corrections
    OCR_CORRECTIONS = {
        'T0PLAM': 'TOPLAM',
        'TOPLA M': 'TOPLAM',
        'TOPL AM': 'TOPLAM',
        'T OPLAM': 'TOPLAM',
        'TOPIAM': 'TOPLAM',
        'TUTAP': 'TUTAR',
        'TUTAF': 'TUTAR',
        'TUTARI': 'TUTAR',
        'KOV': 'KDV',
        'K0V': 'KDV',
        'TOPKDV': 'TOP KDV',
        'TAR1H': 'TARIH',
        'TARlH': 'TARIH',
        'TARLH': 'TARIH',
        'HAKİT': 'NAKİT',
        'HAKIT': 'NAKIT',
        'NAKlT': 'NAKIT',
    }

    def __init__(self):
        """Initialize the regex structure engine."""
        pass

    def _fix_ocr_errors(self, text: str) -> str:
        """Apply common OCR error corrections."""
        fixed = text
        for wrong, correct in self.OCR_CORRECTIONS.items():
            fixed = fixed.replace(wrong, correct)
        return fixed

    def _parse_amount(self, amount_str: str) -> Optional[float]:
        """Parse amount string to float, handling Turkish number format."""
        if not amount_str:
            return None

        # Clean the string
        cleaned = amount_str.strip()

        # Remove common prefixes/suffixes
        cleaned = cleaned.replace('*', '').replace('#', '').replace('■', '')
        cleaned = cleaned.replace('+', '').replace('-', '')
        cleaned = cleaned.replace('TL', '').replace('₺', '').replace('€', '')

        # Handle space in numbers: "2.040 00" -> "2.040,00" or "2040.00"
        # Pattern: digits, dot, digits, space, digits (Turkish format with space)
        space_match = re.match(r'([\d.]+)\s+(\d{2})$', cleaned)
        if space_match:
            cleaned = space_match.group(1) + ',' + space_match.group(2)

        # Remove remaining spaces
        cleaned = cleaned.replace(' ', '')

        # Handle Turkish format: 1.234,56 -> 1234.56
        if ',' in cleaned and '.' in cleaned:
            # Turkish format: dots as thousands separator, comma as decimal
            cleaned = cleaned.replace('.', '').replace(',', '.')
        elif ',' in cleaned:
            # Only comma: could be decimal separator
            cleaned = cleaned.replace(',', '.')
        elif cleaned.count('.') > 1:
            # Multiple dots: thousand separators (e.g., 1.234.567)
            parts = cleaned.split('.')
            if len(parts[-1]) == 2:
                # Last part is decimal
                cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
            else:
                # All dots are thousand separators
                cleaned = cleaned.replace('.', '')

        try:
            return float(cleaned)
        except ValueError:
            return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date from text."""
        patterns = [
            r'(\d{2}[./-]\d{2}[./-]\d{4})',  # DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY
            r'(\d{2}[./-]\d{2}[./-]\d{2})',  # DD.MM.YY, DD/MM/YY, DD-MM-YY
            r'TARİH[:\s]*(\d{2}[./-]\d{2}[./-]\d{2,4})',
            r'TARIH[:\s]*(\d{2}[./-]\d{2}[./-]\d{2,4})',
            r'DATE[:\s]*(\d{2}[./-]\d{2}[./-]\d{2,4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # Validate: skip invalid dates like 80/03/36
                parts = re.split(r'[./-]', date_str)
                if len(parts) == 3:
                    day, month = int(parts[0]), int(parts[1])
                    if day > 31 or month > 12:
                        continue  # Invalid date, try next pattern
                return date_str
        return None

    def _extract_time(self, text: str) -> Optional[str]:
        """Extract time from text."""
        patterns = [
            r'(\d{2}:\d{2}:\d{2})',  # HH:MM:SS
            r'(\d{2}:\d{2})',        # HH:MM
            r'SAAT[:\s]*(\d{2}:\d{2})',
            r'SAAT[:\s]*(\d{2})\s+(\d{2})',  # SAAT 14 33 (space separated)
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    # Space separated format: combine as HH:MM
                    return f"{groups[0]}:{groups[1]}"
                return match.group(1)
        return None

    def _extract_total(self, text: str) -> Optional[float]:
        """Extract total amount from text."""
        text_upper = text.upper()
        lines = text_upper.split('\n')

        # Direct patterns - amount on same line
        direct_patterns = [
            # TOPLAM followed by amount (same line)
            r'TOPLAM[\s]*[:\.]?\s*[*#■+-]?\s*([\d.,\s]+)',
            # GENEL TOPLAM
            r'GENEL\s*TOPLAM[\s]*[:\.]?\s*[*#■+-]?\s*([\d.,\s]+)',
            # TOTAL (international)
            r'TOTAL[\s]*[:\.]?\s*[€$£₺]?\s*([\d.,\s]+)',
            # TUTAR
            r'TUTAR[\s]*[:\.]?\s*[*#■+-]?\s*([\d.,\s]+)',
            # NAKİT or KART payment
            r'(?:NAKİT|NAKIT|KART|KREDI)[:\s]*([\d.,\s]+)',
            # Amount followed by TL
            r'([\d.,]+)\s*TL',
        ]

        for pattern in direct_patterns:
            match = re.search(pattern, text_upper)
            if match:
                amount = self._parse_amount(match.group(1))
                if amount and amount > 0:
                    return amount

        # Multi-line search: find TOPLAM/NAKIT line and get value from nearby lines
        keywords = ['TOPLAM', 'NAKIT', 'NAKİT', 'TOTAL', 'TUTAR']

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            for keyword in keywords:
                if keyword in line_stripped:
                    # Check if amount is on same line after keyword
                    after_keyword = line_stripped.split(keyword)[-1]
                    amount = self._parse_amount(after_keyword)
                    if amount and amount > 0:
                        return amount

                    # Check next 6 lines for amounts (values can be several lines below)
                    for j in range(1, 7):
                        if i + j < len(lines):
                            next_line = lines[i + j].strip()
                            # Skip empty lines and keyword-only lines
                            if not next_line or len(next_line) < 2:
                                continue
                            if any(k in next_line for k in keywords + ['EKU', 'FIS', 'NO']):
                                continue
                            # Try to parse the line as amount
                            amount = self._parse_amount(next_line)
                            if amount and amount > 0:
                                return amount
                            # Try to find amount pattern in line
                            amount_match = re.search(r'[*#+-]?\s*([\d.,\s]+)', next_line)
                            if amount_match:
                                amount = self._parse_amount(amount_match.group(1))
                                if amount and amount > 0:
                                    return amount

        return None

    def _extract_kdv(self, text: str) -> Dict[str, Any]:
        """Extract KDV (VAT) details from text."""
        text_upper = text.upper()
        kdv_details = {}

        # KDV percentage and amount
        patterns = [
            r'KDV\s*%?\s*(\d+)[:\s]*([\d.,]+)',  # KDV %20: 123.45
            r'%(\d+)\s*KDV[:\s]*([\d.,]+)',      # %20 KDV: 123.45
            r'KDV\s*TUTAR[I]?[:\s]*([\d.,]+)',   # KDV TUTARI: 123.45
            r'TOPKDV[:\s]*([\d.,]+)',            # TOPKDV: 123.45
        ]

        for pattern in patterns:
            match = re.search(pattern, text_upper)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    kdv_details['kdv_orani'] = int(groups[0])
                    kdv_details['kdv_tutari'] = self._parse_amount(groups[1])
                elif len(groups) == 1:
                    kdv_details['kdv_tutari'] = self._parse_amount(groups[0])
                break

        # KDV matrah (tax base)
        matrah_patterns = [
            r'MATRAH[:\s]*([\d.,]+)',
            r'KDV\s*MATRAH[I]?[:\s]*([\d.,]+)',
        ]

        for pattern in matrah_patterns:
            match = re.search(pattern, text_upper)
            if match:
                kdv_details['kdv_matrah'] = self._parse_amount(match.group(1))
                break

        return kdv_details

    def _extract_merchant(self, text: str, lines: List[str]) -> Optional[str]:
        """Extract merchant name from text."""
        # Blacklist words that are not merchant names
        blacklist = [
            'SERI', 'SIRA', 'NO:', 'NO.', 'ABS', 'FIS', 'FİŞ',
            'TARIH', 'TARİH', 'SAAT', 'VKN', 'TCKN', 'PERAKENDE',
            'SATIS', 'SATIŞ', 'E-ARŞİV', 'E-ARSIV', 'FATURA',
            'BILGI', 'BİLGİ', 'ETTN', 'MUSTERI', 'MÜŞTERİ',
            'ADRES', 'TEL', 'TOPLAM', 'KDV', 'TUTAR'
        ]

        for line in lines[:10]:  # Check first 10 lines
            line_clean = line.strip()
            line_upper = line_clean.upper()

            # Skip short lines
            if len(line_clean) < 3:
                continue

            # Skip lines with blacklisted words
            if any(bl in line_upper for bl in blacklist):
                continue

            # Skip lines that are mostly numbers
            digit_ratio = sum(c.isdigit() for c in line_clean) / len(line_clean) if line_clean else 1
            if digit_ratio > 0.5:
                continue

            # Skip lines starting with special characters
            if line_clean[0] in '*#■-=':
                continue

            return line_clean

        return None

    def extract(self, ocr_text: str) -> RegexStructuredReceipt:
        """
        Extract structured data from OCR text.

        Args:
            ocr_text: Raw text from OCR engine

        Returns:
            RegexStructuredReceipt with extracted fields
        """
        # Fix common OCR errors
        fixed_text = self._fix_ocr_errors(ocr_text)
        lines = [l.strip() for l in fixed_text.split('\n') if l.strip()]

        return RegexStructuredReceipt(
            merchant_name=self._extract_merchant(fixed_text, lines),
            date=self._extract_date(fixed_text),
            time=self._extract_time(fixed_text),
            total_amount=self._extract_total(fixed_text),
            tax_details=self._extract_kdv(fixed_text),
            raw_text=ocr_text
        )

    def process_vision_result(self, vision_result) -> RegexStructuredReceipt:
        """
        Process GoogleVisionEngine result directly.

        Args:
            vision_result: OCRResult from GoogleVisionEngine

        Returns:
            RegexStructuredReceipt with extracted fields
        """
        return self.extract(vision_result.raw_text)


def extract_with_regex(ocr_text: str) -> Dict[str, Any]:
    """
    Convenience function to extract structured data using regex.

    Args:
        ocr_text: Raw text from OCR

    Returns:
        Dictionary with extracted fields
    """
    engine = RegexStructureEngine()
    result = engine.extract(ocr_text)
    return result.to_dict()
