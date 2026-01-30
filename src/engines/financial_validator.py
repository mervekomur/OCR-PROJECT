"""
Financial Validator
Validates and corrects OCR results using mathematical consistency checks.
Handles common OCR character confusions in financial documents.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of financial validation."""
    is_valid: bool = False
    original_total: Optional[float] = None
    corrected_total: Optional[float] = None
    original_date: Optional[str] = None
    corrected_date: Optional[str] = None
    tax_valid: bool = False
    corrections_made: List[str] = field(default_factory=list)
    confidence_score: float = 0.0


class FinancialValidator:
    """
    Validates OCR results using financial logic and fixes common errors.

    Common OCR confusions in financial documents:
    - 0 <-> O, 8 <-> B, 1 <-> I/l/7, 5 <-> S, 6 <-> G
    - Decimal separators: . vs ,
    - Thousand separators confusion
    """

    # Character substitution maps for financial context
    DIGIT_CORRECTIONS = {
        'O': '0', 'o': '0',
        'I': '1', 'l': '1', 'i': '1',
        'Z': '2', 'z': '2',
        'E': '3',
        'A': '4',
        'S': '5', 's': '5',
        'G': '6', 'b': '6',
        'T': '7', 't': '7',
        'B': '8',
        'g': '9', 'q': '9',
    }

    # Common Turkish KDV rates
    VALID_KDV_RATES = [1, 8, 10, 18, 20]

    def __init__(self, tolerance: float = 0.02):
        """
        Initialize validator.

        Args:
            tolerance: Acceptable difference ratio for validation (default 2%)
        """
        self.tolerance = tolerance

    def _fix_digit_string(self, text: str) -> str:
        """Convert a string that should be numeric, fixing OCR errors."""
        if not text:
            return text

        result = []
        for char in text:
            if char in self.DIGIT_CORRECTIONS:
                result.append(self.DIGIT_CORRECTIONS[char])
            elif char.isdigit() or char in '.,- ':
                result.append(char)
            # Skip other characters

        return ''.join(result)

    def _parse_turkish_amount(self, amount_str: str) -> Optional[float]:
        """Parse Turkish formatted amount string."""
        if not amount_str:
            return None

        # Fix potential OCR errors in digits
        cleaned = self._fix_digit_string(amount_str)

        # Remove spaces and special chars
        cleaned = cleaned.replace(' ', '').replace('-', '')

        # Handle Turkish format: 1.234,56 -> 1234.56
        if ',' in cleaned and '.' in cleaned:
            cleaned = cleaned.replace('.', '').replace(',', '.')
        elif ',' in cleaned:
            cleaned = cleaned.replace(',', '.')
        elif cleaned.count('.') > 1:
            # Multiple dots: thousand separators
            parts = cleaned.split('.')
            if len(parts[-1]) == 2:
                cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
            else:
                cleaned = cleaned.replace('.', '')

        try:
            return float(cleaned)
        except ValueError:
            return None

    def _validate_kdv_calculation(
        self,
        total: Optional[float],
        kdv_rate: Optional[int],
        kdv_amount: Optional[float],
        matrah: Optional[float]
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate KDV calculation and suggest corrections.

        Formula: matrah * (kdv_rate/100) = kdv_amount
                 matrah + kdv_amount = total
        """
        corrections = {}

        if total is None:
            return False, corrections

        # If we have KDV rate and total, calculate expected values
        if kdv_rate and kdv_rate in self.VALID_KDV_RATES:
            rate_decimal = kdv_rate / 100

            # Calculate expected matrah from total
            expected_matrah = total / (1 + rate_decimal)
            expected_kdv = total - expected_matrah

            corrections['calculated_matrah'] = round(expected_matrah, 2)
            corrections['calculated_kdv'] = round(expected_kdv, 2)

            # Validate if provided values match
            if matrah:
                matrah_diff = abs(matrah - expected_matrah) / expected_matrah
                if matrah_diff <= self.tolerance:
                    return True, corrections

            if kdv_amount:
                kdv_diff = abs(kdv_amount - expected_kdv) / expected_kdv if expected_kdv > 0 else 1
                if kdv_diff <= self.tolerance:
                    return True, corrections

        # Try to infer KDV rate from values
        if matrah and total and total > matrah:
            implied_kdv = total - matrah
            implied_rate = (implied_kdv / matrah) * 100

            # Find closest valid rate
            closest_rate = min(self.VALID_KDV_RATES, key=lambda r: abs(r - implied_rate))
            if abs(closest_rate - implied_rate) <= 2:  # Within 2% tolerance
                corrections['inferred_kdv_rate'] = closest_rate
                corrections['inferred_kdv_amount'] = round(implied_kdv, 2)
                return True, corrections

        return False, corrections

    def _validate_date(self, date_str: Optional[str]) -> Tuple[bool, Optional[str]]:
        """Validate and fix date string."""
        if not date_str:
            return False, None

        # Detect separator
        separator = '.' if '.' in date_str else ('/' if '/' in date_str else '-')

        # Fix digit OCR errors while preserving separators
        fixed_parts = []
        for part in re.split(r'[./-]', date_str):
            fixed_part = self._fix_digit_string(part)
            fixed_parts.append(fixed_part)

        if len(fixed_parts) != 3:
            return False, date_str

        try:
            day = int(fixed_parts[0]) if fixed_parts[0] else 0
            month = int(fixed_parts[1]) if fixed_parts[1] else 0
            year_str = fixed_parts[2]

            corrections_made = False

            # Day validation (1-31)
            if day > 31:
                # Try common fixes: 81 -> 01 (8 misread as 0), 87 -> 07
                if day >= 80 and day <= 89:
                    day = day - 80
                    corrections_made = True
                elif day >= 70 and day <= 79:
                    day = day - 70
                    corrections_made = True
                elif day >= 40 and day <= 49:
                    day = day - 40
                    corrections_made = True

            # Month validation (1-12)
            if month > 12:
                if month >= 80 and month <= 89:
                    month = month - 80
                    corrections_made = True
                elif month >= 70 and month <= 79:
                    month = month - 70
                    corrections_made = True

            # Year validation
            year_int = int(year_str) if year_str else 0

            # Handle 2-digit year
            if len(year_str) == 2:
                if year_int >= 80:
                    # 80-99 likely means 1980-1999 but could be OCR error
                    # 80 could be misread 00, so try 2000
                    year_int = 2000 + (year_int - 80) if year_int >= 80 else 1900 + year_int
                    corrections_made = True
                elif year_int > 30:
                    year_int = 1900 + year_int
                else:
                    year_int = 2000 + year_int

            # Handle obviously wrong years
            if year_int < 1990 or year_int > 2030:
                # Try to fix: 2824 -> 2024
                if 2800 <= year_int <= 2899:
                    year_int = year_int - 800
                    corrections_made = True
                elif 2700 <= year_int <= 2799:
                    year_int = year_int - 700
                    corrections_made = True

            # Final sanity check
            if 1 <= day <= 31 and 1 <= month <= 12 and 1990 <= year_int <= 2030:
                corrected = f"{day:02d}{separator}{month:02d}{separator}{year_int}"
                return True, corrected
            else:
                # Return original if can't fix
                return False, date_str

        except (ValueError, IndexError):
            pass

        return False, date_str

    def _find_amounts_in_text(self, text: str) -> List[float]:
        """Find all potential amounts in text."""
        amounts = []

        # Pattern for Turkish/European amounts
        patterns = [
            r'[\d]{1,3}(?:[.,]\d{3})*[.,]\d{2}',  # 1.234,56 or 1,234.56
            r'[\d]+[.,]\d{2}',                      # 123,45 or 123.45
            r'[\d]{1,3}(?:\s\d{3})+[.,]\d{2}',     # 1 234,56
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                amount = self._parse_turkish_amount(match)
                if amount and amount > 0:
                    amounts.append(amount)

        return sorted(set(amounts), reverse=True)

    def _validate_total_by_items(
        self,
        total: Optional[float],
        raw_text: str
    ) -> Tuple[bool, Optional[float]]:
        """
        Try to validate total by summing item prices.
        """
        if not total:
            return False, None

        amounts = self._find_amounts_in_text(raw_text)

        if len(amounts) < 2:
            return False, total

        # Check if total is the largest amount
        if amounts and total >= max(amounts) * 0.9:
            return True, total

        # Try to find amounts that sum to total
        for i, amt in enumerate(amounts):
            if abs(amt - total) / total <= self.tolerance:
                return True, total

        return False, total

    def validate_and_correct(
        self,
        structured_data: Dict[str, Any],
        raw_text: str
    ) -> ValidationResult:
        """
        Validate structured data and apply corrections.

        Args:
            structured_data: Extracted receipt data
            raw_text: Original OCR text for re-analysis

        Returns:
            ValidationResult with corrections
        """
        result = ValidationResult()
        result.original_total = structured_data.get('total_amount')
        result.original_date = structured_data.get('date')

        corrections = []

        # 1. Validate and fix date
        date_valid, corrected_date = self._validate_date(result.original_date)
        result.corrected_date = corrected_date
        if corrected_date and corrected_date != result.original_date:
            corrections.append(f"Date: {result.original_date} -> {corrected_date}")

        # 2. Get tax details
        tax = structured_data.get('tax_details', {})
        kdv_rate = tax.get('kdv_orani')
        kdv_amount = tax.get('kdv_tutari')
        matrah = tax.get('kdv_matrah')

        # 3. Validate KDV calculation
        result.tax_valid, tax_corrections = self._validate_kdv_calculation(
            result.original_total, kdv_rate, kdv_amount, matrah
        )

        if tax_corrections:
            for key, value in tax_corrections.items():
                corrections.append(f"Tax {key}: {value}")

        # 4. Validate total
        total_valid, corrected_total = self._validate_total_by_items(
            result.original_total, raw_text
        )
        result.corrected_total = corrected_total or result.original_total

        # 5. If no total found, try to extract from amounts
        if result.corrected_total is None:
            amounts = self._find_amounts_in_text(raw_text)
            if amounts:
                # Largest amount is likely the total
                result.corrected_total = amounts[0]
                corrections.append(f"Total inferred from text: {amounts[0]}")

        # Calculate confidence
        confidence_factors = []
        if date_valid:
            confidence_factors.append(0.3)
        if result.corrected_total:
            confidence_factors.append(0.4)
        if result.tax_valid:
            confidence_factors.append(0.3)

        result.confidence_score = sum(confidence_factors)
        result.is_valid = result.confidence_score >= 0.4
        result.corrections_made = corrections

        return result

    def correct_amount_string(self, text: str) -> str:
        """
        Correct OCR errors in an amount string.

        Examples:
            "1.S34,S6" -> "1.534,56"
            "2.O4O,OO" -> "2.040,00"
        """
        return self._fix_digit_string(text)


def validate_receipt(structured_data: Dict[str, Any], raw_text: str) -> ValidationResult:
    """
    Convenience function to validate receipt data.

    Args:
        structured_data: Dict with total_amount, date, tax_details
        raw_text: Original OCR text

    Returns:
        ValidationResult with corrections
    """
    validator = FinancialValidator()
    return validator.validate_and_correct(structured_data, raw_text)
