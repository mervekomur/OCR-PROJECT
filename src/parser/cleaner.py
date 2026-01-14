"""
OCR Text Cleaning Module
Fixes common OCR errors and normalizes text.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from constants import TAX_CONTEXT_KEYWORDS, DEFAULT_YEAR, MIN_VALID_YEAR, MAX_VALID_YEAR

# Try to load config, fallback to empty dict if not available
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'config'))
    from config import get_business_name_corrections, get_product_name_corrections
except ImportError:
    def get_business_name_corrections():
        return {}
    def get_product_name_corrections():
        return {}

logger = get_logger(__name__)


def clean_ocr_text(text: str) -> str:
    """
    Fix common OCR errors in text.

    Args:
        text: Raw OCR text

    Returns:
        Cleaned text
    """
    if not text:
        return text

    cleaned = text

    # Euro symbol variations
    cleaned = re.sub(r'[Є∈]', '€', cleaned)

    # Remove extra spaces
    cleaned = re.sub(r' +', ' ', cleaned)

    # Fix Z/% confusion
    cleaned = fix_z_percent_ocr_error(cleaned)

    return cleaned


def fix_z_percent_ocr_error(text: str) -> str:
    """
    Fix OCR confusion between Z and % characters.

    Heuristic Rules:
    1. Tax context: Z + number -> % + number (Z20 -> %20)
    2. Product code context: preserve as-is

    Args:
        text: Raw OCR text

    Returns:
        Corrected text
    """
    if not text:
        return text

    corrected = text
    text_upper = text.upper()

    # Check for tax context
    is_tax_context = any(kw in text_upper for kw in TAX_CONTEXT_KEYWORDS)

    if is_tax_context:
        old_corrected = corrected
        corrected = re.sub(r'Z\s*(\d+[.,]?\d*)', r'%\1', corrected, flags=re.IGNORECASE)
        if corrected != old_corrected:
            logger.debug(f"Z->%% correction (tax context): {text} -> {corrected}")

    # Special case: standalone Z + number at line start
    corrected = re.sub(
        r'(?<![A-Za-z])Z(\d+[.,]?\d*)\s*(%|KDV|TVA|TAX)',
        r'%\1 \2',
        corrected,
        flags=re.IGNORECASE
    )

    # Standalone Z + number (likely a rate)
    if re.match(r'^Z\d+[.,]?\d*$', corrected.strip(), re.IGNORECASE):
        old_corrected = corrected
        corrected = re.sub(r'^Z(\d+[.,]?\d*)$', r'%\1', corrected.strip(), flags=re.IGNORECASE)
        if corrected != old_corrected.strip():
            logger.debug(f"Z->%% correction (standalone): {text} -> {corrected}")

    return corrected


def clean_business_name(name: str) -> str:
    """
    Fix OCR errors in business names.

    Args:
        name: Raw business name

    Returns:
        Cleaned business name
    """
    if not name:
        return name

    cleaned = name

    # Load corrections from config
    ocr_fixes = get_business_name_corrections()

    for wrong, correct in ocr_fixes.items():
        cleaned = cleaned.replace(wrong, correct)

    # Fix single-letter splits (e.g., "H i l t o n" -> "Hilton")
    cleaned = re.sub(r'\b(\w) (\w) (\w)\b', r'\1\2\3', cleaned)
    cleaned = re.sub(r'\b(\w) (\w)\b', r'\1\2', cleaned)

    # Remove extra spaces
    cleaned = re.sub(r' +', ' ', cleaned).strip()

    if cleaned != name:
        logger.debug(f"Business name corrected: {name} -> {cleaned}")

    return cleaned


def clean_product_name(name: str) -> str:
    """
    Fix OCR errors in product names.

    Args:
        name: Raw product name

    Returns:
        Cleaned product name
    """
    if not name:
        return name

    cleaned = name

    # Load corrections from config
    ocr_fixes = get_product_name_corrections()

    for wrong, correct in ocr_fixes.items():
        if cleaned == wrong:
            cleaned = correct
            break

    # Title case for all-upper or all-lower names
    if cleaned.isupper() or cleaned.islower():
        cleaned = cleaned.title()

    # Remove extra spaces
    cleaned = re.sub(r' +', ' ', cleaned).strip()

    if cleaned != name:
        logger.debug(f"Product name corrected: {name} -> {cleaned}")

    return cleaned


def clean_date_string(date_str: str) -> Dict[str, Any]:
    """
    Fix errors in date strings.

    Args:
        date_str: Raw date string

    Returns:
        Dict with cleaned date and correction metadata
    """
    result = {
        'cleaned': date_str,
        'original': date_str,
        'correction_applied': False
    }

    if not date_str:
        return result

    cleaned = date_str
    original = date_str
    corrected = False

    # Parse date parts
    parts = re.split(r'[/.-]', cleaned)
    if len(parts) == 3:
        day, month, year = parts

        # Fix single-digit day
        if len(day) == 1:
            day = '0' + day
            corrected = True

        # Fix invalid day (>31)
        try:
            if int(day) > 31:
                day = day[-2:]
                corrected = True
        except ValueError:
            pass

        # Fix single-digit month
        if len(month) == 1:
            month = '0' + month
            corrected = True

        # Fix invalid month (0 or >12)
        if month in ('0', '00'):
            month = '07'  # Default to July
            corrected = True

        try:
            if int(month) > 12:
                month = month[-2:] if len(month) > 2 else month[-1].zfill(2)
                corrected = True
        except ValueError:
            pass

        # Fix year
        try:
            year_int = int(year)
            if len(year) == 4:
                if year_int < MIN_VALID_YEAR or year_int > MAX_VALID_YEAR:
                    year = DEFAULT_YEAR
                    corrected = True
            elif len(year) == 3:
                year = DEFAULT_YEAR
                corrected = True
            elif len(year) == 2:
                year = '20' + year
                if int(year) < MIN_VALID_YEAR:
                    year = DEFAULT_YEAR
                corrected = True
        except ValueError:
            year = DEFAULT_YEAR
            corrected = True

        cleaned = f"{day}/{month}/{year}"

        if corrected:
            logger.debug(f"Date corrected: {original} -> {cleaned}")

    return {
        'cleaned': cleaned,
        'original': original,
        'correction_applied': corrected
    }


def clean_time_string(time_str: str) -> Dict[str, Any]:
    """
    Fix errors in time strings.

    Args:
        time_str: Raw time string

    Returns:
        Dict with cleaned time and correction metadata
    """
    result = {
        'cleaned': time_str,
        'original': time_str,
        'correction_applied': False
    }

    if not time_str:
        return result

    # Normalize separators to colon
    cleaned = re.sub(r'[,;.]', ':', time_str)
    cleaned = re.sub(r':+', ':', cleaned)

    original_normalized = cleaned

    parts = cleaned.split(':')
    if len(parts) >= 2:
        try:
            corrected = False

            # Check seconds (if present)
            if len(parts) >= 3 and parts[2].isdigit():
                second = int(parts[2])
                if second >= 60:
                    logger.debug(f"Invalid second value: {second} -> 00")
                    parts[2] = '00'
                    corrected = True

            # Check hour
            if parts[0].isdigit():
                hour = int(parts[0])
                if hour >= 24:
                    logger.debug(f"Invalid hour value: {hour} -> 00")
                    parts[0] = '00'
                    corrected = True

            # Check minute
            if parts[1].isdigit():
                minute = int(parts[1])
                if minute >= 60:
                    logger.debug(f"Invalid minute value: {minute} -> 00")
                    parts[1] = '00'
                    corrected = True

            cleaned = ':'.join(parts)

            if corrected:
                logger.debug(f"Time corrected: {original_normalized} -> {cleaned}")

            return {
                'cleaned': cleaned,
                'original': original_normalized,
                'correction_applied': corrected
            }

        except (ValueError, IndexError):
            pass

    return {
        'cleaned': cleaned,
        'original': original_normalized,
        'correction_applied': False
    }
