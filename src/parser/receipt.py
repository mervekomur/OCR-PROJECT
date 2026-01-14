"""
Receipt Parser Main Module
Parses OCR output into structured receipt data.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from .cleaner import (
    clean_ocr_text,
    clean_business_name,
    clean_date_string,
    clean_time_string
)
from .extractor import (
    extract_date,
    extract_time,
    extract_amount,
    extract_business_name,
    extract_transaction_type
)
from .item_parser import extract_items

logger = get_logger(__name__)


def parse_receipt(ocr_lines: List[tuple]) -> Dict[str, Any]:
    """
    Parse OCR output into structured receipt data.

    Args:
        ocr_lines: EasyOCR output [(bbox, text, confidence), ...]

    Returns:
        Dict: Structured receipt data
    """
    # Extract texts from OCR output
    texts = [item[1] for item in ocr_lines]
    full_text = '\n'.join(texts)

    # Clean OCR errors
    cleaned_text = clean_ocr_text(full_text)

    # Extract date and time
    raw_date = extract_date(cleaned_text)
    raw_time = extract_time(cleaned_text)

    # Apply date/time corrections
    date_result = clean_date_string(raw_date)
    time_result = clean_time_string(raw_time)

    # Extract total amount
    total = extract_amount(
        cleaned_text,
        ["TOPLAM", "TUTAR", "TOTAL", "AMOUNT", "SUM", "GESAMT"]
    )

    # Extract items
    items = extract_items(texts)

    # Extract transaction type
    transaction_info = extract_transaction_type(cleaned_text, texts)

    # Extract and clean merchant name
    raw_merchant = extract_business_name(texts)

    # Build receipt structure
    receipt = {
        "merchant": clean_business_name(raw_merchant),
        "date": date_result['cleaned'],
        "time": time_result['cleaned'],
        "total": total,
        "transaction_type": transaction_info['type'],
        "items": items,
        "metadata": {
            "original_ocr_date": date_result['original'],
            "date_correction_applied": date_result['correction_applied'],
            "original_ocr_time": time_result['original'],
            "time_correction_applied": time_result['correction_applied'],
            "transaction_subtype": transaction_info['subtype'],
            "document_number": transaction_info['document_number'],
            "is_invoice": transaction_info['is_invoice']
        }
    }

    logger.info(f"Parsed receipt: {receipt['merchant']} - {receipt['total']}")

    return receipt


def to_json(receipt: Dict[str, Any], indent: int = 2) -> str:
    """
    Convert receipt dict to JSON string.

    Args:
        receipt: Receipt dictionary
        indent: JSON indentation level

    Returns:
        JSON string
    """
    return json.dumps(receipt, ensure_ascii=False, indent=indent)


def clean_none_values(d: Dict) -> Dict:
    """
    Remove None values from dictionary.

    Args:
        d: Dictionary to clean

    Returns:
        Dictionary without None values
    """
    cleaned = {}
    for key, value in d.items():
        if isinstance(value, dict):
            nested = clean_none_values(value)
            if nested:
                cleaned[key] = nested
        elif value is not None:
            cleaned[key] = value
    return cleaned
