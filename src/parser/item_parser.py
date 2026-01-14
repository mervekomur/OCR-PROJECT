"""
Item Parser Module
Extracts product items from receipt OCR text.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from constants import SKIP_KEYWORDS, MIN_PRODUCT_NAME_LENGTH, MIN_LETTER_RATIO

logger = get_logger(__name__)


def should_skip_text(text: str) -> bool:
    """
    Check if text line should be skipped.

    Args:
        text: Text line to check

    Returns:
        True if should be skipped
    """
    text_upper = text.upper()
    for kw in SKIP_KEYWORDS:
        if kw in text_upper:
            return True
    return False


def is_product_name(text: str) -> bool:
    """
    Check if text is likely a product name.

    Args:
        text: Text to check

    Returns:
        True if likely a product name
    """
    text = text.strip()

    if len(text) < MIN_PRODUCT_NAME_LENGTH:
        return False

    if should_skip_text(text):
        return False

    # Should be mostly letters (including Turkish characters)
    letter_count = sum(1 for c in text if c.isalpha())
    return letter_count >= len(text) * MIN_LETTER_RATIO


def extract_price_from_text(text: str) -> Optional[float]:
    """
    Extract price from text.

    Args:
        text: Text to parse

    Returns:
        Price as float or None
    """
    text = text.strip()

    # Special case: 8-prefix OCR error (* read as 8)
    # e.g., 8242,00 -> 242,00
    match_8_prefix = re.match(r'^8(\d{2,}[.,]\d{2})$', text)
    if match_8_prefix:
        price_str = match_8_prefix.group(1).replace(',', '.')
        try:
            return float(price_str)
        except ValueError:
            pass

    patterns = [
        r'^\*(\d+[.,]\d{2})$',  # *242,00
        r'^(\d+[.,]\d{2})$',    # 242,00
    ]

    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            price_str = match.group(1).replace(',', '.')
            try:
                return float(price_str)
            except ValueError:
                continue

    return None


def try_single_line_match(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to match product+price in a single line.

    Supported formats:
    - 2 x EKMEK 10,00
    - 2*EKMEK 10,00
    - EKMEK 2 10,00
    - EKMEK 10,00

    Args:
        text: Text line to parse

    Returns:
        Item dict or None
    """
    item = None

    # Pattern 1: QUANTITY x PRODUCT PRICE
    pattern1 = r'^(\d+)\s*[xX\*]\s*(.+?)\s+(\d+[.,]\d{2})$'
    match = re.match(pattern1, text)
    if match:
        qty = int(match.group(1))
        name = match.group(2).strip()
        price_str = match.group(3).replace(',', '.')
        item = {"quantity": qty, "name": name, "price": float(price_str)}
        return item

    # Pattern 2: PRODUCT QUANTITY PRICE
    pattern2 = r'^([A-Za-zÀ-ÿĞğÜüŞşİıÖöÇç\s]+)\s+(\d{1,2})\s+(\d+[.,]\d{2})$'
    match = re.match(pattern2, text)
    if match:
        name = match.group(1).strip()
        qty = int(match.group(2))
        price_str = match.group(3).replace(',', '.')
        if 1 <= qty <= 99:
            item = {"quantity": qty, "name": name, "price": float(price_str)}
            return item

    # Pattern 3: PRODUCT PRICE
    pattern3 = r'^([A-Za-zÀ-ÿĞğÜüŞşİıÖöÇç\s]+)\s+(\d+[.,]\d{2})$'
    match = re.match(pattern3, text)
    if match:
        name = match.group(1).strip()
        price_str = match.group(2).replace(',', '.')
        if len(name) >= 2:
            item = {"quantity": 1, "name": name, "price": float(price_str)}
            return item

    # Pattern 4: *PRICE PRODUCT
    pattern4 = r'^\*(\d+[.,]\d{2})\s+(.+)$'
    match = re.match(pattern4, text)
    if match:
        price_str = match.group(1).replace(',', '.')
        name = match.group(2).strip()
        if len(name) >= 2:
            item = {"quantity": 1, "name": name, "price": float(price_str)}
            return item

    # Pattern 5: PRODUCT *PRICE
    pattern5 = r'^(.+?)\s+\*(\d+[.,]\d{2})$'
    match = re.match(pattern5, text)
    if match:
        name = match.group(1).strip()
        price_str = match.group(2).replace(',', '.')
        if len(name) >= 2:
            item = {"quantity": 1, "name": name, "price": float(price_str)}
            return item

    return None


def clean_item_name(name: str) -> str:
    """
    Clean and normalize item name.

    Args:
        name: Raw item name

    Returns:
        Cleaned item name
    """
    # Import here to avoid circular import
    from .cleaner import clean_product_name

    cleaned = re.sub(r'\s+', ' ', name).strip()
    cleaned = clean_product_name(cleaned)
    return cleaned


def extract_items(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Extract product items from receipt text lines.

    Supported formats:
    Single line:
    - 2 x EKMEK 10,00
    - 2*EKMEK 10,00
    - EKMEK 2 10,00
    - EKMEK 10,00

    Consecutive lines:
    - Line 1: "Unlu Mamuller"
    - Line 2: "B10" (skipped)
    - Line 3: "*242,00"

    Args:
        texts: List of OCR text lines

    Returns:
        List of item dicts: [{"quantity": 2, "name": "EKMEK", "price": 10.00}, ...]
    """
    items = []
    used_indices = set()

    i = 0
    while i < len(texts):
        text = texts[i].strip()

        if not text or len(text) < 2 or i in used_indices:
            i += 1
            continue

        if should_skip_text(text):
            i += 1
            continue

        # 1. Try single line match
        item = try_single_line_match(text)
        if item:
            item['name'] = clean_item_name(item['name'])
            items.append(item)
            used_indices.add(i)
            logger.debug(f"Single line item: {item['quantity']}x {item['name']} = {item['price']}")
            i += 1
            continue

        # 2. Try consecutive line match
        if is_product_name(text):
            product_name = text
            price = None
            price_index = None

            # Look at next 1-3 lines
            for j in range(i + 1, min(i + 4, len(texts))):
                if j in used_indices:
                    continue

                next_text = texts[j].strip()
                price = extract_price_from_text(next_text)

                if price:
                    price_index = j
                    break

            if price and price_index:
                cleaned_name = clean_item_name(product_name)
                item = {
                    "quantity": 1,
                    "name": cleaned_name,
                    "price": price
                }
                items.append(item)
                used_indices.add(i)
                used_indices.add(price_index)
                logger.debug(f"Multi-line item: {item['quantity']}x {item['name']} = {item['price']}")

        i += 1

    return items
