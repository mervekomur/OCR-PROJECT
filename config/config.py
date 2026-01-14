"""
Configuration loader for OCR project.
Loads JSON config files and provides access to settings.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

# Config directory path
CONFIG_DIR = Path(__file__).parent


def load_json_config(filename: str) -> Dict[str, Any]:
    """
    Load a JSON configuration file.

    Args:
        filename: Name of the config file (without path)

    Returns:
        Dict containing configuration data
    """
    config_path = CONFIG_DIR / filename
    if not config_path.exists():
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_ocr_corrections() -> Dict[str, Dict[str, str]]:
    """Get OCR text corrections dictionary."""
    return load_json_config('ocr_corrections.json')


def get_business_name_corrections() -> Dict[str, str]:
    """Get business name corrections."""
    corrections = get_ocr_corrections()
    return corrections.get('business_name_corrections', {})


def get_product_name_corrections() -> Dict[str, str]:
    """Get product name corrections."""
    corrections = get_ocr_corrections()
    return corrections.get('product_name_corrections', {})
