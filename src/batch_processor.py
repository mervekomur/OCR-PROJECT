"""
Batch Receipt Processor
Processes multiple receipts and generates clean JSON output.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List

from utils.logger import get_logger
from constants import SUPPORTED_IMAGE_EXTENSIONS
from ocr_engine import OCREngine
from parser import parse_receipt

logger = get_logger(__name__)


def find_receipt_file(base_path: str, name: str) -> str:
    """
    Find receipt file with various extensions.

    Args:
        base_path: Data directory path
        name: File name without extension (e.g., 'fis1')

    Returns:
        Full path to found file

    Raises:
        FileNotFoundError: If file not found with any extension
    """
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        full_path = os.path.join(base_path, f"{name}{ext}")
        if os.path.exists(full_path):
            return full_path

    raise FileNotFoundError(
        f"File not found: {name} (tried extensions: {SUPPORTED_IMAGE_EXTENSIONS})"
    )


def safe_ocr_extract(engine: OCREngine, file_path: str) -> list:
    """
    Safe OCR extraction with error handling.

    Args:
        engine: OCREngine instance
        file_path: Path to image file

    Returns:
        OCR results or empty list on error
    """
    try:
        return engine.extract_text_from_file(file_path, detail=True)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return []
    except Exception as e:
        logger.error(f"OCR error ({file_path}): {e}")
        return []


def load_receipt_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load receipt configuration from JSON file.

    Args:
        config_path: Path to config file (default: config/receipts.json)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'receipts.json'

    if not Path(config_path).exists():
        logger.warning(f"Config file not found: {config_path}")
        return {'receipts': {}}

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_receipt(
    fis_name: str,
    definition: Dict[str, Any],
    refined_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build receipt data in standard format.

    Args:
        fis_name: Receipt name (e.g., 'fis1')
        definition: Receipt definition from config
        refined_data: Manually refined data

    Returns:
        Standardized receipt dictionary
    """
    receipt = {
        'merchant': definition.get('merchant'),
        'date': refined_data.get('date'),
        'time': refined_data.get('time'),
        'total': refined_data.get('total'),
        'transaction_type': definition.get('type', 'SATIS'),
        'items': refined_data.get('items', []),
        'metadata': {
            'original_ocr_date': refined_data.get('metadata', {}).get('original_ocr_date'),
            'date_correction_applied': refined_data.get('metadata', {}).get('date_correction_applied', False),
            'original_ocr_time': refined_data.get('metadata', {}).get('original_ocr_time'),
            'time_correction_applied': refined_data.get('metadata', {}).get('time_correction_applied', False),
            'transaction_subtype': refined_data.get('metadata', {}).get('transaction_subtype'),
            'document_number': refined_data.get('metadata', {}).get('document_number'),
            'is_invoice': refined_data.get('metadata', {}).get('is_invoice', False),
            'currency': definition.get('currency', 'TRY')
        }
    }

    # Add extra metadata fields
    extra_keys = [
        'address', 'siret', 'naf', 'tva_number', 'code_ape',
        'net_amount', 'tax_rate', 'tax_amount', 'gross_amount',
        'payment_method', 'article_count', 'phone', 'bank',
        'installments', 'installment_amount', 'approval_code', 'ref_no',
        'ticket_number', 'service_type', 'note', 'ettn', 'customer'
    ]

    for key in extra_keys:
        value = refined_data.get('metadata', {}).get(key)
        if value is not None:
            receipt['metadata'][key] = value

    return receipt


def process_receipt_file(
    engine: OCREngine,
    file_path: str
) -> Dict[str, Any]:
    """
    Process a single receipt file.

    Args:
        engine: OCREngine instance
        file_path: Path to receipt image

    Returns:
        Parsed receipt data
    """
    logger.info(f"Processing: {file_path}")

    ocr_result = safe_ocr_extract(engine, file_path)
    if not ocr_result:
        return None

    receipt = parse_receipt(ocr_result)
    return receipt


def process_directory(
    data_dir: str,
    output_file: str = 'parsed_receipts.json',
    languages: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Process all receipt images in a directory.

    Args:
        data_dir: Directory containing receipt images
        output_file: Output JSON file path
        languages: OCR languages

    Returns:
        List of parsed receipts
    """
    logger.info(f"Processing directory: {data_dir}")

    engine = OCREngine(languages=languages or ['tr', 'en'])
    receipts = []

    # Find all image files
    data_path = Path(data_dir)
    image_files = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_files.extend(data_path.glob(f"*{ext}"))

    for image_file in sorted(image_files):
        receipt = process_receipt_file(engine, str(image_file))
        if receipt:
            receipts.append(receipt)
            logger.info(f"  {receipt['merchant']} - {receipt['total']}")

    # Save to JSON
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(receipts, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(receipts)} receipts to {output_file}")

    return receipts


def process_from_config(
    config_path: str = None,
    output_file: str = 'parsed_receipts.json'
) -> List[Dict[str, Any]]:
    """
    Process receipts using configuration file.

    Args:
        config_path: Path to receipts config file
        output_file: Output JSON file path

    Returns:
        List of processed receipts
    """
    config = load_receipt_config(config_path)
    receipts_config = config.get('receipts', {})

    if not receipts_config:
        logger.warning("No receipt configurations found")
        return []

    receipts = []

    for fis_name, fis_config in receipts_config.items():
        definition = fis_config.get('definition', {})
        refined_data = fis_config.get('refined_data', {})

        logger.info(f"[{fis_name}] {definition.get('merchant', 'Unknown')}...")

        receipt = build_receipt(fis_name, definition, refined_data)
        receipts.append(receipt)

        logger.info(f"  Date: {receipt['date']}, Total: {receipt['total']}")

    # Save to JSON
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(receipts, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(receipts)} receipts to {output_file}")

    return receipts
