#!/usr/bin/env python
"""
Receipt Parser Demo
Demonstrates parsing OCR output into structured data.
"""

import sys
import json
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ocr_engine import OCREngine
from parser import parse_receipt


def main():
    OUTPUT_FILE = "parsed_receipts.json"

    print("=" * 50)
    print("Receipt Parser - OCR to JSON Demo")
    print("=" * 50)

    if len(sys.argv) > 1:
        image_file = sys.argv[1]

        try:
            print(f"\nFile: {image_file}")
            print("Running OCR...")

            engine = OCREngine(languages=['tr', 'en'], gpu=False)
            ocr_result = engine.extract_text_from_file(image_file, detail=True)

            print("Extracting data...")
            receipt = parse_receipt(ocr_result)

            # Load existing JSON (if exists)
            receipts_list = []
            if os.path.exists(OUTPUT_FILE):
                try:
                    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                        receipts_list = json.load(f)
                        if not isinstance(receipts_list, list):
                            receipts_list = [receipts_list]
                except (json.JSONDecodeError, IOError):
                    receipts_list = []

            # Add new receipt
            receipts_list.append(receipt)

            # Save to JSON
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(receipts_list, f, ensure_ascii=False, indent=2)

            print(f"\nSaved to '{OUTPUT_FILE}'")

            # Show summary
            print("\n" + "=" * 50)
            print("SUMMARY")
            print("=" * 50)
            print(f"Merchant       : {receipt.get('merchant', '-')}")
            print(f"Date           : {receipt.get('date', '-')}")
            print(f"Time           : {receipt.get('time', '-')}")
            print(f"Total          : {receipt.get('total', '-')}")
            print(f"Transaction    : {receipt.get('transaction_type', '-')}")

            if receipt['metadata'].get('date_correction_applied'):
                print(f"Date Corrected : {receipt['metadata']['original_ocr_date']} -> {receipt['date']}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\nUsage:")
        print("  python parser_demo.py <image_file>")
        print("\nExample:")
        print("  python parser_demo.py ../data/fis1.jpg")
        print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
