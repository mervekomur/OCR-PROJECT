"""
Batch Receipt Extraction Script
Processes all receipts in data/ folder using Google Vision + Regex extraction.
Outputs results to CSV.
"""

import os
import sys
import csv
from pathlib import Path
from datetime import datetime

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.engines import GoogleVisionEngine
from src.engines.regex_structure_engine import RegexStructureEngine


def find_receipt_images(data_dir: str) -> list:
    """Find all receipt images in data directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in image_extensions:
                images.append(os.path.join(root, file))

    return sorted(images)


def process_receipts(images: list, output_csv: str):
    """Process all receipt images and save to CSV."""

    # Initialize engines
    print("Initializing engines...")
    vision_engine = GoogleVisionEngine()
    structure_engine = RegexStructureEngine()

    # CSV headers
    headers = [
        'file_path',
        'file_name',
        'merchant_name',
        'date',
        'time',
        'total_amount',
        'kdv_orani',
        'kdv_tutari',
        'kdv_matrah',
        'ocr_confidence',
        'processing_time',
        'status'
    ]

    results = []
    success_count = 0
    error_count = 0

    print(f"\nProcessing {len(images)} images...\n")
    print("-" * 60)

    for i, image_path in enumerate(images, 1):
        file_name = os.path.basename(image_path)
        print(f"[{i}/{len(images)}] {file_name}...", end=" ")

        try:
            # OCR with Google Vision
            ocr_result = vision_engine.extract(image_path)

            # Extract structured data
            structured = structure_engine.process_vision_result(ocr_result)

            # Prepare row
            tax = structured.tax_details or {}
            row = {
                'file_path': image_path,
                'file_name': file_name,
                'merchant_name': structured.merchant_name or '',
                'date': structured.date or '',
                'time': structured.time or '',
                'total_amount': structured.total_amount or '',
                'kdv_orani': tax.get('kdv_orani', ''),
                'kdv_tutari': tax.get('kdv_tutari', ''),
                'kdv_matrah': tax.get('kdv_matrah', ''),
                'ocr_confidence': f"{ocr_result.confidence:.2%}",
                'processing_time': f"{ocr_result.processing_time:.2f}s",
                'status': 'OK'
            }

            results.append(row)
            success_count += 1

            # Print summary
            total_str = f"{structured.total_amount:.2f}" if structured.total_amount else "N/A"
            print(f"OK | Total: {total_str} | Date: {structured.date or 'N/A'}")

        except Exception as e:
            error_count += 1
            results.append({
                'file_path': image_path,
                'file_name': file_name,
                'merchant_name': '',
                'date': '',
                'time': '',
                'total_amount': '',
                'kdv_orani': '',
                'kdv_tutari': '',
                'kdv_matrah': '',
                'ocr_confidence': '',
                'processing_time': '',
                'status': f'ERROR: {str(e)[:50]}'
            })
            print(f"ERROR: {str(e)[:40]}")

    print("-" * 60)

    # Write CSV
    print(f"\nWriting results to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {len(images)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Output file: {output_csv}")

    # Stats on extracted data
    totals_found = sum(1 for r in results if r['total_amount'])
    dates_found = sum(1 for r in results if r['date'])
    merchants_found = sum(1 for r in results if r['merchant_name'])

    print(f"\nExtraction Stats:")
    print(f"  - Totals found: {totals_found}/{len(images)}")
    print(f"  - Dates found: {dates_found}/{len(images)}")
    print(f"  - Merchants found: {merchants_found}/{len(images)}")

    return results


def main():
    """Main entry point."""
    # Configuration
    data_dir = "data"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"receipt_extraction_{timestamp}.csv"

    print("=" * 60)
    print("BATCH RECEIPT EXTRACTION")
    print("Google Vision OCR + Regex Structure Engine")
    print("=" * 60)

    # Find images
    images = find_receipt_images(data_dir)

    if not images:
        print(f"\nNo images found in {data_dir}/")
        print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
        return

    print(f"\nFound {len(images)} images in {data_dir}/")

    # Process
    process_receipts(images, output_csv)


if __name__ == "__main__":
    main()
