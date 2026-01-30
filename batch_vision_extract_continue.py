"""
Batch Receipt Extraction Script - Continue from where we left off
Processes remaining receipts starting from 'makbuz' files.
Includes rate limit protection with delays.
"""

import os
import sys
import csv
import time
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

    return images


def sort_images_makbuz_first(images: list) -> list:
    """
    Sort images: 'makbuz' files first, then reverse alphabetical order.
    This prioritizes newer/later files.
    """
    makbuz_files = []
    other_files = []

    for img in images:
        filename = os.path.basename(img).lower()
        if filename.startswith('makbuz'):
            makbuz_files.append(img)
        else:
            other_files.append(img)

    # Sort makbuz files alphabetically
    makbuz_files.sort(key=lambda x: os.path.basename(x).lower())

    # Sort other files in reverse alphabetical order (later files first)
    other_files.sort(key=lambda x: os.path.basename(x).lower(), reverse=True)

    return makbuz_files + other_files


def get_previously_processed_files(csv_pattern: str = "receipt_extraction_*.csv") -> set:
    """Get list of files already processed from previous CSV files."""
    import glob

    processed = set()
    csv_files = glob.glob(csv_pattern)

    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('status') == 'OK':
                        processed.add(row.get('file_path', ''))
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")

    return processed


def process_receipts(images: list, output_csv: str, delay_seconds: float = 3.0):
    """Process receipt images with rate limit protection."""

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

    print(f"\nProcessing {len(images)} images...")
    print(f"Rate limit protection: {delay_seconds}s delay between requests")
    print("-" * 60)

    for i, image_path in enumerate(images, 1):
        file_name = os.path.basename(image_path)
        print(f"[{i}/{len(images)}] {file_name}...", end=" ", flush=True)

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
            error_msg = str(e)
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
                'status': f'ERROR: {error_msg[:50]}'
            })
            print(f"ERROR: {error_msg[:40]}")

            # If rate limited, wait longer
            if '429' in error_msg:
                print(f"    Rate limited! Waiting {delay_seconds * 3}s...")
                time.sleep(delay_seconds * 3)

        # Rate limit protection - wait between requests
        if i < len(images):
            time.sleep(delay_seconds)

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

    # Stats
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
    delay_seconds = 3.0  # Wait 3 seconds between requests
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"receipt_extraction_continued_{timestamp}.csv"

    print("=" * 60)
    print("BATCH RECEIPT EXTRACTION - CONTINUED")
    print("Google Vision OCR + Regex Structure Engine")
    print("=" * 60)

    # Find all images
    all_images = find_receipt_images(data_dir)
    print(f"\nTotal images in {data_dir}/: {len(all_images)}")

    # Get previously processed files
    processed = get_previously_processed_files()
    print(f"Previously processed (OK status): {len(processed)}")

    # Filter out already processed
    remaining_images = [img for img in all_images if img not in processed]
    print(f"Remaining to process: {len(remaining_images)}")

    if not remaining_images:
        print("\nAll images have been processed!")
        return

    # Sort: makbuz files first, then reverse alphabetical
    sorted_images = sort_images_makbuz_first(remaining_images)

    print(f"\nProcessing order (first 10):")
    for i, img in enumerate(sorted_images[:10], 1):
        print(f"  {i}. {os.path.basename(img)}")
    if len(sorted_images) > 10:
        print(f"  ... and {len(sorted_images) - 10} more")

    print()

    # Process
    process_receipts(sorted_images, output_csv, delay_seconds)


if __name__ == "__main__":
    main()
