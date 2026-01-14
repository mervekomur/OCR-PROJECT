#!/usr/bin/env python
"""
Batch Processing Demo
Demonstrates processing multiple receipt images.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from batch_processor import process_directory


def main():
    print("=" * 60)
    print("BATCH RECEIPT PROCESSOR DEMO")
    print("=" * 60)
    print()

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'parsed_receipts.json'

        receipts = process_directory(data_dir, output_file)

        print()
        print("=" * 60)
        print(f"COMPLETED: {len(receipts)} receipts processed")
        print(f"Output: {output_file}")
        print("=" * 60)
    else:
        print("Usage:")
        print("  python batch_demo.py <data_directory> [output_file]")
        print()
        print("Example:")
        print("  python batch_demo.py ../data parsed_receipts.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    main()
