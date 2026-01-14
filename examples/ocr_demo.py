#!/usr/bin/env python
"""
OCR Engine Demo
Demonstrates basic OCR text extraction.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ocr_engine import OCREngine


def main():
    print("=" * 50)
    print("OCR Engine - EasyOCR Demo")
    print("=" * 50)

    if len(sys.argv) > 1:
        image_file = sys.argv[1]

        try:
            engine = OCREngine(languages=['tr', 'en'], gpu=False)
            result = engine.extract_text_from_file(image_file, detail=True)

            print(f"\nFile: {image_file}")
            print("-" * 50)
            print("EXTRACTED TEXT:")
            print("-" * 50)

            for bbox, text, confidence in result:
                conf_percent = confidence * 100
                print(f"[{conf_percent:.1f}%] {text}")

            print("-" * 50)
            print("\nFULL TEXT:")
            print("-" * 50)
            full_text = '\n'.join([item[1] for item in result])
            print(full_text)

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("\nUsage:")
        print("-" * 50)
        print("python ocr_demo.py <image_file>")
        print("\nExample:")
        print("  python ocr_demo.py ../data/fis1.jpg")


if __name__ == "__main__":
    main()
