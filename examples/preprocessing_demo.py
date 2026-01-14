#!/usr/bin/env python
"""
Preprocessing Demo
Demonstrates image preprocessing for OCR.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import preprocess_file


def main():
    print("=" * 50)
    print("Image Preprocessing Demo")
    print("=" * 50)

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output_processed.png"

        try:
            result = preprocess_file(
                input_path=input_file,
                output_path=output_file,
                noise_method="gaussian",
                threshold_method="adaptive"
            )
            print(f"Success! Image size: {result.shape}")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("\nUsage:")
        print("-" * 50)
        print("python preprocessing_demo.py <input_image> [output_image]")
        print("\nExample:")
        print("  python preprocessing_demo.py ../data/fis1.jpg processed.png")
        print("\nNoise Methods: gaussian, median, bilateral, nlm")
        print("Threshold Methods: binary, otsu, adaptive, adaptive_gaussian")


if __name__ == "__main__":
    main()
