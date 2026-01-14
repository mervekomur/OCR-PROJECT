#!/usr/bin/env python
"""
Ensemble OCR Comparison Demo
Compares multiple OCR engines on a receipt image.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from engines import EnsembleOCR, compare_engines


def main():
    print("=" * 80)
    print("ENSEMBLE OCR COMPARISON DEMO")
    print("=" * 80)

    # Check available engines
    ensemble = EnsembleOCR()
    available = ensemble.get_available_engines()

    print(f"\nAvailable engines: {', '.join(available) if available else 'None'}")

    if not available:
        print("\nNo OCR engines available. Please install dependencies:")
        print("  pip install easyocr              # EasyOCR")
        print("  pip install paddlepaddle paddleocr  # PaddleOCR")
        print("  pip install torch transformers   # Donut & GOT-OCR")
        return

    if len(sys.argv) > 1:
        image_path = sys.argv[1]

        # Optional: specify engines
        engines = None
        if len(sys.argv) > 2:
            engines = sys.argv[2].split(',')

        # Run comparison
        result = compare_engines(
            image_path,
            engines=engines,
            show_table=True
        )

        # Optionally save to JSON
        if len(sys.argv) > 3 and sys.argv[3] == '--json':
            output_file = 'comparison_result.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ensemble.to_json(result), f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_file}")

    else:
        print("\nUsage:")
        print("  python ensemble_demo.py <image_file> [engines] [--json]")
        print()
        print("Arguments:")
        print("  image_file  : Path to receipt image")
        print("  engines     : Comma-separated engine names (optional)")
        print("  --json      : Save results to JSON file")
        print()
        print("Examples:")
        print("  python ensemble_demo.py ../data/fis1.jpg")
        print("  python ensemble_demo.py ../data/fis1.jpg easyocr,paddleocr")
        print("  python ensemble_demo.py ../data/fis1.jpg easyocr --json")
        print()
        print("Available engines:")
        for name in available:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
