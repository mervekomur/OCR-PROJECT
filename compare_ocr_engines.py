"""
OCR Engine Comparison: Google Vision + Claude vs Claude Vision Direct

Compares two approaches:
1. Google Vision OCR → Claude Structure Engine (current)
2. Claude Vision Direct (single API call)

Outputs results to: comparison_results.json
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from engines.google_vision_engine import GoogleVisionEngine
from engines.structure_engine import StructureEngine
from engines.claude_vision_engine import ClaudeVisionEngine


def run_google_claude_pipeline(image_path: str) -> dict:
    """Run Google Vision → Claude Structure pipeline."""
    result = {
        "engine": "google_vision + claude_structure",
        "success": False,
        "error": None,
        "timings": {},
        "data": None,
        "tokens": None
    }

    try:
        # Step 1: Google Vision OCR
        start = time.time()
        vision_engine = GoogleVisionEngine()
        vision_result = vision_engine.extract(image_path)
        google_time = time.time() - start
        result["timings"]["google_vision"] = round(google_time, 2)

        # Step 2: Claude Structure
        start = time.time()
        structure_engine = StructureEngine()
        sap_doc = structure_engine.extract_structured_data(vision_result.raw_text)
        claude_time = time.time() - start
        result["timings"]["claude_structure"] = round(claude_time, 2)
        result["timings"]["total"] = round(google_time + claude_time, 2)

        result["success"] = True
        result["data"] = {
            "raw_text": vision_result.raw_text[:500] + "..." if len(vision_result.raw_text) > 500 else vision_result.raw_text,
            "header": sap_doc.header,
            "financials": sap_doc.financials,
            "items": sap_doc.items[:3] if len(sap_doc.items) > 3 else sap_doc.items,
            "validation": sap_doc.validation
        }

    except Exception as e:
        result["error"] = str(e)

    return result


def run_claude_vision_direct(image_path: str) -> dict:
    """Run Claude Vision direct pipeline."""
    result = {
        "engine": "claude_vision_direct",
        "success": False,
        "error": None,
        "timings": {},
        "data": None,
        "tokens": None
    }

    try:
        start = time.time()
        engine = ClaudeVisionEngine()
        ocr_result = engine.extract(image_path)
        total_time = time.time() - start

        result["timings"]["total"] = round(total_time, 2)
        result["success"] = True
        result["tokens"] = {
            "input": ocr_result.metadata.get("input_tokens"),
            "output": ocr_result.metadata.get("output_tokens"),
            "total": ocr_result.metadata.get("tokens_used")
        }
        result["data"] = {
            "raw_text": ocr_result.raw_text[:500] + "..." if len(ocr_result.raw_text) > 500 else ocr_result.raw_text,
            "header": ocr_result.fields.get("header", {}),
            "financials": ocr_result.fields.get("financials", {}),
            "items": ocr_result.fields.get("items", [])[:3],
            "validation": ocr_result.fields.get("validation", {})
        }

    except Exception as e:
        result["error"] = str(e)

    return result


def compare_engines(image_paths: list, output_file: str = "comparison_results.json"):
    """
    Compare both engines on given images.

    Args:
        image_paths: List of image file paths
        output_file: Output JSON file name
    """
    results = {
        "comparison_date": datetime.now().isoformat(),
        "images_tested": len(image_paths),
        "summary": {
            "google_claude": {"success": 0, "failed": 0, "avg_time": 0},
            "claude_vision": {"success": 0, "failed": 0, "avg_time": 0, "avg_tokens": 0}
        },
        "detailed_results": []
    }

    google_times = []
    claude_times = []
    claude_tokens = []

    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {Path(image_path).name}")
        print("-" * 50)

        image_result = {
            "image": Path(image_path).name,
            "image_path": str(image_path),
            "google_claude": None,
            "claude_vision": None
        }

        # Run Google + Claude pipeline
        print("  Running Google Vision + Claude...")
        gc_result = run_google_claude_pipeline(image_path)
        image_result["google_claude"] = gc_result

        if gc_result["success"]:
            results["summary"]["google_claude"]["success"] += 1
            google_times.append(gc_result["timings"]["total"])
            print(f"    OK - {gc_result['timings']['total']}s")
        else:
            results["summary"]["google_claude"]["failed"] += 1
            print(f"    FAILED - {gc_result['error']}")

        # Run Claude Vision direct
        print("  Running Claude Vision Direct...")
        cv_result = run_claude_vision_direct(image_path)
        image_result["claude_vision"] = cv_result

        if cv_result["success"]:
            results["summary"]["claude_vision"]["success"] += 1
            claude_times.append(cv_result["timings"]["total"])
            if cv_result["tokens"]:
                claude_tokens.append(cv_result["tokens"]["total"])
            print(f"    OK - {cv_result['timings']['total']}s, {cv_result['tokens']['total']} tokens")
        else:
            results["summary"]["claude_vision"]["failed"] += 1
            print(f"    FAILED - {cv_result['error']}")

        results["detailed_results"].append(image_result)

    # Calculate averages
    if google_times:
        results["summary"]["google_claude"]["avg_time"] = round(sum(google_times) / len(google_times), 2)
    if claude_times:
        results["summary"]["claude_vision"]["avg_time"] = round(sum(claude_times) / len(claude_times), 2)
    if claude_tokens:
        results["summary"]["claude_vision"]["avg_tokens"] = int(sum(claude_tokens) / len(claude_tokens))

    # Save results
    output_path = Path(__file__).parent / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\nGoogle Vision + Claude:")
    print(f"  Success: {results['summary']['google_claude']['success']}/{len(image_paths)}")
    print(f"  Avg Time: {results['summary']['google_claude']['avg_time']}s")

    print(f"\nClaude Vision Direct:")
    print(f"  Success: {results['summary']['claude_vision']['success']}/{len(image_paths)}")
    print(f"  Avg Time: {results['summary']['claude_vision']['avg_time']}s")
    print(f"  Avg Tokens: {results['summary']['claude_vision']['avg_tokens']}")

    print(f"\nResults saved to: {output_path}")

    return results


def main():
    """Main entry point."""
    # Default test images
    data_dir = Path(__file__).parent / "data"

    # Find test images
    test_images = []

    # Priority: makbuz files, then o* files
    for pattern in ["makbuz*.png", "makbuz*.jpeg", "o*.png", "o*.jpeg", "o*.jpg"]:
        matches = list(data_dir.glob(pattern))
        test_images.extend(matches)

    # Test with more images (up to 10)
    test_images = test_images[:10]

    if not test_images:
        print("No test images found in data/ directory")
        return

    print("=" * 60)
    print("OCR ENGINE COMPARISON")
    print("Google Vision + Claude vs Claude Vision Direct")
    print("=" * 60)
    print(f"\nTest images: {len(test_images)}")
    for img in test_images:
        print(f"  - {img.name}")

    compare_engines([str(p) for p in test_images])


if __name__ == "__main__":
    main()
