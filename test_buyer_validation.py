"""
Test script for StructureEngine with Buyer (Alici) Validation
"""
import sys
import os

# Encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from engines.google_vision_engine import GoogleVisionEngine
from engines.structure_engine import StructureEngine


def test_buyer_validation(image_path: str):
    """Test StructureEngine with buyer validation on a single image."""

    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print('='*60)

    # Step 1: OCR with Google Vision
    print("\n[1] Google Vision OCR...")
    vision = GoogleVisionEngine()
    ocr_result = vision.extract(image_path)

    if not ocr_result.raw_text:
        print("OCR Error: No text extracted")
        return

    print(f"OCR Success - {len(ocr_result.raw_text)} characters")
    print("\n--- Raw OCR Text (first 500 chars) ---")
    print(ocr_result.raw_text[:500])
    print("...")

    # Step 2: Structure extraction with Claude
    print("\n[2] Claude Structure Extraction (with Buyer Validation)...")
    structure = StructureEngine()
    sap_doc = structure.extract_structured_data(ocr_result.raw_text)

    # Step 3: Show results
    print("\n--- SAP Document ---")
    print(sap_doc.to_json(indent=2))

    # Step 4: Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    header = sap_doc.header
    validation = sap_doc.validation

    # Seller
    satici = header.get("satici", {})
    print(f"\nSATICI (Seller):")
    print(f"  Firma: {satici.get('firma_adi', 'N/A')}")
    print(f"  VKN: {satici.get('vkn', 'N/A')}")

    # Buyer
    alici = header.get("alici", {})
    print(f"\nALICI (Buyer):")
    print(f"  Unvan/Ad: {alici.get('unvan_veya_ad', 'N/A')}")
    print(f"  VKN/TCKN: {alici.get('vkn_veya_tckn', 'N/A')}")
    print(f"  Tip: {alici.get('tip', 'N/A')}")

    # Validation results
    print(f"\nVALIDATION RESULTS:")
    print(f"  Alici Gecerli: {validation.get('alici_gecerli', False)}")
    print(f"  Alici Tipi: {validation.get('alici_tipi', 'N/A')}")
    print(f"  FLO Sirketi: {validation.get('alici_flo_sirketi', 'N/A')}")
    print(f"  Islem Durumu: {validation.get('islem_durumu', 'N/A')}")

    if validation.get('red_sebebi'):
        print(f"  Red Sebebi: {validation.get('red_sebebi')}")

    # Warnings
    warnings = validation.get('uyarilar', [])
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    [{w.get('kod')}] {w.get('seviye')}: {w.get('mesaj')}")

    # Financials
    fin = sap_doc.financials
    print(f"\nFINANCIALS:")
    print(f"  Brut Tutar: {fin.get('brut_tutar', 'N/A')}")
    print(f"  KDV Orani: %{fin.get('kdv_orani', 'N/A')}")
    print(f"  KDV Tutari: {fin.get('kdv_tutari', 'N/A')}")

    return sap_doc


if __name__ == "__main__":
    # Test with o4.png (which has detailed receipt info)
    test_image = os.path.join(os.path.dirname(__file__), "data", "o4.png")

    if not os.path.exists(test_image):
        # Try another image
        test_image = os.path.join(os.path.dirname(__file__), "data", "o2.jpeg")

    if os.path.exists(test_image):
        test_buyer_validation(test_image)
    else:
        print("Test image not found!")
