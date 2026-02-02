"""
Test script for improved Claude Vision Engine v2

Tests:
- TESK rule (KDV %0 for taxi receipts)
- Multi-language detection
- Account code matching
- KKEG calculation
- VKN checksum validation
"""

import os
import sys
import json
import io
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from engines.claude_vision_engine import ClaudeVisionEngine


def test_single_image(engine: ClaudeVisionEngine, image_path: str) -> dict:
    """Test a single image and return results."""
    print(f"\n{'='*60}")
    print(f"Testing: {Path(image_path).name}")
    print('='*60)

    try:
        result = engine.extract_to_sap_json(image_path)

        # Print key information
        print(f"\n📄 Belge Türü: {result['header'].get('belge_turu', 'N/A')}")
        print(f"🌍 Tespit Edilen Dil: {result['metadata'].get('detected_language', 'N/A')}")

        # Financial info
        fin = result.get('financials', {})
        print(f"\n💰 Finansal Bilgiler:")
        print(f"   Brüt Tutar: {fin.get('brut_tutar', 'N/A')} {fin.get('para_birimi', 'TRY')}")
        print(f"   KDV Oranı: %{fin.get('kdv_orani', 'N/A')}")
        print(f"   KDV Tutarı: {fin.get('kdv_tutari', 'N/A')}")

        # TESK check
        val = result.get('validation', {})
        if val.get('tesk_tespit'):
            print(f"\n🚕 TESK TESPİT EDİLDİ → KDV %0 uygulandı")

        # KKEG check
        if val.get('kkeg_hesaplama'):
            kkeg = val['kkeg_hesaplama']
            print(f"\n🚗 KKEG HESAPLAMASI (Binek Araç):")
            print(f"   Matrah: {kkeg['matrah']:.2f}")
            print(f"   Gider (%70): {kkeg['gider_tutar']:.2f}")
            print(f"   KKEG Gider (%30): {kkeg['kkeg_gider']:.2f}")
            print(f"   KKEG KDV: {kkeg['kkeg_kdv']:.2f}")

        # Account code matching
        if val.get('hesap_bilgisi'):
            hesap = val['hesap_bilgisi']
            print(f"\n📊 Hesap Kodu Eşleşmesi:")
            print(f"   Hesap Adı: {hesap.get('hesap_adi', 'N/A')}")
            print(f"   Hesap Kodu: {hesap.get('hesap_kodu', 'N/A')}")

        if val.get('esleslen_anahtar_kelimeler'):
            print(f"   Anahtar Kelimeler: {', '.join(val['esleslen_anahtar_kelimeler'])}")

        # Validation status
        print(f"\n✅ İşlem Durumu: {val.get('islem_durumu', 'N/A')}")
        if val.get('red_sebebi'):
            print(f"❌ Red Sebebi: {val['red_sebebi']}")

        # Tevkifat check
        tevkifat = val.get('tevkifat', {})
        if tevkifat.get('tevkifatli'):
            print(f"\n📋 TEVKİFATLI FATURA")
            print(f"   Tevkifat Oranı: {tevkifat.get('tevkifat_orani', 'N/A')}")

        # Timing
        print(f"\n⏱️ İşlem Süresi: {result['metadata'].get('processing_time', 0):.2f}s")
        print(f"📊 Token Kullanımı: {result['metadata'].get('tokens_used', 0)}")

        return {"success": True, "result": result}

    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        return {"success": False, "error": str(e)}


def run_tests():
    """Run tests on sample images."""
    print("="*60)
    print("CLAUDE VISION ENGINE v2 - TEST")
    print("FLO Masraf Modülü - Tüm İş Kuralları Entegre")
    print("="*60)

    # Initialize engine
    engine = ClaudeVisionEngine()

    # Find test images
    data_dir = Path(__file__).parent / "data"
    test_images = []

    # Get various image types
    for pattern in ["makbuz*.png", "makbuz*.jpeg", "o*.png", "o*.jpeg", "o*.jpg"]:
        matches = list(data_dir.glob(pattern))
        test_images.extend(matches)

    # Limit for testing
    test_images = test_images[:5]

    if not test_images:
        print("No test images found!")
        return

    print(f"\n📁 Test Görüntüleri: {len(test_images)}")
    for img in test_images:
        print(f"   - {img.name}")

    # Results storage
    results = {
        "test_date": datetime.now().isoformat(),
        "engine_version": "v2",
        "images_tested": len(test_images),
        "results": []
    }

    success_count = 0
    tesk_detected_count = 0
    kkeg_applied_count = 0

    for image_path in test_images:
        test_result = test_single_image(engine, str(image_path))
        results["results"].append({
            "image": image_path.name,
            **test_result
        })

        if test_result.get("success"):
            success_count += 1
            val = test_result.get("result", {}).get("validation", {})
            if val.get("tesk_tespit"):
                tesk_detected_count += 1
            if val.get("kkeg_hesaplama"):
                kkeg_applied_count += 1

    # Summary
    print("\n" + "="*60)
    print("TEST ÖZET")
    print("="*60)
    print(f"✅ Başarılı: {success_count}/{len(test_images)}")
    print(f"🚕 TESK Tespit: {tesk_detected_count}")
    print(f"🚗 KKEG Uygulandı: {kkeg_applied_count}")

    # Save results
    output_file = Path(__file__).parent / "test_results_v2.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n📄 Sonuçlar kaydedildi: {output_file}")


if __name__ == "__main__":
    run_tests()
