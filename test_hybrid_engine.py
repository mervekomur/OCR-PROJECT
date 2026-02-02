"""
Test Hybrid Vision Engine (Google Vision + Claude)
Tests all 42 files with the hybrid approach.
"""

import os
import sys
import json
import io
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from engines.hybrid_vision_engine import HybridVisionEngine


def main():
    print("=" * 70)
    print("HYBRID ENGINE TEST (Google Vision + Claude)")
    print("=" * 70)

    engine = HybridVisionEngine()

    folder = Path("C:/Users/MK/Desktop/OCR PROJECT/Makbuz- Taksi fişi")
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    print(f"\nToplam dosya: {len(images)}")
    print("-" * 70)

    results = []

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {image_path.name}")

        try:
            result = engine.extract_to_sap_json(str(image_path))

            header = result.get('header', {})
            financials = result.get('financials', {})
            validation = result.get('validation', {})
            metadata = result.get('metadata', {})

            brut = financials.get('brut_tutar')
            kdv_oran = financials.get('kdv_orani')
            kdv_tutar = financials.get('kdv_tutari')
            para_birimi = financials.get('para_birimi', 'TRY')
            google_conf = metadata.get('google_confidence', 0)

            print(f"   Google OCR: {google_conf:.1%} | Brut: {brut} {para_birimi} | KDV %{kdv_oran}")

            if header.get('semantic_completions'):
                for c in header['semantic_completions'][:2]:
                    print(f"   Semantic: '{c['original']}' -> '{c['completed']}'")

            results.append({
                "dosya": image_path.name,
                "basarili": True,
                "google_confidence": google_conf,
                "brut_tutar": brut,
                "para_birimi": para_birimi,
                "kdv_orani": kdv_oran,
                "kdv_tutari": kdv_tutar,
                "dil": metadata.get('detected_language'),
                "yurt_disi": metadata.get('is_foreign', False),
                "tesk": metadata.get('tesk_detected', False)
            })

        except Exception as e:
            print(f"   HATA: {e}")
            results.append({
                "dosya": image_path.name,
                "basarili": False,
                "hata": str(e)
            })

    # Summary
    print("\n" + "=" * 70)
    print("OZET RAPOR")
    print("=" * 70)

    successful = [r for r in results if r.get('basarili')]
    failed = [r for r in results if not r.get('basarili')]

    print(f"\nToplam: {len(results)}")
    print(f"Basarili: {len(successful)}")
    print(f"Basarisiz: {len(failed)}")

    if successful:
        avg_google_conf = sum(r.get('google_confidence', 0) for r in successful) / len(successful)
        print(f"\nOrtalama Google OCR Guven: {avg_google_conf:.1%}")

    # KDV distribution
    kdv_rates = {}
    for r in successful:
        rate = r.get('kdv_orani')
        kdv_rates[rate] = kdv_rates.get(rate, 0) + 1

    print(f"\n--- KDV ORANLARI ---")
    for rate, count in sorted(kdv_rates.items(), key=lambda x: str(x[0])):
        print(f"  %{rate}: {count}")

    # Save results
    output = {
        "test_tarihi": datetime.now().isoformat(),
        "engine": "hybrid_vision (Google Vision + Claude)",
        "toplam": len(results),
        "basarili": len(successful),
        "basarisiz": len(failed),
        "detaylar": results
    }

    output_file = Path(__file__).parent / "test_hybrid_engine_sonuclari.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSonuclar: {output_file}")


if __name__ == "__main__":
    main()
