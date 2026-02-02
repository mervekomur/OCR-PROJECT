"""
Full KDV Heuristics Test - All 42 Files
Tests mathematical heuristics for KDV calculation on all receipts.
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

from engines.claude_vision_engine import ClaudeVisionEngine


def main():
    print("=" * 70)
    print("KDV HEURISTICS FULL TEST - 42 FILES")
    print("=" * 70)

    engine = ClaudeVisionEngine()

    folder = Path("C:/Users/MK/Desktop/OCR PROJECT/Makbuz- Taksi fişi")
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    print(f"\nToplam dosya: {len(images)}")
    print("-" * 70)

    results = []

    # Counters
    kdv_methods = {
        "ocr_dogrulandi": 0,
        "ocr_oran": 0,
        "cross_check_20": 0,
        "cross_check_10": 0,
        "cross_check_8": 0,
        "smart_default": 0,
        "semantic_completion": 0,
        "fallback_20": 0,
        "yurt_disi": 0,
        "eksik_veri": 0,
        "unknown": 0
    }

    kdv_rates = {0: 0, 8: 0, 10: 0, 20: 0, "other": 0}
    tahmini_count = 0
    kesin_count = 0

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
            kdv_method = financials.get('kdv_hesaplama_yontemi', 'unknown')
            tahmini = financials.get('kdv_tahmini_atandi', False)
            para_birimi = financials.get('para_birimi', 'TRY')

            # Count methods
            if kdv_method in kdv_methods:
                kdv_methods[kdv_method] += 1
            else:
                kdv_methods["unknown"] += 1

            # Count rates
            if kdv_oran in kdv_rates:
                kdv_rates[kdv_oran] += 1
            else:
                kdv_rates["other"] += 1

            # Count estimated vs confirmed
            if tahmini:
                tahmini_count += 1
            else:
                kesin_count += 1

            # Output
            method_short = kdv_method[:15] if kdv_method else "N/A"
            status = "TAHMINI" if tahmini else "KESIN"

            print(f"   Brut: {brut} {para_birimi} | KDV %{kdv_oran} | {method_short} | [{status}]")

            # Semantic completions
            if header.get('semantic_completions'):
                for c in header['semantic_completions'][:2]:  # Max 2
                    print(f"   Semantic: '{c['original']}' -> '{c['completed']}'")

            results.append({
                "dosya": image_path.name,
                "basarili": True,
                "brut_tutar": brut,
                "para_birimi": para_birimi,
                "kdv_orani": kdv_oran,
                "kdv_tutari": kdv_tutar,
                "kdv_matrah": financials.get('kdv_matrah'),
                "hesaplama_yontemi": kdv_method,
                "tahmini_atandi": tahmini,
                "kdv_notu": financials.get('kdv_notu'),
                "semantic_completions": header.get('semantic_completions'),
                "dil": metadata.get('detected_language'),
                "yurt_disi": metadata.get('yurt_disi_masrafi', False)
            })

        except Exception as e:
            print(f"   HATA: {e}")
            results.append({
                "dosya": image_path.name,
                "basarili": False,
                "hata": str(e)
            })
            kdv_methods["eksik_veri"] += 1

    # Summary
    print("\n" + "=" * 70)
    print("OZET RAPOR")
    print("=" * 70)

    successful = [r for r in results if r.get('basarili')]
    failed = [r for r in results if not r.get('basarili')]

    print(f"\nToplam: {len(results)}")
    print(f"Basarili: {len(successful)}")
    print(f"Basarisiz: {len(failed)}")

    print(f"\n--- KDV HESAPLAMA YONTEMLERI ---")
    for method, count in sorted(kdv_methods.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / len(results) * 100
            print(f"  {method}: {count} ({pct:.1f}%)")

    print(f"\n--- KDV ORANLARI ---")
    for rate, count in sorted(kdv_rates.items(), key=lambda x: str(x[0])):
        if count > 0:
            pct = count / len(results) * 100
            print(f"  %{rate}: {count} ({pct:.1f}%)")

    print(f"\n--- GUVENILIRLIK ---")
    print(f"  Kesin (OCR/Cross-check): {kesin_count} ({kesin_count/len(results)*100:.1f}%)")
    print(f"  Tahmini (Smart default/Fallback): {tahmini_count} ({tahmini_count/len(results)*100:.1f}%)")

    # Yurt disi vs Turkiye
    yurt_disi = [r for r in successful if r.get('yurt_disi')]
    turkiye = [r for r in successful if not r.get('yurt_disi')]
    print(f"\n--- ULKE DAGILIMI ---")
    print(f"  Turkiye: {len(turkiye)}")
    print(f"  Yurt Disi: {len(yurt_disi)}")

    # Semantic completions
    with_semantic = [r for r in successful if r.get('semantic_completions')]
    print(f"\n--- SEMANTIC COMPLETION ---")
    print(f"  Tamamlanan kelime iceren: {len(with_semantic)}")

    # Save results
    output = {
        "test_tarihi": datetime.now().isoformat(),
        "test_turu": "KDV Heuristics Full Test",
        "toplam": len(results),
        "basarili": len(successful),
        "basarisiz": len(failed),
        "kdv_yontemleri": kdv_methods,
        "kdv_oranlari": {str(k): v for k, v in kdv_rates.items()},
        "kesin_sayisi": kesin_count,
        "tahmini_sayisi": tahmini_count,
        "yurt_disi_sayisi": len(yurt_disi),
        "turkiye_sayisi": len(turkiye),
        "semantic_completion_sayisi": len(with_semantic),
        "detaylar": results
    }

    output_file = Path(__file__).parent / "test_kdv_heuristics_full_sonuclari.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSonuclar: {output_file}")


if __name__ == "__main__":
    main()
