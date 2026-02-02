"""
Test KDV Mathematical Heuristics

Analyzes:
1. ny634 files (hotel receipts with missing KDV)
2. İsmoni file (textile with partial word completion)

Tests:
- Reverse Calculation
- Cross-Check with 95% tolerance
- Smart Defaulting
- Semantic Completion
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
from logic.tax_calculator import TaxCalculator, SemanticCompleter


def test_semantic_completer():
    """Test semantic completion for partial words."""
    print("=" * 70)
    print("SEMANTIC COMPLETION TEST")
    print("=" * 70)

    completer = SemanticCompleter()

    test_cases = [
        "İsmoni Tekstil Üretim Pazarlam",
        "Elektrome...",
        "TRASER TRAFO SERVİSİ ELEKTROME",
        "Quality Inn near Sunset Park",
        "guest folio",
    ]

    for text in test_cases:
        result = completer.complete_text(text)
        print(f"\nInput: '{text}'")
        if result["completions"]:
            for c in result["completions"]:
                print(f"  -> '{c['original']}' -> '{c['completed']}' ({c['similarity']:.0%})")
        if result["account_info"]:
            print(f"  Account: {result['account_info'].get('hesap_adi')} (KDV %{result['account_info'].get('default_kdv')})")
        else:
            print("  No account match")


def test_reverse_calculation():
    """Test reverse KDV calculation."""
    print("\n" + "=" * 70)
    print("REVERSE CALCULATION TEST")
    print("=" * 70)

    calc = TaxCalculator()

    test_cases = [
        (2099.99, 10),  # Tekstil - %10
        (1307.58, 8),   # CHF - %8
        (6000.0, 20),   # Genel - %20
        (109.0, 10),    # Otel - %10
    ]

    for brut, oran in test_cases:
        matrah, kdv = calc.reverse_calculate_kdv(brut, oran)
        print(f"Brut: {brut:,.2f} | KDV %{oran} -> Matrah: {matrah:,.2f} | KDV: {kdv:,.2f}")


def analyze_specific_files():
    """Analyze ny634 and İsmoni files with new heuristics."""
    print("\n" + "=" * 70)
    print("SPECIFIC FILE ANALYSIS")
    print("=" * 70)

    engine = ClaudeVisionEngine()

    # Files to analyze
    folder = Path("C:/Users/MK/Desktop/OCR PROJECT/Makbuz- Taksi fişi")
    target_files = [
        "ny634_guest-folio-batchd9fc1d03-1d75-4099-abd3-2f93a605e1bf.jpg",
        "ny634_guest_folio_first_page_only.jpg",
        "cleanshot-2024-10-01-at-144932-2x.png",  # İsmoni file
    ]

    results = []

    for filename in target_files:
        filepath = folder / filename
        if not filepath.exists():
            print(f"\n[SKIP] {filename} - File not found")
            continue

        print(f"\n{'='*70}")
        print(f"ANALYZING: {filename}")
        print("=" * 70)

        try:
            result = engine.extract_to_sap_json(str(filepath))

            header = result.get('header', {})
            financials = result.get('financials', {})
            validation = result.get('validation', {})
            metadata = result.get('metadata', {})

            # Basic info
            print(f"\nBelge Turu: {header.get('belge_turu', 'N/A')}")
            print(f"Dil: {metadata.get('detected_language', 'N/A')}")
            print(f"Satici: {header.get('satici', {}).get('firma_adi', 'N/A')}")

            # Financials
            print(f"\n--- FINANSAL BİLGİLER ---")
            print(f"Brut Tutar: {financials.get('brut_tutar', 'N/A')} {financials.get('para_birimi', 'TRY')}")
            print(f"KDV Orani: %{financials.get('kdv_orani', 'N/A')}")
            print(f"KDV Tutari: {financials.get('kdv_tutari', 'N/A')}")
            print(f"KDV Matrah: {financials.get('kdv_matrah', 'N/A')}")

            # Heuristics info
            print(f"\n--- KDV HEURISTICS ---")
            print(f"Hesaplama Yontemi: {financials.get('kdv_hesaplama_yontemi', 'N/A')}")
            print(f"Tahmini Atandi: {financials.get('kdv_tahmini_atandi', False)}")
            print(f"KDV Notu: {financials.get('kdv_notu', 'N/A')}")

            # Semantic completions
            if header.get('semantic_completions'):
                print(f"\n--- SEMANTIC COMPLETIONS ---")
                for c in header['semantic_completions']:
                    print(f"  '{c['original']}' -> '{c['completed']}' ({c['similarity']:.0%})")

            # Full heuristics result
            if validation.get('kdv_heuristics'):
                print(f"\n--- HEURISTICS DETAY ---")
                heur = validation['kdv_heuristics']
                print(f"  Yontem: {heur.get('hesaplama_yontemi')}")
                print(f"  Guven: {heur.get('guven_orani', 0):.0%}")
                if heur.get('notlar'):
                    for note in heur['notlar']:
                        print(f"  - {note}")

            results.append({
                "dosya": filename,
                "basarili": True,
                "brut_tutar": financials.get('brut_tutar'),
                "para_birimi": financials.get('para_birimi'),
                "kdv_orani": financials.get('kdv_orani'),
                "kdv_tutari": financials.get('kdv_tutari'),
                "kdv_matrah": financials.get('kdv_matrah'),
                "hesaplama_yontemi": financials.get('kdv_hesaplama_yontemi'),
                "tahmini_atandi": financials.get('kdv_tahmini_atandi'),
                "semantic_completions": header.get('semantic_completions'),
                "full_result": result
            })

        except Exception as e:
            print(f"HATA: {e}")
            results.append({
                "dosya": filename,
                "basarili": False,
                "hata": str(e)
            })

    # Save results
    output = {
        "test_tarihi": datetime.now().isoformat(),
        "test_turu": "KDV Heuristics",
        "dosya_sayisi": len(results),
        "sonuclar": results
    }

    output_file = Path(__file__).parent / "test_kdv_heuristics_sonuclari.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"Sonuclar: {output_file}")
    print("=" * 70)

    return results


def main():
    print("=" * 70)
    print("KDV MATHEMATICAL HEURISTICS TEST")
    print("FLO Document 7.1 and 7.2 Implementation")
    print("=" * 70)

    # Test semantic completer
    test_semantic_completer()

    # Test reverse calculation
    test_reverse_calculation()

    # Analyze specific files
    results = analyze_specific_files()

    # Summary
    print("\n" + "=" * 70)
    print("OZET")
    print("=" * 70)

    successful = [r for r in results if r.get('basarili')]
    print(f"Toplam: {len(results)}")
    print(f"Basarili: {len(successful)}")

    for r in successful:
        kdv_info = f"KDV %{r.get('kdv_orani', 'N/A')}"
        method = r.get('hesaplama_yontemi', 'N/A')
        estimated = "TAHMINI" if r.get('tahmini_atandi') else "KESIN"
        print(f"  {r['dosya'][:40]}: {kdv_info} ({method}) [{estimated}]")


if __name__ == "__main__":
    main()
