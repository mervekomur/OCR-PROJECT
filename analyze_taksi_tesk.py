"""
Taksi Fişi TESK Analizi
FLO İş Kuralı: TESK varsa KDV %0, yoksa KDV %20

Kaynak: v2FLO_Motion_Masraf_Modulu_Is_Analizi - Şekil 4.2 ve Bölüm 8.4
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


def analyze_receipt(engine: ClaudeVisionEngine, image_path: str) -> dict:
    """Analyze a single receipt for TESK rule."""
    filename = Path(image_path).name

    try:
        result = engine.extract_to_sap_json(image_path)

        # Extract key info
        header = result.get('header', {})
        financials = result.get('financials', {})
        validation = result.get('validation', {})
        metadata = result.get('metadata', {})

        return {
            "dosya": filename,
            "basarili": True,
            "belge_turu": header.get('belge_turu', 'N/A'),
            "dil": metadata.get('detected_language', 'N/A'),
            "tesk_tespit": validation.get('tesk_tespit', False),
            "kdv_orani": financials.get('kdv_orani'),
            "kdv_tutari": financials.get('kdv_tutari'),
            "brut_tutar": financials.get('brut_tutar'),
            "para_birimi": financials.get('para_birimi', 'TRY'),
            "firma_adi": header.get('satici', {}).get('firma_adi', 'N/A'),
            "hesap_kodu": validation.get('hesap_bilgisi', {}).get('hesap_kodu') if validation.get('hesap_bilgisi') else None,
            "hesap_adi": validation.get('hesap_bilgisi', {}).get('hesap_adi') if validation.get('hesap_bilgisi') else None,
            "anahtar_kelimeler": validation.get('esleslen_anahtar_kelimeler', []),
            "islem_suresi": metadata.get('processing_time', 0),
            "token": metadata.get('tokens_used', 0),
            "raw_text_preview": result.get('header', {}).get('belge_no', '')
        }

    except Exception as e:
        return {
            "dosya": filename,
            "basarili": False,
            "hata": str(e)
        }


def main():
    print("="*70)
    print("TAKSİ FİŞİ TESK ANALİZİ")
    print("Kural: TESK varsa KDV %0, yoksa KDV %20")
    print("="*70)

    # Initialize engine
    engine = ClaudeVisionEngine()

    # Get all images from the folder
    folder = Path("C:/Users/MK/Desktop/OCR PROJECT/Makbuz- Taksi fişi")

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    print(f"\nToplam dosya: {len(images)}")
    print("-"*70)

    results = []
    tesk_count = 0
    no_tesk_count = 0
    taksi_count = 0

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {image_path.name}")

        result = analyze_receipt(engine, str(image_path))
        results.append(result)

        if result.get('basarili'):
            tesk = "TESK VAR" if result.get('tesk_tespit') else "TESK YOK"
            kdv = result.get('kdv_orani', 'N/A')
            tutar = result.get('brut_tutar', 'N/A')
            pb = result.get('para_birimi', 'TRY')
            firma = result.get('firma_adi', 'N/A')[:30]
            belge = result.get('belge_turu', 'N/A')

            # Count TESK
            if result.get('tesk_tespit'):
                tesk_count += 1
            else:
                no_tesk_count += 1

            # Count taxi keywords
            keywords = result.get('anahtar_kelimeler', [])
            if 'taksi' in keywords or 'taxi' in keywords or 'tesk' in keywords:
                taksi_count += 1

            print(f"   Belge: {belge} | {tesk} | KDV: %{kdv} | Tutar: {tutar} {pb}")
            print(f"   Firma: {firma}")
            if result.get('hesap_adi'):
                print(f"   Hesap: {result.get('hesap_adi')} ({result.get('hesap_kodu')})")
        else:
            print(f"   HATA: {result.get('hata', 'Bilinmeyen hata')}")

    # Summary
    print("\n" + "="*70)
    print("ÖZET RAPOR")
    print("="*70)

    successful = [r for r in results if r.get('basarili')]
    failed = [r for r in results if not r.get('basarili')]

    print(f"\nToplam İşlenen: {len(results)}")
    print(f"Başarılı: {len(successful)}")
    print(f"Başarısız: {len(failed)}")

    print(f"\n--- TESK ANALİZİ ---")
    print(f"TESK Tespit Edilen: {tesk_count}")
    print(f"TESK Bulunmayan: {no_tesk_count}")
    print(f"Taksi Anahtar Kelimesi: {taksi_count}")

    # TESK rule compliance
    print(f"\n--- KDV KURAL UYUMU ---")
    tesk_kdv0 = [r for r in successful if r.get('tesk_tespit') and r.get('kdv_orani') == 0]
    tesk_kdv_other = [r for r in successful if r.get('tesk_tespit') and r.get('kdv_orani') != 0]
    no_tesk_kdv20 = [r for r in successful if not r.get('tesk_tespit') and r.get('kdv_orani') == 20]
    no_tesk_kdv_other = [r for r in successful if not r.get('tesk_tespit') and r.get('kdv_orani') not in [20, None] and r.get('belge_turu') == 'FIS']

    print(f"TESK + KDV %0 (DOĞRU): {len(tesk_kdv0)}")
    print(f"TESK + KDV %0 Değil (YANLIŞ): {len(tesk_kdv_other)}")
    print(f"TESK Yok + KDV %20 (DOĞRU): {len(no_tesk_kdv20)}")

    # Group by document type
    print(f"\n--- BELGE TÜRÜ DAĞILIMI ---")
    doc_types = {}
    for r in successful:
        dt = r.get('belge_turu', 'N/A')
        doc_types[dt] = doc_types.get(dt, 0) + 1
    for dt, count in sorted(doc_types.items(), key=lambda x: -x[1]):
        print(f"  {dt}: {count}")

    # Group by language
    print(f"\n--- DİL DAĞILIMI ---")
    languages = {}
    for r in successful:
        lang = r.get('dil', 'N/A')
        languages[lang] = languages.get(lang, 0) + 1
    for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")

    # TESK detected files
    print(f"\n--- TESK TESPİT EDİLEN DOSYALAR ---")
    tesk_files = [r for r in successful if r.get('tesk_tespit')]
    for r in tesk_files:
        print(f"  {r['dosya']}: KDV %{r.get('kdv_orani')} | {r.get('brut_tutar')} {r.get('para_birimi')}")

    # Save detailed results
    output = {
        "analiz_tarihi": datetime.now().isoformat(),
        "toplam_dosya": len(results),
        "basarili": len(successful),
        "basarisiz": len(failed),
        "tesk_tespit": tesk_count,
        "tesk_yok": no_tesk_count,
        "taksi_anahtar_kelime": taksi_count,
        "detayli_sonuclar": results
    }

    output_file = Path(__file__).parent / "taksi_tesk_analiz_sonuclari.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n📄 Detaylı sonuçlar: {output_file}")


if __name__ == "__main__":
    main()
