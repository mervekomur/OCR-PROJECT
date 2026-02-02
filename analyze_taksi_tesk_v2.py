"""
Taksi Fişi TESK Analizi v2
Yeni Mantık Akışı:

1. Dil Kontrolü: Belge Türkçe mi?
   - Hayır: YURT DIŞI MASRAFI, TESK kuralını pas geç
   - Evet: Adım 2'ye geç

2. Kategori Kontrolü: Taksi Fişi mi?
   - Evet + TESK var: KDV %0
   - Evet + TESK yok: KDV %20
   - Hayır: Kategori bazlı KDV tablosu
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


def analyze_receipt(engine: ClaudeVisionEngine, image_path: str) -> dict:
    """Analyze a single receipt."""
    filename = Path(image_path).name

    try:
        result = engine.extract_to_sap_json(image_path)

        header = result.get('header', {})
        financials = result.get('financials', {})
        validation = result.get('validation', {})
        metadata = result.get('metadata', {})

        return {
            "dosya": filename,
            "basarili": True,
            "belge_turu": header.get('belge_turu', 'N/A'),
            "dil": metadata.get('detected_language', 'N/A'),
            "yurt_disi": metadata.get('yurt_disi_masrafi', False),
            "taksi_fisi": metadata.get('is_taxi', False),
            "tesk_tespit": metadata.get('tesk_detected', False),
            "tesk_uygulandı": metadata.get('tesk_applied', False),
            "kdv_orani": financials.get('kdv_orani'),
            "kdv_notu": validation.get('kdv_notu', ''),
            "brut_tutar": financials.get('brut_tutar'),
            "para_birimi": financials.get('para_birimi', 'TRY'),
            "firma_adi": header.get('satici', {}).get('firma_adi', 'N/A'),
            "hesap_kodu": validation.get('hesap_bilgisi', {}).get('hesap_kodu') if validation.get('hesap_bilgisi') else None,
            "hesap_adi": validation.get('hesap_bilgisi', {}).get('hesap_adi') if validation.get('hesap_bilgisi') else None,
            "islem_suresi": metadata.get('processing_time', 0),
        }

    except Exception as e:
        return {
            "dosya": filename,
            "basarili": False,
            "hata": str(e)
        }


def main():
    print("="*70)
    print("TAKSİ FİŞİ TESK ANALİZİ v2")
    print("Yeni Mantık: Dil -> Kategori -> TESK")
    print("="*70)

    engine = ClaudeVisionEngine()

    folder = Path("C:/Users/MK/Desktop/OCR PROJECT/Makbuz- Taksi fişi")
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    print(f"\nToplam dosya: {len(images)}")
    print("-"*70)

    results = []

    # Counters
    yurt_disi_count = 0
    turkiye_count = 0
    taksi_tesk_count = 0
    taksi_no_tesk_count = 0
    diger_count = 0

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {image_path.name}")

        result = analyze_receipt(engine, str(image_path))
        results.append(result)

        if result.get('basarili'):
            dil = result.get('dil', 'N/A')
            yurt_disi = result.get('yurt_disi', False)
            taksi = result.get('taksi_fisi', False)
            tesk = result.get('tesk_tespit', False)
            tesk_applied = result.get('tesk_uygulandı', False)
            kdv = result.get('kdv_orani', 'N/A')
            tutar = result.get('brut_tutar', 'N/A')
            pb = result.get('para_birimi', 'TRY')
            kdv_notu = result.get('kdv_notu', '')

            # Kategorize
            if yurt_disi:
                kategori = "YURT DIŞI"
                yurt_disi_count += 1
            elif taksi:
                turkiye_count += 1
                if tesk_applied:
                    kategori = "TAKSİ+TESK"
                    taksi_tesk_count += 1
                else:
                    kategori = "TAKSİ (TESK YOK)"
                    taksi_no_tesk_count += 1
            else:
                kategori = "DİĞER"
                turkiye_count += 1
                diger_count += 1

            print(f"   Dil: {dil} | {kategori} | KDV: %{kdv} | {tutar} {pb}")
            if kdv_notu:
                print(f"   Not: {kdv_notu}")
        else:
            print(f"   HATA: {result.get('hata', 'Bilinmeyen')}")

    # Summary
    print("\n" + "="*70)
    print("ÖZET RAPOR")
    print("="*70)

    successful = [r for r in results if r.get('basarili')]
    failed = [r for r in results if not r.get('basarili')]

    print(f"\nToplam İşlenen: {len(results)}")
    print(f"Başarılı: {len(successful)}")
    print(f"Başarısız: {len(failed)}")

    print(f"\n--- MANTIK AKIŞI SONUÇLARI ---")
    print(f"YURT DIŞI MASRAFI: {yurt_disi_count}")
    print(f"TÜRKİYE MASRAFI: {turkiye_count}")
    print(f"  - Taksi + TESK (KDV %0): {taksi_tesk_count}")
    print(f"  - Taksi TESK Yok (KDV %20): {taksi_no_tesk_count}")
    print(f"  - Diğer Kategoriler: {diger_count}")

    # Detailed breakdown
    print(f"\n--- YURT DIŞI MASRAFLARI ---")
    for r in successful:
        if r.get('yurt_disi'):
            print(f"  {r['dosya']}: {r['dil']} | KDV %{r['kdv_orani']} | {r['brut_tutar']} {r['para_birimi']}")

    print(f"\n--- TAKSİ + TESK (KDV %0) ---")
    for r in successful:
        if r.get('taksi_fisi') and r.get('tesk_uygulandı'):
            print(f"  {r['dosya']}: {r['brut_tutar']} {r['para_birimi']}")

    print(f"\n--- TAKSİ TESK YOK (KDV %20) ---")
    for r in successful:
        if r.get('taksi_fisi') and not r.get('tesk_uygulandı') and not r.get('yurt_disi'):
            print(f"  {r['dosya']}: KDV %{r['kdv_orani']} | {r['brut_tutar']} {r['para_birimi']}")

    # Save results
    output = {
        "analiz_tarihi": datetime.now().isoformat(),
        "versiyon": "v2",
        "mantik": "Dil -> Kategori -> TESK",
        "toplam_dosya": len(results),
        "yurt_disi_masrafi": yurt_disi_count,
        "turkiye_masrafi": turkiye_count,
        "taksi_tesk": taksi_tesk_count,
        "taksi_no_tesk": taksi_no_tesk_count,
        "diger": diger_count,
        "detayli_sonuclar": results
    }

    output_file = Path(__file__).parent / "taksi_tesk_analiz_v2.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSonuçlar: {output_file}")


if __name__ == "__main__":
    main()
