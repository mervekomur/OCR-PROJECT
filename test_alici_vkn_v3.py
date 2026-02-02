"""
Test v3: Alıcı VKN Kontrolü

Yeni Mantık:
1. Sadece ALICI bölümündeki VKN kontrol edilir
2. Alıcı VKN 4 FLO şirketinden biri olmalı
3. Kişisel isim tespit edilirse → RED
4. TODO: Çalışan eşleştirme (ileride eklenecek)
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

from engines.claude_vision_engine import ClaudeVisionEngine, FLO_COMPANIES


def analyze_receipt(engine: ClaudeVisionEngine, image_path: str) -> dict:
    """Analyze a single receipt with new logic."""
    filename = Path(image_path).name

    try:
        result = engine.extract_to_sap_json(image_path)

        header = result.get('header', {})
        financials = result.get('financials', {})
        validation = result.get('validation', {})
        metadata = result.get('metadata', {})

        # Satıcı bilgileri
        satici = header.get('satici', {})
        satici_vkn = satici.get('vkn', 'N/A')
        satici_adi = satici.get('firma_adi', 'N/A')

        # Alıcı bilgileri
        alici = header.get('alici', {})
        alici_vkn = alici.get('vkn_veya_tckn', 'YOK')
        alici_adi = alici.get('unvan_veya_ad', 'N/A')

        return {
            "dosya": filename,
            "basarili": True,
            "belge_turu": header.get('belge_turu', 'N/A'),
            "dil": metadata.get('detected_language', 'N/A'),
            # Satıcı
            "satici_vkn": satici_vkn,
            "satici_adi": satici_adi[:30] if satici_adi else 'N/A',
            # Alıcı
            "alici_vkn": alici_vkn,
            "alici_adi": alici_adi[:30] if alici_adi else 'N/A',
            "alici_tipi": validation.get('alici_tipi', 'N/A'),
            # Validasyon
            "alici_gecerli": validation.get('alici_gecerli', False),
            "alici_flo_sirketi": validation.get('alici_flo_sirketi'),
            "islem_durumu": validation.get('islem_durumu', 'N/A'),
            "red_sebebi": validation.get('red_sebebi'),
            # Finansal
            "brut_tutar": financials.get('brut_tutar'),
            "para_birimi": financials.get('para_birimi', 'TRY'),
        }

    except Exception as e:
        return {
            "dosya": filename,
            "basarili": False,
            "hata": str(e)
        }


def main():
    print("=" * 70)
    print("TEST v3: ALICI VKN KONTROLÜ ve ÇALIŞAN EŞLEŞTİRME")
    print("=" * 70)
    print("\nFLO Şirketleri (Geçerli Alıcı VKN'ler):")
    for vkn, info in FLO_COMPANIES.items():
        print(f"  {vkn} → {info['name']}")
    print("-" * 70)

    engine = ClaudeVisionEngine()

    folder = Path("C:/Users/MK/Desktop/OCR PROJECT/Makbuz- Taksi fişi")
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    # Tüm dosyaları test et
    print(f"\nTest edilecek dosya: {len(images)}")
    print("-" * 70)

    results = []

    # Sayaçlar
    flo_alici = 0
    sahis_fatura = 0
    perakende = 0
    diger_sirket = 0

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {image_path.name}")

        result = analyze_receipt(engine, str(image_path))
        results.append(result)

        if result.get('basarili'):
            # Satıcı bilgisi
            print(f"   SATICI: {result['satici_adi']} (VKN: {result['satici_vkn']})")

            # Alıcı bilgisi
            alici_vkn = result['alici_vkn']
            alici_adi = result['alici_adi']
            print(f"   ALICI:  {alici_adi} (VKN: {alici_vkn})")

            # Validasyon sonucu
            durum = result['islem_durumu']
            if durum == "ONAYLANDI":
                if result.get('alici_flo_sirketi'):
                    print(f"   SONUÇ:  ✅ ONAYLANDI - FLO Şirketi: {result['alici_flo_sirketi']}")
                    flo_alici += 1
                else:
                    print(f"   SONUÇ:  ✅ ONAYLANDI - Perakende Fiş")
                    perakende += 1
            else:
                sebep = result.get('red_sebebi', 'Bilinmiyor')
                print(f"   SONUÇ:  ❌ RED - {sebep}")

                if result.get('alici_tipi') == 'GERCEK_KISI':
                    sahis_fatura += 1
                elif result.get('alici_tipi') == 'TUZEL_KISI':
                    diger_sirket += 1
        else:
            print(f"   HATA: {result.get('hata', 'Bilinmeyen')}")

    # Özet
    print("\n" + "=" * 70)
    print("ÖZET RAPOR")
    print("=" * 70)

    successful = [r for r in results if r.get('basarili')]
    print(f"\nToplam İşlenen: {len(results)}")
    print(f"Başarılı: {len(successful)}")

    print(f"\n--- ALICI DAĞILIMI ---")
    print(f"FLO Şirketi (ONAY): {flo_alici}")
    print(f"Perakende Fiş (ONAY): {perakende}")
    print(f"Şahıs Faturası (RED): {sahis_fatura}")
    print(f"Diğer Şirket (RED): {diger_sirket}")

    # Detaylı sonuçları kaydet
    output = {
        "test_tarihi": datetime.now().isoformat(),
        "versiyon": "v3",
        "toplam": len(results),
        "flo_alici": flo_alici,
        "perakende": perakende,
        "sahis_fatura": sahis_fatura,
        "diger_sirket": diger_sirket,
        "detaylar": results
    }

    output_file = Path(__file__).parent / "test_alici_vkn_v3_sonuclari.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSonuçlar: {output_file}")


if __name__ == "__main__":
    main()
