# -*- coding: utf-8 -*-
"""
Raw Image Scanner + Ensemble Test
SAM olmadan fis alani tespiti
"""

import os
import sys
import cv2
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from ocr_module import ReceiptScanner, EnsembleOCR

print("="*70)
print("RAW IMAGE TEST - Scanner + Ensemble (SAM Alternatifi)")
print("="*70)

project_root = os.path.dirname(os.path.abspath(__file__))

# Test dosyalari
test_files = [
    "fis1.jpeg",
    "fis4.png",
    "fis6.png",
    "fis7.png",
    "fis9.png",
    "fis16.jpeg",
]

# Cikti klasoru
output_dir = os.path.join(project_root, "results", "raw_scan_test")
os.makedirs(output_dir, exist_ok=True)

# Scanner ve OCR
scanner = ReceiptScanner()
ocr = EnsembleOCR(use_gpu=False, auto_scan=False)  # Manuel kontrol icin

print(f"\nCikti: {output_dir}")
print(f"Test sayisi: {len(test_files)}")

# Sonuclar
results = []

print("\n" + "="*70)
print("ADIM 1: SCANNER (Fis Alani Tespiti)")
print("="*70)

for filename in test_files:
    filepath = os.path.join(project_root, "data", filename)

    if not os.path.exists(filepath):
        print(f"\n[ATLA] {filename} bulunamadi")
        continue

    print(f"\n>>> {filename}")

    # Orijinal boyut
    orig = cv2.imread(filepath)
    orig_h, orig_w = orig.shape[:2]
    print(f"  Orijinal: {orig_w}x{orig_h}")

    # Scan (combined: contour + color)
    scanned, debug = scanner.scan_combined(filepath, return_debug=True)

    result = {
        'dosya': filename,
        'orijinal_boyut': f"{orig_w}x{orig_h}",
        'scan_basarili': debug['found'],
    }

    if debug['found']:
        scan_h, scan_w = scanned.shape[:2]
        result['scan_boyut'] = f"{scan_w}x{scan_h}"
        method = debug.get('method', 'unknown')
        result['method'] = method
        print(f"  [OK] Fis bulundu: {scan_w}x{scan_h} (metod: {method})")

        # Kaydet
        scan_path = os.path.join(output_dir, f"scanned_{filename}")
        cv2.imwrite(scan_path, scanned)
        result['scan_path'] = scan_path
        print(f"  Kaydedildi: scanned_{filename}")
    else:
        result['scan_boyut'] = None
        result['method'] = None
        tried = debug.get('tried', [])
        print(f"  [!] Bulunamadi (denenen: {tried})")

    results.append(result)

print("\n" + "="*70)
print("ADIM 2: ENSEMBLE OCR (Karsilastirma)")
print("="*70)

for r in results:
    filename = r['dosya']
    filepath = os.path.join(project_root, "data", filename)

    print(f"\n>>> {filename}")
    print("-"*50)

    # Orijinal uzerinde OCR
    print("  [Orijinal]")
    orig_result = ocr.process(filepath)
    r['orig_tarih'] = orig_result.get('tarih')
    r['orig_toplam'] = orig_result.get('toplam')
    print(f"    Tarih:  {r['orig_tarih'] or '-'}")
    print(f"    Toplam: {r['orig_toplam'] or '-'}")

    # Scan basariliysa, scan uzerinde de OCR
    if r['scan_basarili'] and r.get('scan_path'):
        print("  [Scanned]")
        scan_result = ocr.process(r['scan_path'])
        r['scan_tarih'] = scan_result.get('tarih')
        r['scan_toplam'] = scan_result.get('toplam')
        print(f"    Tarih:  {r['scan_tarih'] or '-'}")
        print(f"    Toplam: {r['scan_toplam'] or '-'}")
    else:
        r['scan_tarih'] = None
        r['scan_toplam'] = None

# Ozet tablo
print("\n" + "="*70)
print("SONUC TABLOSU")
print("="*70)

print(f"\n{'Dosya':<12} | {'Scan':<6} | {'Orijinal':<20} | {'Scanned':<20}")
print(f"{'':<12} | {'':<6} | {'Tarih':<9} {'Toplam':<9} | {'Tarih':<9} {'Toplam':<9}")
print("-"*70)

scan_ok = 0
for r in results:
    scan_status = "OK" if r['scan_basarili'] else "-"
    if r['scan_basarili']:
        scan_ok += 1

    ot = r.get('orig_tarih') or '-'
    op = f"{r['orig_toplam']:.0f}" if r.get('orig_toplam') else '-'
    st = r.get('scan_tarih') or '-'
    sp = f"{r['scan_toplam']:.0f}" if r.get('scan_toplam') else '-'

    print(f"{r['dosya']:<12} | {scan_status:<6} | {ot:<9} {op:<9} | {st:<9} {sp:<9}")

print("-"*70)
print(f"Scanner basari: {scan_ok}/{len(results)} fiste alan bulundu")

print("\n" + "="*70)
print("TEST TAMAMLANDI")
print("="*70)
print(f"\nScanned resimler: {output_dir}")
