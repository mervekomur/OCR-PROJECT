# -*- coding: utf-8 -*-
"""
ReceiptScanner Test - Arka plan temizleme testi
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ocr_module import ReceiptScanner
import cv2

print("="*60)
print("RECEIPT SCANNER TEST")
print("="*60)

project_root = os.path.dirname(os.path.abspath(__file__))
scanner = ReceiptScanner()

# Test resimleri (ham, kesilmemis)
test_images = [
    os.path.join(project_root, "data", "fis1.jpeg"),
    os.path.join(project_root, "data", "fis13.jpeg"),
    os.path.join(project_root, "data", "fis16.jpeg"),
]

output_dir = os.path.join(project_root, "results", "scanner_test")
os.makedirs(output_dir, exist_ok=True)

print(f"\nCikti klasoru: {output_dir}")

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"\n[ATLA] {img_path} bulunamadi")
        continue

    filename = os.path.basename(img_path)
    print(f"\n>>> {filename}")

    # Scan
    result, debug = scanner.scan(img_path, return_debug=True)

    if debug['found']:
        print(f"  [OK] Fis bulundu!")
        print(f"  Yeni boyut: {debug.get('warped_size', 'N/A')}")

        # Kaydet
        output_path = os.path.join(output_dir, f"scanned_{filename}")
        cv2.imwrite(output_path, result)
        print(f"  Kaydedildi: {output_path}")
    else:
        print(f"  [!] Fis kenari bulunamadi, orijinal kullanilacak")

print("\n" + "="*60)
print("TEST TAMAMLANDI")
print("="*60)
