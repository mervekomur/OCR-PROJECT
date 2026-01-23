# -*- coding: utf-8 -*-
"""
Multi-Filter Ensemble Test
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from ocr_module import EnsembleOCR

print("="*60)
print("MULTI-FILTER ENSEMBLE TEST")
print("="*60)

# Test dosyasi
test_file = "data/fis1.jpeg"

print(f"\nTest: {test_file}")
print("-"*60)

# Multi-filter KAPALI
print("\n[1] Multi-filter KAPALI:")
ocr_off = EnsembleOCR(multi_filter=False)
result_off = ocr_off.process(test_file)
print(f"    Tarih:  {result_off.get('tarih') or '-'} ({result_off.get('tarih_kaynak', '-')})")
print(f"    Toplam: {result_off.get('toplam') or '-'} ({result_off.get('toplam_kaynak', '-')})")

# Multi-filter ACIK
print("\n[2] Multi-filter ACIK:")
ocr_on = EnsembleOCR(multi_filter=True)
result_on = ocr_on.process(test_file)
print(f"    Tarih:  {result_on.get('tarih') or '-'} ({result_on.get('tarih_kaynak', '-')})")
print(f"    Toplam: {result_on.get('toplam') or '-'} ({result_on.get('toplam_kaynak', '-')})")

# Karsilastirma
print("\n" + "="*60)
print("KARSILASTIRMA")
print("="*60)
print(f"\nBeklenen: Tarih=05.08.2025, Toplam=242.00")
print(f"\n{'Mod':<20} | {'Tarih':<15} | {'Toplam':<10}")
print("-"*50)
print(f"{'Multi-filter OFF':<20} | {result_off.get('tarih') or '-':<15} | {result_off.get('toplam') or '-'}")
print(f"{'Multi-filter ON':<20} | {result_on.get('tarih') or '-':<15} | {result_on.get('toplam') or '-'}")

print("\n" + "="*60)
