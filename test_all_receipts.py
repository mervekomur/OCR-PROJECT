# -*- coding: utf-8 -*-
"""
Tum fisler icin Multi-Filter Ensemble testi
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from glob import glob
from ocr_module import EnsembleOCR

# Tum fisleri bul
files = sorted(glob('data/fis*.jpg') + glob('data/fis*.jpeg') + glob('data/fis*.png'))
print(f'Bulunan fisler: {len(files)}')
for f in files:
    print(f'  - {os.path.basename(f)}')

print('\n' + '='*70)
print('MULTI-FILTER ENSEMBLE TEST - TUM FISLER')
print('='*70)

ocr = EnsembleOCR(multi_filter=True)

results = []
for i, f in enumerate(files, 1):
    print(f'\n[{i}/{len(files)}] {os.path.basename(f)}...')
    result = ocr.process(f)

    tarih = result.get('tarih') or '-'
    toplam = result.get('toplam')
    toplam_str = f'{toplam:.2f}' if toplam else '-'
    tk = result.get('tarih_kaynak', '-')
    pk = result.get('toplam_kaynak', '-')

    print(f'    Tarih:  {tarih} ({tk})')
    print(f'    Toplam: {toplam_str} ({pk})')

    results.append({
        'dosya': os.path.basename(f),
        'tarih': tarih,
        'toplam': toplam_str,
        'tarih_kaynak': tk,
        'toplam_kaynak': pk
    })

# Ozet tablo
print('\n' + '='*70)
print('SONUC TABLOSU')
print('='*70)
print(f"{'Dosya':<15} | {'Tarih':<12} | {'Toplam':<10} | {'T.Kaynak':<10} | {'P.Kaynak':<10}")
print('-'*70)
for r in results:
    print(f"{r['dosya']:<15} | {r['tarih']:<12} | {r['toplam']:<10} | {r['tarih_kaynak']:<10} | {r['toplam_kaynak']:<10}")

# Istatistik
tarih_found = len([r for r in results if r['tarih'] != '-'])
toplam_found = len([r for r in results if r['toplam'] != '-'])
print('\n' + '='*70)
print(f'ISTATISTIK: Tarih {tarih_found}/{len(results)}, Toplam {toplam_found}/{len(results)}')
print('='*70)
