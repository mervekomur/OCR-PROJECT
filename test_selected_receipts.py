# -*- coding: utf-8 -*-
"""
Secili fisler icin Multi-Filter Ensemble testi
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from ocr_module import EnsembleOCR

# Ana fis dosyalari (processed olmayanlar)
files = [
    'data/fis1.jpeg',
    'data/fis2.jpeg',
    'data/fis3.jpg',
    'data/fis4.png',
    'data/fis5.png',
    'data/fis6.jpeg',
]

# Var olanlari filtrele
files = [f for f in files if os.path.exists(f)]

print(f'Test edilecek fisler: {len(files)}')
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
        'tarih_kaynak': tk or '-',
        'toplam_kaynak': pk or '-'
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
