# -*- coding: utf-8 -*-
"""
=============================================================================
OCR Optimization Test
=============================================================================

SAM donuk - Sadece OCR preprocessing karsilastirmasi.

Girdi: Desktop/test_crops klasorundeki hazir kesilmis resimler
Cikti: Terminal'de 3 varyasyonun sonuclari

Varyasyonlar:
    A: Raw (islem yok)
    B: Adaptive Threshold
    C: Scale x2 + Bilateral Filter

Kullanim:
    python ocr_optimization.py
"""

import os
import cv2
import numpy as np
from glob import glob

# =============================================================================
# YAPILANDIRMA
# =============================================================================

# Girdi klasoru
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "test_crops")

# EasyOCR ayarlari (RAM dostu)
OCR_LANGUAGES = ['tr', 'en']
OCR_GPU = False

# =============================================================================
# PREPROCESSING VARYASYONLARI
# =============================================================================

def preprocess_raw(image):
    """Varyasyon A: Hicbir islem yok (Raw)."""
    # Grayscale yap (EasyOCR icin)
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def preprocess_adaptive_threshold(image):
    """Varyasyon B: Adaptive Threshold."""
    # Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Adaptive Threshold
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    return binary


def preprocess_scale_bilateral(image):
    """Varyasyon C: Scale x2 + Bilateral Filter."""
    # Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Scale x2 (buyut)
    h, w = gray.shape[:2]
    scaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Bilateral Filter (kenar koruyarak yumusatma)
    filtered = cv2.bilateralFilter(scaled, d=9, sigmaColor=75, sigmaSpace=75)

    return filtered


# =============================================================================
# OCR
# =============================================================================

_ocr_reader = None

def get_ocr_reader():
    """EasyOCR reader (singleton, RAM dostu)."""
    global _ocr_reader

    if _ocr_reader is not None:
        return _ocr_reader

    try:
        import easyocr
        print("EasyOCR yukleniyor (GPU=False, RAM dostu)...")
        _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=OCR_GPU, verbose=False)
        return _ocr_reader
    except ImportError:
        print("HATA: easyocr yuklu degil!")
        print("Kur: pip install easyocr")
        return None


def read_text(image):
    """Resimden metin oku."""
    reader = get_ocr_reader()
    if reader is None:
        return ""

    results = reader.readtext(image, detail=0)
    return '\n'.join(results)


# =============================================================================
# TEST FONKSIYONU
# =============================================================================

def test_image(image_path, output_dir):
    """Tek bir resim uzerinde 3 varyasyonu test et."""

    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]
    print(f"\n{'='*60}")
    print(f"RESIM: {filename}")
    print('='*60)

    # Resmi yukle
    image = cv2.imread(image_path)
    if image is None:
        print("  HATA: Resim yuklenemedi!")
        return None

    print(f"  Boyut: {image.shape[:2]}")

    results = {}

    # --- Varyasyon A: Raw ---
    print("\n  [A] RAW (islem yok)...")
    img_a = preprocess_raw(image)
    text_a = read_text(img_a)
    chars_a = len([c for c in text_a if c.strip()])
    results['A_raw'] = {'text': text_a, 'chars': chars_a}
    print(f"      Karakter: {chars_a}")

    # --- Varyasyon B: Adaptive Threshold ---
    print("\n  [B] ADAPTIVE THRESHOLD...")
    img_b = preprocess_adaptive_threshold(image)
    text_b = read_text(img_b)
    chars_b = len([c for c in text_b if c.strip()])
    results['B_adaptive'] = {'text': text_b, 'chars': chars_b}
    print(f"      Karakter: {chars_b}")

    # --- Varyasyon C: Scale + Bilateral ---
    print("\n  [C] SCALE x2 + BILATERAL FILTER...")
    img_c = preprocess_scale_bilateral(image)
    text_c = read_text(img_c)
    chars_c = len([c for c in text_c if c.strip()])
    results['C_scale'] = {'text': text_c, 'chars': chars_c}
    print(f"      Karakter: {chars_c}")

    # --- Goruntuleri kaydet ---
    cv2.imwrite(os.path.join(output_dir, f'{name}_A_raw.jpg'), img_a)
    cv2.imwrite(os.path.join(output_dir, f'{name}_B_adaptive.jpg'), img_b)
    cv2.imwrite(os.path.join(output_dir, f'{name}_C_scale.jpg'), img_c)
    print(f"\n  Goruntuler kaydedildi: {name}_A/B/C_*.jpg")

    # --- Karsilastirma ---
    print("\n  " + "-"*50)
    print("  KARSILASTIRMA:")
    print("  " + "-"*50)

    best_key = max(results.keys(), key=lambda k: results[k]['chars'])
    for key, data in results.items():
        marker = " <-- EN IYI" if key == best_key else ""
        print(f"    {key}: {data['chars']} karakter{marker}")

    # --- Metinleri goster ---
    print("\n  " + "-"*50)
    print("  OKUNAN METINLER:")
    print("  " + "-"*50)

    for key, data in results.items():
        print(f"\n  [{key}]:")
        # Ilk 500 karakteri goster
        text_preview = data['text'][:500]
        for line in text_preview.split('\n'):
            print(f"    {line}")
        if len(data['text']) > 500:
            print("    ...")

    return results


# =============================================================================
# ANA FONKSIYON
# =============================================================================

def main():
    """Ana fonksiyon."""

    print("\n" + "="*60)
    print("OCR OPTIMIZATION TEST")
    print("="*60)
    print(f"Girdi klasoru: {INPUT_DIR}")
    print(f"EasyOCR: diller={OCR_LANGUAGES}, gpu={OCR_GPU}")

    # Klasoru kontrol et
    if not os.path.exists(INPUT_DIR):
        print(f"\nHATA: Klasor bulunamadi: {INPUT_DIR}")
        print("Lutfen Desktop/test_crops klasorunu olusturup icine kesilmis fis resimleri koy.")
        return

    # Resimleri bul
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    files = []
    for pattern in patterns:
        files.extend(glob(os.path.join(INPUT_DIR, pattern)))
    files = sorted(set(files))

    if not files:
        print(f"\nHATA: Klasorde resim bulunamadi: {INPUT_DIR}")
        print("Desteklenen formatlar: jpg, jpeg, png")
        return

    print(f"\n{len(files)} resim bulundu")

    # Cikti klasoru (islenmiş goruntuler icin)
    output_dir = os.path.join(INPUT_DIR, "comparison")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Islenmiş goruntuler: {output_dir}")

    # EasyOCR'u bir kere yukle
    reader = get_ocr_reader()
    if reader is None:
        return

    # Her resmi test et
    all_results = {}
    for filepath in files:
        result = test_image(filepath, output_dir)
        if result:
            all_results[os.path.basename(filepath)] = result

    # --- GENEL OZET ---
    if all_results:
        print("\n" + "="*60)
        print("GENEL OZET")
        print("="*60)

        totals = {'A_raw': 0, 'B_adaptive': 0, 'C_scale': 0}
        wins = {'A_raw': 0, 'B_adaptive': 0, 'C_scale': 0}

        for filename, results in all_results.items():
            best_key = max(results.keys(), key=lambda k: results[k]['chars'])
            wins[best_key] += 1
            for key in totals:
                totals[key] += results[key]['chars']

        print("\nToplam karakter sayisi:")
        for key, total in totals.items():
            print(f"  {key}: {total}")

        print("\nEn iyi sonuc sayisi:")
        for key, win in wins.items():
            print(f"  {key}: {win} resimde en iyi")

        best_overall = max(totals.keys(), key=lambda k: totals[k])
        print(f"\n>>> KAZANAN: {best_overall} <<<")


if __name__ == '__main__':
    main()
