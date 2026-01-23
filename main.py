# -*- coding: utf-8 -*-
"""
=============================================================================
FIS OCR - Final Pipeline (Entegre)
=============================================================================

SAM (Segmentasyon) + ReceiptOCR (Akilli OCR) Entegrasyonu

Kullanim:
    python main.py                      # Tum fisleri isle
    python main.py data/fis1.jpeg       # Tek fis isle

Ciktilar (results/ klasoru):
    - [dosyaadi]_crop.jpg      (Kirpilmis fis)
    - [dosyaadi]_text.txt      (OCR metni + cikartilan veriler)
"""

import os
import sys
import gc
import json
import cv2
import numpy as np
import torch
from glob import glob
from datetime import datetime

# OCR Modulu (kendi gelistirdigimiz)
from ocr_module import ReceiptOCR

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# SAM SEGMENTASYON
# =============================================================================

_mask_generator = None

def get_mask_generator():
    """SAM mask generator (singleton)."""
    global _mask_generator

    if _mask_generator is not None:
        return _mask_generator

    try:
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print("HATA: segment_anything yuklu degil!")
        return None

    model_path = os.path.join(PROJECT_ROOT, 'sam_vit_b.pth')
    if not os.path.exists(model_path):
        print(f"HATA: SAM model bulunamadi: {model_path}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM yukleniyor... ({device})")

    sam = sam_model_registry["vit_b"](checkpoint=model_path)
    sam.to(device)

    _mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=1000,
    )

    return _mask_generator


def find_receipt(image, mask_generator):
    """Resimde fisi bul ve koselerini dondur."""
    h, w = image.shape[:2]
    img_area = h * w
    center = (w // 2, h // 2)

    # Maskeleri uret
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)

    # En iyi maskeyi bul
    best_mask = None
    best_score = 0

    for mask_data in masks:
        area = mask_data['area']
        bbox = mask_data['bbox']

        # Alan filtresi
        area_ratio = area / img_area
        if area_ratio < 0.05 or area_ratio > 0.90:
            continue

        # Merkeze yakinlik
        cx = bbox[0] + bbox[2] // 2
        cy = bbox[1] + bbox[3] // 2
        dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
        max_dist = np.sqrt(center[0]**2 + center[1]**2)
        center_score = 1 - (dist / max_dist)

        stability = mask_data.get('stability_score', 0.5)
        score = area_ratio * 0.4 + center_score * 0.4 + stability * 0.2

        if score > best_score:
            best_score = score
            best_mask = mask_data

    if best_mask is None:
        return None

    # Maskeden koseleri cikar
    mask = best_mask['segmentation']
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rect)

    return np.int32(corners)


# =============================================================================
# WARP (PERSPEKTIF DUZELTME)
# =============================================================================

def warp_receipt(image, corners):
    """Fisi duzelt ve kirp."""

    # Koseleri sirala
    pts = corners.astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]      # sol-ust
    rect[2] = pts[np.argmax(s)]      # sag-alt
    rect[1] = pts[np.argmin(diff)]   # sag-ust
    rect[3] = pts[np.argmax(diff)]   # sol-alt

    # Boyutlari hesapla
    (tl, tr, br, bl) = rect

    w1 = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)
    w2 = np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2)
    width = int(max(w1, w2))

    h1 = np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2)
    h2 = np.sqrt((tl[0]-bl[0])**2 + (tl[1]-bl[1])**2)
    height = int(max(h1, h2))

    # Minimum boyut
    width = max(width, 100)
    height = max(height, 100)

    # Hedef noktalar
    dst = np.array([
        [0, 0],
        [width-1, 0],
        [width-1, height-1],
        [0, height-1]
    ], dtype="float32")

    # Warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    # Fis dikey olmali
    if warped.shape[1] > warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped


# =============================================================================
# OCR MOTORU (Singleton)
# =============================================================================

_ocr_engine = None

def get_ocr_engine():
    """ReceiptOCR motoru (singleton - RAM dostu)."""
    global _ocr_engine

    if _ocr_engine is None:
        print("  OCR motoru yukleniyor...")
        _ocr_engine = ReceiptOCR(languages=['tr', 'en'], use_gpu=False)

    return _ocr_engine


# =============================================================================
# ANA ISLEM
# =============================================================================

def process_receipt(image_path, output_dir):
    """Tek bir fisi isle: SAM -> Warp -> OCR -> Extract."""

    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]

    print(f"\n{'='*60}")
    print(f"Isleniyor: {filename}")
    print('='*60)

    # Resmi yukle
    image = cv2.imread(image_path)
    if image is None:
        print("  HATA: Resim yuklenemedi")
        return None

    # SAM ile fisi bul
    print("  [1/3] SAM segmentasyon...")
    mask_gen = get_mask_generator()
    if mask_gen is None:
        return None

    corners = find_receipt(image, mask_gen)
    if corners is None:
        print("  HATA: Fis bulunamadi")
        return None

    # Warp
    print("  [2/3] Warp & Crop...")
    warped = warp_receipt(image, corners)
    print(f"        Boyut: {warped.shape[:2]}")

    # Crop'u kaydet
    crop_path = os.path.join(output_dir, f'{name}_crop.jpg')
    cv2.imwrite(crop_path, warped)

    # OCR + Akilli Cikarim
    print("  [3/3] OCR + Veri Cikartma...")
    ocr = get_ocr_engine()

    # Direkt crop uzerinde OCR yap
    preprocessed = ocr.preprocess_image(warped)
    raw_text = ocr.read_text(preprocessed)
    extracted = ocr.extract_data(raw_text)

    tarih = extracted.get('tarih', '-')
    toplam = extracted.get('toplam', '-')
    chars = len([c for c in raw_text if c.strip()])

    # Sonuc raporu
    print(f"\n  {'─'*40}")
    print(f"  SONUC:")
    print(f"  {'─'*40}")
    print(f"  Tarih  : {tarih if tarih else 'Bulunamadi'}")
    print(f"  Toplam : {toplam if toplam else 'Bulunamadi'} TL")
    print(f"  Karakter: {chars}")

    # Text dosyasina kaydet
    text_path = os.path.join(output_dir, f'{name}_text.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*50}\n")
        f.write(f"FIS BILGILERI\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Dosya    : {filename}\n")
        f.write(f"Islem    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tarih    : {tarih if tarih else 'Bulunamadi'}\n")
        f.write(f"Toplam   : {toplam if toplam else 'Bulunamadi'} TL\n")
        f.write(f"Karakter : {chars}\n")
        f.write(f"\n{'='*50}\n")
        f.write(f"HAM METIN\n")
        f.write(f"{'='*50}\n\n")
        f.write(raw_text)

    print(f"\n  Kaydedildi: {name}_crop.jpg, {name}_text.txt")

    # Sonuc objesi dondur
    return {
        'dosya': filename,
        'tarih': tarih,
        'toplam': toplam,
        'karakter': chars
    }


def main():
    """Ana fonksiyon."""
    import argparse

    parser = argparse.ArgumentParser(description='Fis OCR')
    parser.add_argument('input', nargs='?', default=None,
                        help='Resim dosyasi veya klasor (varsayilan: data/)')
    args = parser.parse_args()

    # Cikti klasoru
    output_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("FIS OCR - Final Pipeline")
    print("="*60)
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Dosyalari bul
    if args.input:
        input_path = args.input
        if not os.path.isabs(input_path):
            input_path = os.path.join(PROJECT_ROOT, input_path)

        if os.path.isfile(input_path):
            files = [input_path]
        else:
            print(f"HATA: Dosya bulunamadi: {input_path}")
            return
    else:
        # data/ klasorundeki tum fisleri bul
        data_dir = os.path.join(PROJECT_ROOT, 'data')
        patterns = ['fis*.jpg', 'fis*.jpeg', 'fis*.png',
                    'fis*.JPG', 'fis*.JPEG', 'fis*.PNG']

        files = []
        for pattern in patterns:
            files.extend(glob(os.path.join(data_dir, pattern)))
        files = sorted(set(files))

    if not files:
        print("HATA: Islenecek dosya bulunamadi!")
        return

    print(f"\n{len(files)} dosya islenecek")

    # Her dosyayi isle
    results = []
    skipped = 0

    for filepath in files:
        # Resume mantigi - zaten islenmis dosyalari atla
        filename = os.path.basename(filepath)
        name = os.path.splitext(filename)[0]
        text_path = os.path.join(output_dir, f'{name}_text.txt')

        if os.path.exists(text_path):
            print(f"\n[SKIP] {filename} - zaten islendi, atlaniyor")
            skipped += 1
            continue

        result = process_receipt(filepath, output_dir)
        if result:
            results.append(result)

        # RAM temizligi
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # FINAL RAPOR
    # =========================================================================
    print("\n")
    print("=" * 70)
    print("                         FINAL RAPOR")
    print("=" * 70)

    if results:
        print(f"\n{'Dosya':<20} {'Tarih':<15} {'Toplam':>12}")
        print("-" * 70)

        for r in results:
            dosya = r['dosya'][:18] + '..' if len(r['dosya']) > 20 else r['dosya']
            tarih = r['tarih'] if r['tarih'] else '-'
            toplam = f"{r['toplam']:.2f} TL" if r['toplam'] else '-'
            print(f"{dosya:<20} {tarih:<15} {toplam:>12}")

        print("-" * 70)

    # Istatistikler
    tarih_bulunan = len([r for r in results if r.get('tarih')])
    toplam_bulunan = len([r for r in results if r.get('toplam')])

    print(f"\nISTATISTIKLER:")
    print(f"  Toplam dosya     : {len(files)}")
    print(f"  Atlanan (resume) : {skipped}")
    print(f"  Islenen          : {len(results)}")
    print(f"  Tarih bulunan    : {tarih_bulunan}/{len(results)}")
    print(f"  Toplam bulunan   : {toplam_bulunan}/{len(results)}")
    print(f"\nCikti klasoru: {output_dir}/")
    print("=" * 70)

    # JSON'a da kaydet
    if results:
        json_path = os.path.join(output_dir, 'rapor.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'tarih': datetime.now().isoformat(),
                'toplam_dosya': len(files),
                'islenen': len(results),
                'atlanan': skipped,
                'sonuclar': results
            }, f, ensure_ascii=False, indent=2)
        print(f"JSON rapor: {json_path}")


if __name__ == '__main__':
    main()
