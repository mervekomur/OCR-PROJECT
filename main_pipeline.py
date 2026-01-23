# -*- coding: utf-8 -*-
"""
=============================================================================
ANA PIPELINE - Fiş OCR Sistemi
=============================================================================

Tek tuşla: Fotoğraf → Metin

Pipeline Sırası:
    1. SAM Segmentasyon (Fişi bul)
    2. Corner Cut / Warp (Perspektif düzelt)
    3. Scan Effect (Görüntü iyileştirme)
    4. EasyOCR (Metin çıkar)

Kullanım:
    python main_pipeline.py
    python main_pipeline.py data/fis2.jpeg

Çıktılar:
    - final_output.txt (OCR sonucu)
    - pipeline_steps.jpg (Görsel süreç)
"""

import os
import sys
import cv2
import numpy as np
import argparse
from datetime import datetime

# =============================================================================
# ADIM 1: SAM SEGMENTASYON
# =============================================================================

def load_sam_model(model_path=None):
    """SAM modelini yükle."""
    try:
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print("HATA: segment_anything yuklu degil!")
        print("Kurulum: pip install segment-anything")
        return None, None

    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'sam_vit_b.pth')

    if not os.path.exists(model_path):
        print(f"HATA: SAM model bulunamadi: {model_path}")
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  SAM yukleniyor... (device: {device})")

    sam = sam_model_registry["vit_b"](checkpoint=model_path)
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=1000,
    )

    return sam, mask_generator


def find_receipt_mask(masks, image_shape):
    """En uygun fiş maskesini bul."""
    h, w = image_shape[:2]
    img_area = h * w
    center = (w // 2, h // 2)

    best_mask = None
    best_score = 0

    for mask_data in masks:
        area = mask_data['area']
        bbox = mask_data['bbox']

        # Alan filtresi: %5 ile %90 arası
        area_ratio = area / img_area
        if area_ratio < 0.05 or area_ratio > 0.90:
            continue

        # Merkeze yakınlık
        mask_center_x = bbox[0] + bbox[2] // 2
        mask_center_y = bbox[1] + bbox[3] // 2
        dist_to_center = np.sqrt((mask_center_x - center[0])**2 + (mask_center_y - center[1])**2)
        max_dist = np.sqrt(center[0]**2 + center[1]**2)
        center_score = 1 - (dist_to_center / max_dist)

        # Stability score
        stability = mask_data.get('stability_score', 0.5)

        # Toplam skor
        score = area_ratio * 0.4 + center_score * 0.4 + stability * 0.2

        if score > best_score:
            best_score = score
            best_mask = mask_data

    return best_mask


def mask_to_corners(mask):
    """Maskeden 4 köşe noktası çıkar."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)

    # Approximate polygon
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        return approx.reshape(4, 2)

    # 4 köşe bulunamazsa minAreaRect kullan
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int32(box)


def segment_receipt(image, mask_generator):
    """SAM ile fişi segmente et ve köşeleri bul."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("  Maskeler uretiliyor...")
    masks = mask_generator.generate(image_rgb)
    print(f"  {len(masks)} maske bulundu")

    best_mask = find_receipt_mask(masks, image.shape)

    if best_mask is None:
        print("  UYARI: Uygun maske bulunamadi!")
        return None, None

    mask = best_mask['segmentation']
    corners = mask_to_corners(mask)

    print(f"  Fis maskesi bulundu (alan: {best_mask['area']})")

    return mask, corners


# =============================================================================
# ADIM 2: WARP PERSPECTIVE (CORNER CUT)
# =============================================================================

def order_points(pts):
    """Köşe noktalarını sırala: [sol-üst, sağ-üst, sağ-alt, sol-alt]"""
    pts = pts.astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    # Toplam ve fark hesapla
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    rect[0] = pts[np.argmin(s)]   # sol-üst: x+y en küçük
    rect[2] = pts[np.argmax(s)]   # sağ-alt: x+y en büyük
    rect[1] = pts[np.argmin(diff)] # sağ-üst: x-y en küçük
    rect[3] = pts[np.argmax(diff)] # sol-alt: x-y en büyük

    return rect


def warp_perspective(image, corners):
    """Perspektif düzeltme - fişi düz dikdörtgen yap."""

    # minAreaRect kullan - daha güvenilir
    rect = cv2.minAreaRect(corners.astype(np.float32))
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # Rect bilgileri: ((cx, cy), (width, height), angle)
    (cx, cy), (w, h), angle = rect

    # Fiş dikey olmalı - width < height olacak şekilde ayarla
    if w > h:
        w, h = h, w
        angle += 90

    # Kaynak noktaları sırala
    src_pts = order_points(box.astype("float32"))

    # Hedef boyutları (dikey fiş için height > width)
    dst_width = int(w)
    dst_height = int(h)

    # Minimum boyut kontrolü
    dst_width = max(dst_width, 100)
    dst_height = max(dst_height, 100)

    # Hedef noktalar
    dst_pts = np.array([
        [0, 0],
        [dst_width - 1, 0],
        [dst_width - 1, dst_height - 1],
        [0, dst_height - 1]
    ], dtype="float32")

    # Perspektif dönüşümü
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (dst_width, dst_height))

    # Son kontrol: hala yataysa döndür
    if warped.shape[1] > warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    print(f"  Warp: {corners.shape} corners -> {warped.shape[:2]} output")

    return warped


# =============================================================================
# ADIM 3: SCAN EFFECT
# =============================================================================

def to_grayscale(image):
    """BGR -> Grayscale (HAM)"""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_shadows(gray_image, blur_kernel=51):
    """Division Method ile gölge giderme."""
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    background = cv2.GaussianBlur(gray_image, (blur_kernel, blur_kernel), 0)
    gray_float = gray_image.astype(np.float32)
    background_float = np.maximum(background.astype(np.float32), 1.0)
    divided = (gray_float / background_float) * 255.0
    result = cv2.normalize(divided, None, 0, 255, cv2.NORM_MINMAX)

    return result.astype(np.uint8)


def adaptive_threshold(image, block_size=21, C=12):
    """Adaptive Threshold."""
    if block_size % 2 == 0:
        block_size += 1

    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )


def morphological_closing(binary_image, kernel_size=2):
    """Morphological Closing - kopuklukları kapat."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    inverted = cv2.bitwise_not(binary_image)
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
    return cv2.bitwise_not(closed)


def scan_effect(image):
    """
    Scan Effect v4: Shadow Removal + Adaptive Threshold + Morph Closing

    Returns:
        tuple: (processed_image, grayscale_image)
    """
    # Grayscale (HAM hali)
    gray = to_grayscale(image)

    # Shadow removal
    no_shadow = remove_shadows(gray, blur_kernel=51)

    # Adaptive threshold
    binary = adaptive_threshold(no_shadow, block_size=21, C=12)

    # Morphological closing
    closed = morphological_closing(binary, kernel_size=2)

    return closed, gray


# =============================================================================
# ADIM 4: EASYOCR
# =============================================================================

def run_easyocr(image, languages=['tr', 'en']):
    """EasyOCR ile metin çıkar."""
    try:
        import easyocr
    except ImportError:
        print("HATA: easyocr yuklu degil!")
        print("Kurulum: pip install easyocr")
        return "[EasyOCR yuklu degil]"

    print("  EasyOCR baslatiliyor...")
    reader = easyocr.Reader(languages, gpu=True, verbose=False)

    print("  Metin okunuyor...")
    results = reader.readtext(image, detail=0)

    text = '\n'.join(results)
    return text


# =============================================================================
# GÖRSEL ÇIKTI
# =============================================================================

def create_pipeline_visual(original, mask, warped, scanned, output_path):
    """Pipeline adımlarını yan yana göster."""

    def resize_to_height(img, target_h):
        if img is None:
            return np.zeros((target_h, 200, 3), dtype=np.uint8)
        scale = target_h / img.shape[0]
        new_w = max(1, int(img.shape[1] * scale))
        resized = cv2.resize(img, (new_w, target_h))
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        return resized

    target_h = 600

    # Orijinal
    img1 = resize_to_height(original, target_h)

    # Maske overlay
    if mask is not None:
        mask_overlay = original.copy()
        mask_overlay[mask] = [0, 255, 0]
        mask_overlay = cv2.addWeighted(original, 0.7, mask_overlay, 0.3, 0)
        img2 = resize_to_height(mask_overlay, target_h)
    else:
        img2 = resize_to_height(None, target_h)

    # Warped
    img3 = resize_to_height(warped, target_h)

    # Scanned
    img4 = resize_to_height(scanned, target_h)

    # Etiketler
    labels = ['1. Orijinal', '2. SAM Maske', '3. Warped', '4. Scan Effect']
    font = cv2.FONT_HERSHEY_SIMPLEX

    for img, label in zip([img1, img2, img3, img4], labels):
        cv2.putText(img, label, (10, 30), font, 0.7, (0, 0, 255), 2)

    # Birleştir
    combined = np.hstack([img1, img2, img3, img4])
    cv2.imwrite(output_path, combined)

    return combined


# =============================================================================
# KARŞILAŞTIRMA RAPORU
# =============================================================================

def extract_key_fields(text):
    """Metinden önemli alanları çıkar."""
    import re

    fields = {
        'toplam': '-',
        'tarih': '-',
        'isyeri': '-',
        'onay_kodu': '-',
        'saat': '-'
    }

    lines = text.upper().split('\n')
    text_upper = text.upper()

    # İşyeri adı (genelde ilk satır)
    for line in lines[:5]:
        line = line.strip()
        if len(line) > 3 and 'FIRIN' in line or 'MARKET' in line or 'CAFE' in line:
            fields['isyeri'] = line
            break
        elif len(line) > 5 and line[0].isalpha():
            fields['isyeri'] = line
            break

    # Toplam tutar
    toplam_patterns = [
        r'TOPLAM[:\s]*\*?(\d+[.,]\d{2})',
        r'(\d{2,3}[.,]\d{2})\s*TL',
        r'\*(\d+[.,]\d{2})'
    ]
    for pattern in toplam_patterns:
        match = re.search(pattern, text_upper)
        if match:
            fields['toplam'] = match.group(1) + ' TL'
            break

    # Tarih
    tarih_match = re.search(r'(\d{2}[./]\d{2}[./]\d{4})', text)
    if tarih_match:
        fields['tarih'] = tarih_match.group(1)

    # Saat
    saat_match = re.search(r'(\d{2}[.:]\d{2}[.:]\d{2})', text)
    if saat_match:
        fields['saat'] = saat_match.group(1)

    # Onay kodu
    onay_match = re.search(r'ONAY[^:]*[:\s]*(\d{6})', text_upper)
    if onay_match:
        fields['onay_kodu'] = onay_match.group(1)

    return fields


def print_comparison_report(text_stage3, text_stage4):
    """Stage 3 vs Stage 4 karşılaştırma raporu."""

    fields3 = extract_key_fields(text_stage3)
    fields4 = extract_key_fields(text_stage4)

    chars3 = len([c for c in text_stage3 if c.strip()])
    chars4 = len([c for c in text_stage4 if c.strip()])
    words3 = len(text_stage3.split())
    words4 = len(text_stage4.split())

    print("\n" + "=" * 70)
    print("*** STAGE 3 vs STAGE 4 - OCR KAPISMASI ***")
    print("=" * 70)

    print("\n" + "-" * 70)
    print(f"{'ALAN':<15} | {'STAGE 3 (Warped)':<25} | {'STAGE 4 (Scan)':<25}")
    print("-" * 70)
    print(f"{'Isyeri':<15} | {fields3['isyeri'][:25]:<25} | {fields4['isyeri'][:25]:<25}")
    print(f"{'Tarih':<15} | {fields3['tarih']:<25} | {fields4['tarih']:<25}")
    print(f"{'Saat':<15} | {fields3['saat']:<25} | {fields4['saat']:<25}")
    print(f"{'Toplam':<15} | {fields3['toplam']:<25} | {fields4['toplam']:<25}")
    print(f"{'Onay Kodu':<15} | {fields3['onay_kodu']:<25} | {fields4['onay_kodu']:<25}")
    print("-" * 70)
    print(f"{'Karakter':<15} | {chars3:<25} | {chars4:<25}")
    print(f"{'Kelime':<15} | {words3:<25} | {words4:<25}")
    print("-" * 70)

    # Kazanan
    score3 = sum([1 for v in fields3.values() if v != '-']) + (chars3 // 100)
    score4 = sum([1 for v in fields4.values() if v != '-']) + (chars4 // 100)

    print("\n" + "=" * 70)
    if score3 > score4:
        print(">>> KAZANAN: STAGE 3 (Warped/Grayscale) <<<")
        print("    Dogal gri tonlari makine icin daha okunakli!")
    elif score4 > score3:
        print(">>> KAZANAN: STAGE 4 (Scan Effect) <<<")
        print("    Temizlenmis goruntu daha iyi sonuc verdi!")
    else:
        print(">>> BERABERE! Her iki yontem de benzer sonuc verdi. <<<")
    print("=" * 70)

    return fields3, fields4


# =============================================================================
# ANA PIPELINE
# =============================================================================

def run_pipeline(input_path, output_dir=None):
    """
    Ana pipeline - tüm adımları sırayla çalıştır.
    Stage 3 vs Stage 4 karşılaştırması yapar.

    Args:
        input_path: Girdi görüntü yolu
        output_dir: Çıktı klasörü (None ise proje kökü/results)

    Returns:
        tuple: (text_stage3, text_stage4)
    """
    print("\n" + "=" * 70)
    print("FIS OCR PIPELINE - Cift Motorlu")
    print("=" * 70)
    print(f"Girdi: {input_path}")
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Çıktı klasörü - results/
    project_root = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(project_root, 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Görüntüyü yükle
    print("\n[0/5] Goruntu yukleniyor...")
    image = cv2.imread(input_path)
    if image is None:
        print(f"HATA: Goruntu yuklenemedi: {input_path}")
        return None, None

    print(f"  Boyut: {image.shape}")

    # ADIM 1: SAM Segmentasyon
    print("\n[1/5] SAM SEGMENTASYON")
    print("-" * 40)

    sam, mask_generator = load_sam_model()
    if mask_generator is None:
        print("  SAM yuklenemedi, warp atlaniyor...")
        mask = None
        corners = None
        warped = image.copy()
    else:
        mask, corners = segment_receipt(image, mask_generator)

        if corners is None:
            print("  Koseler bulunamadi, warp atlaniyor...")
            warped = image.copy()
        else:
            # ADIM 2: Warp
            print("\n[2/5] WARP PERSPECTIVE")
            print("-" * 40)
            warped = warp_perspective(image, corners)
            print(f"  Warped boyut: {warped.shape}")

    # ADIM 3: Stage 3 - Grayscale (Warped)
    print("\n[3/5] STAGE 3 - GRAYSCALE")
    print("-" * 40)
    grayscale = to_grayscale(warped)
    print("  Grayscale donusumu tamamlandi")

    # ADIM 4: Stage 4 - Scan Effect
    print("\n[4/5] STAGE 4 - SCAN EFFECT")
    print("-" * 40)
    scanned, _ = scan_effect(warped)
    print("  Shadow removal + Adaptive threshold + Closing uygulandi")

    # ADIM 5: OCR - Her iki stage için
    print("\n[5/5] EASYOCR - CIFT MOTOR")
    print("-" * 40)

    # EasyOCR reader başlat (bir kere)
    try:
        import easyocr
        print("  EasyOCR baslatiliyor...")
        reader = easyocr.Reader(['tr', 'en'], gpu=True, verbose=False)
    except ImportError:
        print("HATA: easyocr yuklu degil!")
        return None, None

    # Stage 3 OCR
    print("\n  Stage 3 (Warped) okunuyor...")
    results3 = reader.readtext(grayscale, detail=0)
    text_stage3 = '\n'.join(results3)
    chars3 = len([c for c in text_stage3 if c.strip()])
    print(f"  -> {chars3} karakter okundu")

    # Stage 4 OCR
    print("\n  Stage 4 (Scan Effect) okunuyor...")
    results4 = reader.readtext(scanned, detail=0)
    text_stage4 = '\n'.join(results4)
    chars4 = len([c for c in text_stage4 if c.strip()])
    print(f"  -> {chars4} karakter okundu")

    # Karşılaştırma raporu
    print_comparison_report(text_stage3, text_stage4)

    # Sonuçları kaydet
    print("\n" + "=" * 70)
    print("CIKTILAR (results/ klasoru)")
    print("=" * 70)

    # Stage 3 OCR sonucu
    txt3_path = os.path.join(output_dir, 'text_stage3_warped.txt')
    with open(txt3_path, 'w', encoding='utf-8') as f:
        f.write(f"STAGE 3 - WARPED (Grayscale)\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Karakter: {chars3}\n")
        f.write("=" * 50 + "\n\n")
        f.write(text_stage3)
    print(f"  Stage 3 OCR: {txt3_path}")

    # Stage 4 OCR sonucu
    txt4_path = os.path.join(output_dir, 'text_stage4_scanned.txt')
    with open(txt4_path, 'w', encoding='utf-8') as f:
        f.write(f"STAGE 4 - SCAN EFFECT\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Karakter: {chars4}\n")
        f.write("=" * 50 + "\n\n")
        f.write(text_stage4)
    print(f"  Stage 4 OCR: {txt4_path}")

    # Görsel pipeline
    visual_path = os.path.join(output_dir, 'pipeline_steps.jpg')
    create_pipeline_visual(image, mask, warped, scanned, visual_path)
    print(f"  Gorsel:      {visual_path}")

    # Ara çıktılar
    cv2.imwrite(os.path.join(output_dir, 'img_stage1_original.jpg'), image)
    cv2.imwrite(os.path.join(output_dir, 'img_stage3_warped.jpg'), grayscale)
    cv2.imwrite(os.path.join(output_dir, 'img_stage4_scanned.jpg'), scanned)

    # Mask overlay
    if mask is not None:
        mask_overlay = image.copy()
        mask_overlay[mask] = [0, 255, 0]
        mask_overlay = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)
        cv2.imwrite(os.path.join(output_dir, 'img_stage2_mask.jpg'), mask_overlay)

    print("\n" + "=" * 70)
    print("PIPELINE TAMAMLANDI!")
    print("=" * 70)

    return text_stage3, text_stage4


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fis OCR Pipeline')
    parser.add_argument('input', nargs='?', default='data/fis1.jpeg',
                        help='Girdi goruntu yolu (varsayilan: data/fis1.jpeg)')
    parser.add_argument('-o', '--output', default=None,
                        help='Cikti klasoru (varsayilan: proje koku)')

    args = parser.parse_args()

    # Tam yol
    if not os.path.isabs(args.input):
        args.input = os.path.join(os.path.dirname(__file__), args.input)

    if not os.path.exists(args.input):
        print(f"HATA: Dosya bulunamadi: {args.input}")
        sys.exit(1)

    run_pipeline(args.input, args.output)


if __name__ == '__main__':
    main()
