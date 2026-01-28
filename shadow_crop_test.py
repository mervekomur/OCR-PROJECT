# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from paddleocr import PaddleOCR

# 1. Resim Yolu
image_path = 'data/o20.jpg'

if not os.path.exists(image_path):
    print(f"HATA: {image_path} bulunamadi!")
    exit()

print(f"Resim isleniyor: {image_path}...")

# 2. Orijinal Resmi Oku
img_original = cv2.imread(image_path)
img = img_original.copy()

# ==========================================
# ADIM 1: GOLGE KALDIRMA
# ==========================================
print("\n[1] Golge kaldiriliyor...")
rgb_planes = cv2.split(img)

result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_norm_planes.append(norm_img)

img_no_shadow = cv2.merge(result_norm_planes)
cv2.imwrite('step1_no_shadow.png', img_no_shadow)
print("   -> step1_no_shadow.png kaydedildi")

# ==========================================
# ADIM 2: OTSU THRESHOLDING
# ==========================================
print("\n[2] Otsu thresholding uygulanıyor...")
gray = cv2.cvtColor(img_no_shadow, cv2.COLOR_BGR2GRAY)

# Gaussian blur (gurultu azaltma)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Otsu thresholding
_, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite('step2_otsu.png', otsu)
print("   -> step2_otsu.png kaydedildi")

# ==========================================
# ADIM 3: CANNY EDGE DETECTION
# ==========================================
print("\n[3] Canny edge detection uygulanıyor...")
edges = cv2.Canny(blurred, 50, 150)
cv2.imwrite('step3_canny.png', edges)
print("   -> step3_canny.png kaydedildi")

# ==========================================
# ADIM 4: KONTUR BULMA VE FIS CERCEVESI
# ==========================================
print("\n[4] Konturler bulunuyor...")

# Yontem 1: Otsu uzerinden beyaz alan bul (fis beyaz, arka plan renkli)
# Ters cevir - fis siyah olsun
otsu_inv = cv2.bitwise_not(otsu)

# Morfolojik islemler - delikleri kapat
kernel_close = np.ones((15, 15), np.uint8)
closed = cv2.morphologyEx(otsu_inv, cv2.MORPH_CLOSE, kernel_close)

# Kenarları genişlet
kernel = np.ones((10, 10), np.uint8)
dilated = cv2.dilate(closed, kernel, iterations=3)

# Konturları bul
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"   -> {len(contours)} kontur bulundu")

# En büyük konturu bul (alan bazlı)
best_contour = None
max_area = 0
img_area = img_original.shape[0] * img_original.shape[1]

for cnt in contours:
    area = cv2.contourArea(cnt)
    # En az resmin %10'u kadar büyük olmalı
    if area > img_area * 0.1 and area > max_area:
        max_area = area
        # Bounding rectangle kullan (daha güvenilir)
        x, y, w, h = cv2.boundingRect(cnt)
        # En-boy oranı kontrol (fiş dikey olmalı)
        aspect = h / w if w > 0 else 0
        if 0.5 < aspect < 5:  # Makul oran
            best_contour = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])

# Kontur çiz
img_with_contour = img_original.copy()
if best_contour is not None:
    cv2.drawContours(img_with_contour, [best_contour], -1, (0, 255, 0), 3)
    cv2.imwrite('step4_contour.png', img_with_contour)
    print("   -> step4_contour.png kaydedildi (yesil cerceve)")

# ==========================================
# ADIM 5: PERSPEKTIF DUZELTME VE KIRPMA
# ==========================================
print("\n[5] Fis kirpiliyor...")

if best_contour is not None and len(best_contour) >= 4:
    # Köşe noktalarını sırala
    pts = best_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # Sol-üst ve sağ-alt
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # sol-üst
    rect[2] = pts[np.argmax(s)]  # sağ-alt

    # Sağ-üst ve sol-alt
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # sağ-üst
    rect[3] = pts[np.argmax(diff)]  # sol-alt

    # Yeni boyutları hesapla
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Hedef noktalar
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Perspektif dönüşümü
    M = cv2.getPerspectiveTransform(rect, dst)
    cropped = cv2.warpPerspective(img_original, M, (maxWidth, maxHeight))

    cv2.imwrite('step5_cropped.png', cropped)
    print("   -> step5_cropped.png kaydedildi (kirpilmis fis)")

    # Kırpılmış fişe de gölge kaldırma uygula
    rgb_planes_crop = cv2.split(cropped)
    result_crop = []
    for plane in rgb_planes_crop:
        dilated = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_crop.append(norm)

    final_clean = cv2.merge(result_crop)
    cv2.imwrite('step6_final_clean.png', final_clean)
    print("   -> step6_final_clean.png kaydedildi (temizlenmis kirpilmis fis)")
else:
    print("   -> Uygun kontur bulunamadi, kirpma yapilamadi")
    # Alternatif: Basit bounding box
    gray_orig = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_orig, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img_original[y:y+h, x:x+w]
        cv2.imwrite('step5_cropped.png', cropped)
        print("   -> step5_cropped.png kaydedildi (basit kirpma)")

# ==========================================
# OCR KARSILASTIRMA
# ==========================================
print("\n" + "="*50)
print("OCR KARSILASTIRMA")
print("="*50)

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def count_results(result):
    if result and result[0]:
        return len(result[0])
    return 0

def get_key_values(result):
    """Tarih, Toplam, KDV degerlerini cikar"""
    values = {'tarih': None, 'toplam': None, 'kdv': None}
    if not result or not result[0]:
        return values

    lines = [line[1][0] for line in result[0]]
    for i, line in enumerate(lines):
        if 'TARIH' in line.upper() and i+1 < len(lines):
            values['tarih'] = lines[i+1]
        if 'TOPLAM' in line.upper() and i+1 < len(lines):
            values['toplam'] = lines[i+1]
        if 'TOPKDV' in line.upper() and i+1 < len(lines):
            values['kdv'] = lines[i+1]
    return values

print("\n--- 1. ORIJINAL RESIM ---")
result1 = ocr.ocr(image_path, cls=True)
vals1 = get_key_values(result1)
print(f"   Satir: {count_results(result1)}, Tarih: {vals1['tarih']}, Toplam: {vals1['toplam']}, KDV: {vals1['kdv']}")

print("\n--- 2. GOLGE KALDIRILMIS ---")
result2 = ocr.ocr('step1_no_shadow.png', cls=True)
vals2 = get_key_values(result2)
print(f"   Satir: {count_results(result2)}, Tarih: {vals2['tarih']}, Toplam: {vals2['toplam']}, KDV: {vals2['kdv']}")

if os.path.exists('step5_cropped.png'):
    print("\n--- 3. KIRPILMIS FIS ---")
    result3 = ocr.ocr('step5_cropped.png', cls=True)
    vals3 = get_key_values(result3)
    print(f"   Satir: {count_results(result3)}, Tarih: {vals3['tarih']}, Toplam: {vals3['toplam']}, KDV: {vals3['kdv']}")

if os.path.exists('step6_final_clean.png'):
    print("\n--- 4. FINAL (KIRPILMIS + TEMIZ) ---")
    result4 = ocr.ocr('step6_final_clean.png', cls=True)
    vals4 = get_key_values(result4)
    print(f"   Satir: {count_results(result4)}, Tarih: {vals4['tarih']}, Toplam: {vals4['toplam']}, KDV: {vals4['kdv']}")

print("\n" + "="*50)
print("TAMAMLANDI! Cikan dosyalar:")
print("  - step1_no_shadow.png (golge kaldirilmis)")
print("  - step2_otsu.png (otsu threshold)")
print("  - step3_canny.png (kenar tespiti)")
print("  - step4_contour.png (cerceve)")
print("  - step5_cropped.png (kirpilmis)")
print("  - step6_final_clean.png (final)")
print("="*50)
