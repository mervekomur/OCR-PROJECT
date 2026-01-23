# -*- coding: utf-8 -*-
"""
fis1.jpeg icin Contour Detection + Perspective Warp
"""

import cv2
import numpy as np
import os

def order_points(pts):
    """Koseleri sirala: sol-ust, sag-ust, sag-alt, sol-alt"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Perspektif duzeltme"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

print("="*60)
print("FIS1 CROP - Contour Detection + Perspective Warp")
print("="*60)

# Dosya yollari
project_root = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(project_root, "data", "fis1.jpeg")
output_path = os.path.join(project_root, "fis1_crop.jpg")
debug_dir = os.path.join(project_root, "results", "debug_fis1")
os.makedirs(debug_dir, exist_ok=True)

# Resmi oku
image = cv2.imread(input_path)
if image is None:
    print(f"HATA: {input_path} bulunamadi!")
    exit(1)

orig = image.copy()
print(f"\nGirdi: {input_path}")
print(f"Boyut: {image.shape[1]}x{image.shape[0]}")

# Kucult (islem hizi icin)
ratio = image.shape[0] / 500.0
resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))

# Grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(debug_dir, "1_gray.jpg"), gray)

# Blur (gurultu azaltma)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imwrite(os.path.join(debug_dir, "2_blurred.jpg"), blurred)

# YONTEM 1: Canny edge detection
print("\n[1] Canny Edge deneniyor...")

best_contour = None
best_area = 0

for canny_low, canny_high in [(20, 50), (30, 100), (50, 150), (75, 200)]:
    edged = cv2.Canny(blurred, canny_low, canny_high)

    # Agresif morfolojik kapatma (kesik kenarlari bagla)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edged = cv2.dilate(edged, kernel, iterations=2)
    edged = cv2.erode(edged, kernel, iterations=1)

    # Konturlari bul
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > (500 * resized.shape[1] * 0.1) and area > best_area:
                best_contour = approx
                best_area = area
                print(f"  Canny({canny_low},{canny_high}): 4 kose bulundu, alan={area:.0f}")

# YONTEM 2: Adaptive Threshold (Canny basarisizsa)
if best_contour is None:
    print("\n[2] Adaptive Threshold deneniyor...")

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite(os.path.join(debug_dir, "3b_adaptive.jpg"), thresh)

    # Morfolojik islemler
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > (500 * resized.shape[1] * 0.1) and area > best_area:
                best_contour = approx
                best_area = area
                print(f"  Adaptive: 4 kose bulundu, alan={area:.0f}")

# YONTEM 3: Dusuk saturasyon (kagit/fis tespiti)
if best_contour is None:
    print("\n[3] Dusuk saturasyon (kagit) tespiti deneniyor...")

    # HSV'ye cevir
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Dusuk saturasyon = kagit/fis (renkli degil)
    # Yuksek value = parlak (karanlik degil)
    mask_low_sat = cv2.inRange(s, 0, 50)  # Dusuk saturasyon
    mask_bright = cv2.inRange(v, 150, 255)  # Parlak

    # Her ikisi de gecerli olan alanlar
    mask = cv2.bitwise_and(mask_low_sat, mask_bright)

    cv2.imwrite(os.path.join(debug_dir, "3c_paper_mask_raw.jpg"), mask)

    # Morfolojik temizlik
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Buyuk kernel ile bosluklari doldur
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    cv2.imwrite(os.path.join(debug_dir, "3c_paper_mask_clean.jpg"), mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    print(f"  {len(contours)} kagit benzeri bolge bulundu")

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        min_area = 500 * resized.shape[1] * 0.2  # %20 minimum
        print(f"    Bolge {i+1}: alan={area:.0f} (min={min_area:.0f})")

        if area > min_area:
            # minAreaRect ile dikdortgen fit et
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            width = rect[1][0]
            height = rect[1][1]
            aspect = max(width, height) / (min(width, height) + 1)
            print(f"    -> En-boy orani: {aspect:.2f}")

            # Fis genelde dikey - en-boy orani en az 1.8 olmali
            if aspect > 1.8:
                best_contour = box
                best_area = area
                print(f"  [OK] Kagit benzeri dikdortgen bulundu!")
                break
            else:
                print(f"    -> Atlandı (en-boy orani < 1.8, fis benzeri degil)")

# Son denenen edge'i kaydet
cv2.imwrite(os.path.join(debug_dir, "3_edges.jpg"), edged)

# YONTEM 4: Metin bolgeleri ile tespit (en son care)
if best_contour is None:
    print("\n[4] Metin bolgesi tespiti deneniyor...")

    # MSER ile metin bolgelerini bul
    mser = cv2.MSER_create()
    mser.setMinArea(50)
    mser.setMaxArea(5000)

    regions, _ = mser.detectRegions(gray)

    if len(regions) > 10:
        # Tum metin bolgelerinin noktalarini topla
        all_points = np.vstack(regions)

        # Convex hull ile dis sinirlari bul
        hull = cv2.convexHull(all_points)

        # minAreaRect ile dikdortgen fit et
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        area = cv2.contourArea(box)
        min_area = 500 * resized.shape[1] * 0.15

        # Debug: metin bolgelerini goster
        debug_mser = resized.copy()
        cv2.drawContours(debug_mser, [box], -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "4_mser_box.jpg"), debug_mser)

        print(f"  {len(regions)} metin bolgesi bulundu")
        print(f"  Toplam alan: {area:.0f}")

        if area > min_area:
            # Kutuyu biraz genislet (%10 margin)
            center = rect[0]
            size = (rect[1][0] * 1.1, rect[1][1] * 1.1)
            angle = rect[2]
            expanded_rect = (center, size, angle)
            box = cv2.boxPoints(expanded_rect)
            box = np.int32(box)

            best_contour = box
            best_area = area
            print(f"  [OK] Metin bolgesi etrafinda dikdortgen bulundu!")

if best_contour is not None:
    print(f"\n[OK] Fis kenari bulundu!")

    # Debug: konturu ciz
    debug_img = resized.copy()
    cv2.drawContours(debug_img, [best_contour], -1, (0, 255, 0), 3)
    cv2.imwrite(os.path.join(debug_dir, "4_contour.jpg"), debug_img)

    # Orijinal boyuta olcekle
    pts = best_contour.reshape(4, 2) * ratio

    # Perspective transform
    warped = four_point_transform(orig, pts)
    print(f"Crop boyutu: {warped.shape[1]}x{warped.shape[0]}")

    # Kaydet
    cv2.imwrite(output_path, warped)
    print(f"\n[KAYDEDILDI] {output_path}")

else:
    print("\n[!] 4 koseli kontur bulunamadi!")
    print("    Alternatif: Manuel kose secimi veya farkli yontem gerekebilir")

    # Debug icin tum konturlari ciz
    debug_img = resized.copy()
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for i, c in enumerate(contours):
        color = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (0,255,255)][i]
        cv2.drawContours(debug_img, [c], -1, color, 2)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print(f"    Kontur {i+1}: {len(approx)} kose, alan={cv2.contourArea(c):.0f}")

    cv2.imwrite(os.path.join(debug_dir, "4_all_contours.jpg"), debug_img)
    print(f"\n    Debug gorselleri: {debug_dir}")

print("\n" + "="*60)
