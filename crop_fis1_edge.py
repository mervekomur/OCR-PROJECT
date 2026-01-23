# -*- coding: utf-8 -*-
"""
fis1.jpeg - Sadece Canny Edge Detection ile kesim
Kenarları birleştirmek için agresif morfoloji
"""

import cv2
import numpy as np
import os

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

print("="*60)
print("FIS1 - CANNY EDGE DETECTION")
print("="*60)

project_root = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(project_root, "data", "fis1.jpeg")
output_path = os.path.join(project_root, "fis1_crop.jpg")
debug_dir = os.path.join(project_root, "results", "debug_fis1_edge")
os.makedirs(debug_dir, exist_ok=True)

image = cv2.imread(input_path)
orig = image.copy()
print(f"Boyut: {image.shape[1]}x{image.shape[0]}")

# Kucult
ratio = image.shape[0] / 500.0
resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))

# Grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# CLAHE - kontrast artir (kenarlar daha belirgin olsun)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)
cv2.imwrite(os.path.join(debug_dir, "1_clahe.jpg"), gray)

# Bilateral Filter - dokuyu yumussat, kenarlari koru
blurred = cv2.bilateralFilter(gray, 9, 75, 75)
cv2.imwrite(os.path.join(debug_dir, "2_bilateral.jpg"), blurred)

# Ek Gaussian blur
blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
cv2.imwrite(os.path.join(debug_dir, "2b_blurred.jpg"), blurred)

best_contour = None
best_area = 0

print("\nCanny Edge Detection denenecek parametreler:")
params = [
    (10, 50),
    (20, 60),
    (30, 80),
    (40, 100),
    (50, 150),
]

for i, (low, high) in enumerate(params):
    print(f"\n[{i+1}] Canny({low}, {high})")

    # Canny
    edges = cv2.Canny(blurred, low, high)
    cv2.imwrite(os.path.join(debug_dir, f"3a_canny_{low}_{high}.jpg"), edges)

    # Morfolojik islemler - HAFIF (sadece kucuk bosluklari kapat)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 1. Hafif dilate
    edges = cv2.dilate(edges, kernel_small, iterations=1)

    # 2. Close - sadece kucuk bosluklari kapat
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_medium, iterations=1)

    cv2.imwrite(os.path.join(debug_dir, f"3b_morph_{low}_{high}.jpg"), edges)

    # Kontur bul
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    print(f"  {len(contours)} kontur bulundu")

    for j, c in enumerate(contours):
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)

        # Farkli epsilon degerleri dene
        for eps_mult in [0.01, 0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(c, eps_mult * peri, True)

            if len(approx) == 4:
                # Minimum alan kontrolu (%15)
                min_area = 500 * resized.shape[1] * 0.15
                if area > min_area and area > best_area:
                    best_contour = approx
                    best_area = area
                    print(f"  [OK] 4 kose bulundu! eps={eps_mult}, alan={area:.0f}")

                    # Debug - konturu ciz
                    debug_img = resized.copy()
                    cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 3)
                    cv2.imwrite(os.path.join(debug_dir, f"4_found_{low}_{high}.jpg"), debug_img)
                    break

        if best_contour is not None and area > best_area * 0.9:
            break

# Hough Lines ile alternatif (eger hala bulunamadiysa)
if best_contour is None:
    print("\n[ALTERNATIF] Hough Lines deneniyor...")

    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)

    if lines is not None:
        print(f"  {len(lines)} cizgi bulundu")

        # Tum cizgilerin noktalarini topla
        all_points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            all_points.append([x1, y1])
            all_points.append([x2, y2])

        if len(all_points) > 4:
            all_points = np.array(all_points)
            hull = cv2.convexHull(all_points)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            area = cv2.contourArea(box)
            min_area = 500 * resized.shape[1] * 0.15

            if area > min_area:
                best_contour = box
                best_area = area
                print(f"  [OK] Hough ile dikdortgen bulundu! alan={area:.0f}")

                debug_img = resized.copy()
                cv2.drawContours(debug_img, [box], -1, (0, 255, 0), 3)
                cv2.imwrite(os.path.join(debug_dir, "5_hough_box.jpg"), debug_img)

# Sonuc
print("\n" + "="*60)
if best_contour is not None:
    pts = best_contour.reshape(4, 2) * ratio
    warped = four_point_transform(orig, pts)
    cv2.imwrite(output_path, warped)
    print(f"[BASARILI] Crop boyutu: {warped.shape[1]}x{warped.shape[0]}")
    print(f"Kaydedildi: {output_path}")
else:
    print("[BASARISIZ] 4 koseli kontur bulunamadi")
    print(f"Debug gorselleri: {debug_dir}")

print("="*60)
