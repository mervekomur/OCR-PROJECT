# -*- coding: utf-8 -*-
"""
fis1.jpeg - SADECE CERCEVE tespiti
Agir blur ile ic detaylari (metin) yok et, sadece dis kenarlari bul
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
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

print("="*60)
print("FIS1 - SADECE CERCEVE TESPITI")
print("="*60)

project_root = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(project_root, "data", "fis1.jpeg")
output_path = os.path.join(project_root, "fis1_crop.jpg")
debug_dir = os.path.join(project_root, "results", "debug_frame")
os.makedirs(debug_dir, exist_ok=True)

image = cv2.imread(input_path)
orig = image.copy()
h, w = image.shape[:2]
print(f"Boyut: {w}x{h}")

# Kucult
ratio = h / 500.0
resized = cv2.resize(image, (int(w / ratio), 500))
cv2.imwrite(os.path.join(debug_dir, "0_resized.jpg"), resized)

# Grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# AGIR BLUR - metin ve ic detaylari yok et, sadece genel sekil kalsin
print("\n[1] Agir blur uygulaniyor...")
blurred = cv2.GaussianBlur(gray, (15, 15), 0)
blurred = cv2.GaussianBlur(blurred, (15, 15), 0)
blurred = cv2.GaussianBlur(blurred, (11, 11), 0)
cv2.imwrite(os.path.join(debug_dir, "1_heavy_blur.jpg"), blurred)

# Sobel gradients - dikey ve yatay kenarlari ayri bul
print("[2] Sobel gradient hesaplaniyor...")
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # Dikey kenarlar
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # Yatay kenarlar

sobelx = np.uint8(np.absolute(sobelx))
sobely = np.uint8(np.absolute(sobely))

cv2.imwrite(os.path.join(debug_dir, "2a_sobel_vertical.jpg"), sobelx)
cv2.imwrite(os.path.join(debug_dir, "2b_sobel_horizontal.jpg"), sobely)

# Birlestir
edges = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv2.imwrite(os.path.join(debug_dir, "2c_sobel_combined.jpg"), edges)

# Threshold
_, edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(debug_dir, "2d_sobel_thresh.jpg"), edges)

# Morfoloji - kenarlari bagla
kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))  # Dikey baglanti
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))  # Yatay baglanti

edges_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v, iterations=2)
edges_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h, iterations=2)
edges = cv2.bitwise_or(edges_v, edges_h)

cv2.imwrite(os.path.join(debug_dir, "3_morph_connected.jpg"), edges)

# Kontur bul
print("[3] Kontur arama...")
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

print(f"  {len(contours)} kontur bulundu")

# Debug - ilk 5 konturu ciz
debug_all = resized.copy()
colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (0,255,255)]
for i, c in enumerate(contours[:5]):
    cv2.drawContours(debug_all, [c], -1, colors[i % 5], 2)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    print(f"  Kontur {i+1}: {len(approx)} kose, alan={area:.0f}")
cv2.imwrite(os.path.join(debug_dir, "4_all_contours.jpg"), debug_all)

# 4 koseli kontur ara
best_contour = None
min_area = resized.shape[0] * resized.shape[1] * 0.2  # min %20

for c in contours:
    peri = cv2.arcLength(c, True)

    for eps in [0.02, 0.03, 0.04, 0.05, 0.06]:
        approx = cv2.approxPolyDP(c, eps * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > min_area:
                rect = cv2.minAreaRect(approx)
                w_rect, h_rect = rect[1]
                aspect = max(w_rect, h_rect) / (min(w_rect, h_rect) + 1)

                print(f"\n  [ADAY] 4 kose, alan={area:.0f}, en-boy={aspect:.2f}")

                if aspect > 1.3:
                    best_contour = approx
                    print(f"  [SECILDI]")
                    break

    if best_contour is not None:
        break

# Kontur bulunamadiysa -> Hough Lines ile dene
if best_contour is None:
    print("\n[4] Hough Lines ile cerceve olusturma...")

    # Dikey cizgileri bul
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                            minLineLength=80, maxLineGap=30)

    if lines is not None:
        print(f"  {len(lines)} cizgi bulundu")

        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Dikey cizgiler (+-20 derece)
            if abs(abs(angle) - 90) < 20:
                vertical_lines.append((x1, y1, x2, y2))
            # Yatay cizgiler
            elif abs(angle) < 20 or abs(angle) > 160:
                horizontal_lines.append((x1, y1, x2, y2))

        print(f"  Dikey: {len(vertical_lines)}, Yatay: {len(horizontal_lines)}")

        # Debug - cizgileri ciz
        debug_lines = resized.copy()
        for x1, y1, x2, y2 in vertical_lines:
            cv2.line(debug_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for x1, y1, x2, y2 in horizontal_lines:
            cv2.line(debug_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, "5_hough_lines.jpg"), debug_lines)

        if len(vertical_lines) >= 1:
            # En soldaki ve en sagdaki dikey cizgiyi bul
            v_x_coords = [(min(l[0], l[2]), max(l[0], l[2])) for l in vertical_lines]

            left_x = min([x[0] for x in v_x_coords])
            right_x = max([x[1] for x in v_x_coords])

            # Fis genisligi kontrol - cok dar veya genis olmamali
            width = right_x - left_x
            if width < resized.shape[1] * 0.2:
                # Cok dar - fis genisligi tahmin et
                right_x = left_x + int(resized.shape[1] * 0.45)
                print(f"  Sag kenar tahmin edildi: {right_x}")

            # Yatay sinirlar - resmin ust ve alt %10'u haric
            top_y = int(resized.shape[0] * 0.02)
            bottom_y = int(resized.shape[0] * 0.95)

            # 4 kose olustur
            best_contour = np.array([
                [left_x, top_y],
                [right_x, top_y],
                [right_x, bottom_y],
                [left_x, bottom_y]
            ], dtype=np.int32)

            print(f"  Cerceve: ({left_x},{top_y}) -> ({right_x},{bottom_y})")

            # Debug
            debug_box = resized.copy()
            cv2.drawContours(debug_box, [best_contour], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_dir, "6_hough_box.jpg"), debug_box)

# Sonuc
print("\n" + "="*60)
if best_contour is not None:
    # Debug - secilen konturu ciz
    debug_selected = resized.copy()
    cv2.drawContours(debug_selected, [best_contour], -1, (0, 255, 0), 3)
    cv2.imwrite(os.path.join(debug_dir, "5_selected.jpg"), debug_selected)

    # Perspective transform
    pts = best_contour.reshape(4, 2) * ratio
    warped = four_point_transform(orig, pts)
    cv2.imwrite(output_path, warped)

    print(f"[BASARILI] Crop: {warped.shape[1]}x{warped.shape[0]}")
    print(f"Kaydedildi: {output_path}")
else:
    print("[BASARISIZ] 4 koseli cerceve bulunamadi")
    print(f"Debug: {debug_dir}")

print("="*60)
