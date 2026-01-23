# -*- coding: utf-8 -*-
"""
Klasik Document Scanner - Canny Edge + Contour + Perspective Warp
"""

import cv2
import numpy as np

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

def scan_document(image_path, save_path=None):
    """
    Klasik document scanner.

    1. Grayscale + Blur
    2. Canny Edge
    3. Find Contours
    4. En buyuk 4 koseli sekli bul
    5. Perspective Warp
    """
    # 1. Resmi oku
    image = cv2.imread(image_path)
    if image is None:
        print(f"HATA: {image_path} bulunamadi!")
        return None

    orig = image.copy()
    ratio = image.shape[0] / 500.0
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    # 2. Grayscale + Blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny Edge Detection
    screenCnt = None
    img_area = image.shape[0] * image.shape[1]
    min_area = img_area * 0.10

    print(f"Resim alani: {img_area}, min alan: {min_area:.0f}")

    import os
    debug_dir = "results/debug_fis3"
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(f"{debug_dir}/0_gray.jpg", gray)

    for canny_low, canny_high in [(50, 150), (30, 100), (75, 200)]:
        edged = cv2.Canny(gray, canny_low, canny_high)

        cv2.imwrite(f"{debug_dir}/1_canny_{canny_low}_{canny_high}.jpg", edged)

        # Morfolojik islem - kesik kenarlari bagla
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edged = cv2.dilate(edged, kernel, iterations=2)
        edged = cv2.erode(edged, kernel, iterations=1)
        cv2.imwrite(f"{debug_dir}/2_morph_{canny_low}_{canny_high}.jpg", edged)

        # 4. Konturlari bul
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

        print(f"\nCanny({canny_low},{canny_high}): {len(cnts)} kontur")

        # Debug - konturlari ciz
        debug_img = image.copy()
        for i, c in enumerate(cnts[:5]):
            color = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (0,255,255)][i]
            cv2.drawContours(debug_img, [c], -1, color, 2)
            area = cv2.contourArea(c)
            print(f"  Kontur {i+1}: alan={area:.0f}")
        cv2.imwrite(f"{debug_dir}/3_contours_{canny_low}_{canny_high}.jpg", debug_img)

        # 5. Yeterince buyuk kontur bul
        for i, c in enumerate(cnts[:5]):
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)

            if area > min_area:
                # Farkli epsilon degerleri dene
                for eps in [0.02, 0.03, 0.04, 0.05, 0.08, 0.10]:
                    approx = cv2.approxPolyDP(c, eps * peri, True)
                    print(f"    eps={eps}: {len(approx)} kose")

                    if len(approx) == 4:
                        screenCnt = approx
                        print(f"  [{i+1}] 4 kose bulundu! alan={area:.0f}, eps={eps}")
                        break

                # 4 kose bulunamadiysa minAreaRect kullan
                if screenCnt is None and area > min_area * 1.5:
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    screenCnt = np.int32(box)
                    print(f"  [{i+1}] minAreaRect kullanildi! alan={area:.0f}")

            if screenCnt is not None:
                break

        if screenCnt is not None:
            break

    # 6. Sonuc
    if screenCnt is not None:
        print("[OK] Fis bulundu!")
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

        if save_path:
            cv2.imwrite(save_path, warped)
            print(f"[OK] Kaydedildi: {save_path}")

        return warped
    else:
        print("[!] 4 koseli sekil bulunamadi!")
        return None

# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    scan_document("data/fis3.jpg", "fis3_classic_scan.jpg")
