"""
fis1 icin Adaptive Thresholding Yontemi
Hedef: Golgeleri siyah, kagidi beyaz yaparak ayirmak
"""
import cv2
import numpy as np

IMAGE_PATH = "data/fis1.jpeg"


def isolate_receipt(image):
    """
    Fis kagidini izole et:
    Canny kenar tespiti + morfoloji ile fis sinirlarini bul
    """
    # Grayscale + blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny kenar tespiti
    edges = cv2.Canny(blurred, 50, 150)

    # Kenarlari kalinlastir ve birlestir
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Bosluklari doldur
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)

    return closed


def find_contours(thresh, min_area=5000):
    """
    Kontur Bulma ve Filtreleme:
    Alana gore sirala, kucuk gurultuleri at
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Alana gore filtrele ve sirala
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            filtered.append(cnt)

    # Buyukten kucuge sirala
    filtered = sorted(filtered, key=cv2.contourArea, reverse=True)

    print(f"Toplam kontur: {len(contours)}")
    print(f"Filtrelenmis (alan >= {min_area}): {len(filtered)}")

    # En buyuk 5 konturu goster
    for i, cnt in enumerate(filtered[:5]):
        area = cv2.contourArea(cnt)
        print(f"   {i+1}. alan={area:.0f}")

    return filtered


def find_quad(contours, padding=40):
    """
    4 Kose Yakalama - minAreaRect ile duzgun dikdortgen
    Padding: kenarlardan biraz fazla al
    """
    if len(contours) == 0:
        return None

    # minAreaRect kullan - duzgun dikdortgen verir
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)

    # Padding ekle
    if padding > 0:
        center = box.mean(axis=0)
        for i in range(len(box)):
            direction = box[i] - center
            norm = np.linalg.norm(direction)
            if norm > 0:
                box[i] = box[i] + (direction / norm) * padding

    quad = box.astype(np.int32).reshape(-1, 1, 2)

    w, h = rect[1]
    angle = rect[2]
    print(f"   -> minAreaRect: {w:.0f}x{h:.0f}, aci={angle:.1f}")
    print(f"   -> Padding eklendi: {padding}px")

    return quad


def order_points(pts):
    """
    4 noktayi sirala: Sol-Ust, Sag-Ust, Sag-Alt, Sol-Alt
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Sol-Ust
    rect[2] = pts[np.argmax(s)]  # Sag-Alt

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Sag-Ust
    rect[3] = pts[np.argmax(diff)]  # Sol-Alt

    return rect


def warp_perspective(image, quad):
    """
    Warp (Perspektif Duzeltme):
    4 noktayi kullanarak fisi duzlestir
    """
    pts = quad.reshape(4, 2).astype(np.float32)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Yeni boyutlar
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
        [0, maxHeight - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


if __name__ == "__main__":
    print(f"Goruntu: {IMAGE_PATH}")
    print("-" * 40)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise ValueError(f"Goruntu okunamadi: {IMAGE_PATH}")
    print(f"Boyut: {img.shape}")

    # 1. Fis Kagidini Izole Et
    thresh = isolate_receipt(img)
    print("1. Fis izolasyonu (Canny + morfoloji)")

    # 2. Kontur Bulma ve Filtreleme
    contours = find_contours(thresh, min_area=5000)
    print("2. Kontur Filtreleme (min_area=5000)")

    # 3. 4 Kose Ara
    quad = find_quad(contours)
    print("3. 4 Kose Arama")

    # Sonuc gorseli
    result = img.copy()
    if quad is not None:
        cv2.drawContours(result, [quad], -1, (0, 255, 0), 3)

        # 4. Warp
        warped = warp_perspective(img, quad)
        print(f"4. Warp: {warped.shape[1]}x{warped.shape[0]}")
        cv2.imwrite("experiments/fis1_camscanner_style.jpg", warped)
    else:
        print("4 koseli kontur bulunamadi!")
        cv2.drawContours(result, contours[:1], -1, (0, 0, 255), 2)

    # Kaydet
    cv2.imwrite("experiments/fis1_threshold_view.jpg", thresh)
    cv2.imwrite("experiments/fis1_result.png", result)
    print("\nKaydedildi: fis1_threshold_view.jpg, fis1_result.png, fis1_camscanner_style.jpg")
