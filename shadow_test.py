import cv2
import numpy as np
import os
from paddleocr import PaddleOCR

# 1. Resim Yolu (Senin klasör yapına göre 'data/o18.png' olabilir, kontrol et)
image_path = 'data/o20.jpg'

if not os.path.exists(image_path):
    print(f"HATA: {image_path} bulunamadi! Yolun dogru oldugundan emin ol.")
    exit()

print(f"Resim isleniyor: {image_path}...")

# 2. Resmi Oku
img = cv2.imread(image_path, -1)

# 3. Senin Bulduğun Gölge Kaldırma Kodu
rgb_planes = cv2.split(img)

result_planes = []
result_norm_planes = []

for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)

# 4. Temizlenmiş Resmi Kaydet
output_path = 'o20_clean.png'
cv2.imwrite(output_path, result_norm)
print(f"Gölgesiz resim kaydedildi: {output_path}")

# ==========================================
# TEST ZAMANI: PADDLE ESKİSİNİ VE YENİSİNİ OKUSUN
# ==========================================

ocr = PaddleOCR(use_angle_cls=True, lang='en') # Dil ayarını gerekirse tr yapabilirsin

print("\n--- 1. ORİJİNAL RESİM (GÖLGELİ) SONUCU ---")
result_original = ocr.ocr(image_path, cls=True)
for line in result_original[0]:
    print(f"{line[1][0]}  (Güven: {line[1][1]:.2f})")

print("\n" + "="*40 + "\n")

print("--- 2. TEMİZLENMİŞ RESİM (GÖLGESİZ) SONUCU ---")
result_clean = ocr.ocr(output_path, cls=True)
for line in result_clean[0]:
    print(f"{line[1][0]}  (Güven: {line[1][1]:.2f})")
