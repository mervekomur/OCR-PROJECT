# -*- coding: utf-8 -*-
"""
Filter Test - 4 farkli filtre karsilastirmasi
fis5.jpg uzerinde test
"""

import cv2
import numpy as np
import os

# Dosya yollari
input_path = "data/fis1.jpeg"
output_dir = "results/filter_test_fis1"
os.makedirs(output_dir, exist_ok=True)

# Resmi oku
image = cv2.imread(input_path)
if image is None:
    print(f"HATA: {input_path} bulunamadi!")
    exit(1)

print(f"Girdi: {input_path}")
print(f"Boyut: {image.shape[1]}x{image.shape[0]}")
print(f"Cikti klasoru: {output_dir}\n")

# =============================================================================
# FILTER 1: Grayscale (Gri tonlama)
# =============================================================================
print("[1/4] Grayscale...")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f"{output_dir}/filter1_gray.jpg", gray)
print("      -> filter1_gray.jpg")

# =============================================================================
# FILTER 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
# =============================================================================
print("[2/5] CLAHE (clipLimit=2.0, tileGridSize=8x8)...")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_enhanced = clahe.apply(gray)
cv2.imwrite(f"{output_dir}/filter2_clahe.jpg", clahe_enhanced)
print("      -> filter2_clahe.jpg")

# =============================================================================
# FILTER 3: Simple Threshold (Basit esikleme)
# =============================================================================
print("[3/5] Simple Threshold (esik=150)...")
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
cv2.imwrite(f"{output_dir}/filter3_binary.jpg", binary)
print("      -> filter3_binary.jpg")

# =============================================================================
# FILTER 4: Adaptive Gaussian Threshold
# =============================================================================
print("[4/5] Adaptive Gaussian Threshold (blockSize=11, C=2)...")
adaptive = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=11,
    C=2
)
cv2.imwrite(f"{output_dir}/filter4_adaptive.jpg", adaptive)
print("      -> filter4_adaptive.jpg")

# =============================================================================
# FILTER 5: Adaptive Threshold + Morphological Dilation
# =============================================================================
print("[5/5] Adaptive + Dilation (yazilari kalinlastir)...")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
morph = cv2.dilate(adaptive, kernel, iterations=1)
cv2.imwrite(f"{output_dir}/filter5_morph.jpg", morph)
print("      -> filter5_morph.jpg")

# =============================================================================
# OZET
# =============================================================================
print("\n" + "="*50)
print("TAMAMLANDI")
print("="*50)
print(f"\nCikti dosyalari:")
print(f"  {output_dir}/filter1_gray.jpg     - Gri tonlama")
print(f"  {output_dir}/filter2_clahe.jpg    - CLAHE (Kontrast Iyilestirme)")
print(f"  {output_dir}/filter3_binary.jpg   - Basit Threshold")
print(f"  {output_dir}/filter4_adaptive.jpg - Adaptive Threshold")
print(f"  {output_dir}/filter5_morph.jpg    - Adaptive + Dilation")
print(f"\nKlasor: {os.path.abspath(output_dir)}")
