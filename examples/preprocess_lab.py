#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
OCR ON ISLEME LABORATUVARI (Preprocessing Lab)
=============================================================================

Bu script, zorlu fis goruntulerini (golgeli, arka planli, egik) OCR icin
hazirlamak amaciyla 5 asamali bir goruntu isleme pipeline'i uygular.

Her adimin ciktisi ayri dosya olarak kaydedilir (sunum/debug icin).

Kullanim:
    python preprocess_lab.py <resim_yolu> [cikti_klasoru]

Ornek:
    python preprocess_lab.py ../data/fis13.jpeg
    python preprocess_lab.py ../data/fis13.jpeg ./sunum_ciktilari

Yazar: OCR Project Team
Tarih: Ocak 2026
=============================================================================
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path


# =============================================================================
# YARDIMCI FONKSIYONLAR
# =============================================================================

def order_points(pts):
    """
    4 kose noktasini siralar: sol-ust, sag-ust, sag-alt, sol-alt.

    Bu siralama perspektif donusumu icin gereklidir.
    Mantik:
    - Sol-ust: x+y toplami en kucuk olan nokta
    - Sag-alt: x+y toplami en buyuk olan nokta
    - Sag-ust: y-x farki en kucuk olan nokta
    - Sol-alt: y-x farki en buyuk olan nokta
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Toplam (x+y) hesapla
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Sol-ust (en kucuk toplam)
    rect[2] = pts[np.argmax(s)]  # Sag-alt (en buyuk toplam)

    # Fark (y-x) hesapla
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Sag-ust (en kucuk fark)
    rect[3] = pts[np.argmax(diff)]  # Sol-alt (en buyuk fark)

    return rect


def four_point_transform(image, pts):
    """
    4 nokta kullanarak perspektif donusumu uygular.

    Bu fonksiyon CamScanner'in "dokuman tarama" ozelligini taklit eder:
    - Egik cekilmis bir fotografi duzlestirir
    - Sadece belge/fis alanini keser
    - "Kus bakisi" (top-down) gorunum olusturur

    Args:
        image: Orijinal goruntu
        pts: 4 kose noktasi (dortgenin koseleri)

    Returns:
        Duzlestirilmis ve kirpilmis goruntu
    """
    # Noktalari sirala
    rect = order_points(pts)
    (tl, tr, br, bl) = rect  # top-left, top-right, bottom-right, bottom-left

    # Yeni goruntunun genisligini hesapla
    # Alt kenarin genisligi
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    # Üst kenarin genisligi
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # En genis olani sec
    maxWidth = max(int(widthA), int(widthB))

    # Yeni goruntunun yuksekligini hesapla
    # Sag kenarin yuksekligi
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    # Sol kenarin yuksekligi
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # En yuksek olani sec
    maxHeight = max(int(heightA), int(heightB))

    # Hedef noktalar (duz dikdortgen)
    dst = np.array([
        [0, 0],                      # Sol-ust
        [maxWidth - 1, 0],           # Sag-ust
        [maxWidth - 1, maxHeight - 1], # Sag-alt
        [0, maxHeight - 1]           # Sol-alt
    ], dtype="float32")

    # Perspektif donusum matrisini hesapla
    M = cv2.getPerspectiveTransform(rect, dst)

    # Donusumu uygula
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def find_largest_quadrilateral(edged, min_area_ratio=0.1):
    """
    Kenar haritasinda en buyuk dortgeni (4 koseli kontur) bulur.

    Args:
        edged: Canny kenar tespiti sonucu (binary goruntu)
        min_area_ratio: Minimum alan orani (goruntu alaninin yuzdesi)

    Returns:
        4 kose noktasi veya None (bulunamazsa)
    """
    # Konturleri bul
    contours, _ = cv2.findContours(
        edged.copy(),
        cv2.RETR_EXTERNAL,      # Sadece dis konturler
        cv2.CHAIN_APPROX_SIMPLE # Basitlestirilmis kontur
    )

    if not contours:
        return None

    # Konturleri alana gore sirala (buyukten kucuge)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Goruntu alanini hesapla
    image_area = edged.shape[0] * edged.shape[1]
    min_area = image_area * min_area_ratio

    # En buyuk 5 konturu kontrol et
    for contour in contours[:5]:
        # Kontur alani minimum alanin altindaysa atla
        if cv2.contourArea(contour) < min_area:
            continue

        # Konturu basitlestir (kose sayisini azalt)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # 4 koseli mi kontrol et
        if len(approx) == 4:
            return approx.reshape(4, 2)

    # 4 koseli kontur bulunamadi - ConvexHull dene
    if contours:
        largest = contours[0]
        hull = cv2.convexHull(largest)
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None


# =============================================================================
# ANA PIPELINE FONKSIYONU
# =============================================================================

def preprocess_pipeline(image_path, output_dir=None):
    """
    5 asamali OCR on isleme pipeline'i.

    Asamalar:
        1. Grayscale & Blur (Gurultu Azaltma)
        2. Canny Edge Detection (Kenar Tespiti)
        3. Perspective Transform (Perspektif Duzeltme)
        4. Adaptive Thresholding (Golge Giderme)
        5. Morphological Operations (Temizlik)

    Args:
        image_path: Giris goruntusunun yolu
        output_dir: Çikti klasoru (varsayilan: giris klasoru/lab_output)

    Returns:
        Tuple: (final_image, warped_image)
    """

    # -------------------------------------------------------------------------
    # HAZIRLIK
    # -------------------------------------------------------------------------
    image_path = Path(image_path)

    if output_dir is None:
        output_dir = image_path.parent / "lab_output"
    else:
        output_dir = Path(output_dir)

    # Çikti klasorunu olustur
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("   OCR ON IŞLEME LABORATUVARI")
    print("   5 Asamali Goruntu Isleme Pipeline")
    print("=" * 70)
    print(f"\n   Giris  : {image_path}")
    print(f"   Çikti  : {output_dir}\n")

    # Orijinal goruntuyu yukle
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(f"Goruntu yuklenemedi: {image_path}")

    print(f"   Orijinal Boyut: {original.shape[1]} x {original.shape[0]} piksel")
    print("-" * 70)


    # =========================================================================
    # ADIM 1: GRAYSCALE & BLUR (Gurultu Azaltma)
    # =========================================================================
    print("\n[ADIM 1] Grayscale & Gaussian Blur")
    print("         Amac: Renk bilgisini kaldir, gurultuyu azalt")

    # Gri tonlamaya cevir
    # Neden? OCR icin renk bilgisi gereksiz, islem hizini artirir
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur uygula
    # Neden? Kucuk gurultuleri (kumlanma) temizler, kenar tespitini iyilestirir
    # 5x5 kernel: Orta seviye yumusatma (cok kucuk = etkisiz, cok buyuk = detay kaybi)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kaydet
    step1_path = output_dir / "step1_gray_blur.jpg"
    cv2.imwrite(str(step1_path), blurred)
    print(f"         Çikti: {step1_path.name}")


    # =========================================================================
    # ADIM 2: CANNY EDGE DETECTION (Kenar Tespiti)
    # =========================================================================
    print("\n[ADIM 2] Canny Edge Detection")
    print("         Amac: Fis/belge kenarlarini tespit et")

    # Canny algoritmasi ile kenar tespiti
    # Parametreler:
    #   - 50: Alt esik (dusuk = daha fazla kenar, gurultu dahil)
    #   - 200: Üst esik (yuksek = sadece guclu kenarlar)
    edges = cv2.Canny(blurred, 50, 200)

    # Kenarlari genislet (dilate)
    # Neden? Kopuk kenarlari birlestir, kontur tespitini kolaylastir
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    # Kaydet
    step2_path = output_dir / "step2_edges.jpg"
    cv2.imwrite(str(step2_path), edges_dilated)
    print(f"         Çikti: {step2_path.name}")


    # =========================================================================
    # ADIM 3: PERSPECTIVE TRANSFORM (Perspektif Duzeltme & Kirpma)
    # =========================================================================
    print("\n[ADIM 3] Perspective Transform (CamScanner Efekti)")
    print("         Amac: Fisi masadan ayir, egriligi duzelt")

    # En buyuk dortgeni bul (fis sinirlari)
    quad = find_largest_quadrilateral(edges_dilated)

    if quad is not None:
        print("         Dortgen bulundu! Perspektif duzeltme uygulaniyor...")

        # Debug: Konturu orijinal goruntu uzerine ciz
        contour_debug = original.copy()
        cv2.drawContours(contour_debug, [quad.astype(int)], -1, (0, 255, 0), 3)
        for point in quad:
            cv2.circle(contour_debug, tuple(point.astype(int)), 8, (0, 0, 255), -1)
        cv2.imwrite(str(output_dir / "step3_contour_debug.jpg"), contour_debug)

        # Perspektif donusumu uygula
        warped = four_point_transform(original, quad)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        print(f"         Yeni boyut: {warped.shape[1]} x {warped.shape[0]} piksel")

    else:
        # FALLBACK: Dortgen bulunamadi
        print("         UYARI: Dortgen bulunamadi!")
        print("         Fallback: Orijinal goruntu ile devam ediliyor...")

        # Kenarlardan %5 kirp (masanin bir kismini cikar)
        h, w = original.shape[:2]
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        warped = original[margin_y:h-margin_y, margin_x:w-margin_x]
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Kaydet
    step3_path = output_dir / "step3_warped.jpg"
    cv2.imwrite(str(step3_path), warped)
    print(f"         Çikti: {step3_path.name}")


    # =========================================================================
    # ADIM 4: ADAPTIVE THRESHOLDING (Golge Giderme)
    # =========================================================================
    print("\n[ADIM 4] Adaptive Thresholding")
    print("         Amac: Golgeleri sil, yazilari siyah-beyaz yap")

    # Adaptive Threshold uygula
    # Neden "Adaptive"?
    #   - Normal threshold: Tum goruntuye tek esik degeri
    #   - Adaptive: Her piksel icin cevresine gore dinamik esik
    #   - Avantaj: Esit olmayan aydinlatma/golgeleri tolere eder
    #
    # Parametreler:
    #   - 255: Maksimum deger (beyaz)
    #   - ADAPTIVE_THRESH_GAUSSIAN_C: Gaussian agirlikli ortalama
    #   - THRESH_BINARY: Siyah/beyaz cikti
    #   - 11: Blok boyutu (komsuluk alani)
    #   - 2: Ortalamadan cikarilacak sabit (fine-tuning)
    threshold = cv2.adaptiveThreshold(
        warped_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Kaydet
    step4_path = output_dir / "step4_threshold.jpg"
    cv2.imwrite(str(step4_path), threshold)
    print(f"         Çikti: {step4_path.name}")


    # =========================================================================
    # ADIM 5: MORPHOLOGICAL OPERATIONS (Temizlik)
    # =========================================================================
    print("\n[ADIM 5] Morphological Operations")
    print("         Amac: Tuz-biber gurultusunu temizle, harfleri netlestir")

    # Morfolojik islemler icin kernel (yapisal element)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Opening (Acma) = Erosion + Dilation
    # Etkisi: Kucuk beyaz noktalari (gurultu) siler
    opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel_small)

    # Closing (Kapama) = Dilation + Erosion
    # Etkisi: Kucuk siyah delikleri (harflerdeki kopukluklar) kapatir
    final = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_small)

    # Kaydet
    step5_path = output_dir / "step5_final_clean.jpg"
    cv2.imwrite(str(step5_path), final)
    print(f"         Çikti: {step5_path.name}")


    # =========================================================================
    # OZET
    # =========================================================================
    print("\n" + "=" * 70)
    print("   PIPELINE TAMAMLANDI!")
    print("=" * 70)
    print(f"""
   Olusturulan Dosyalar:
   ---------------------
   1. step1_gray_blur.jpg    -> Gri + Bulaniklastirilmis
   2. step2_edges.jpg        -> Kenar Haritasi (Canny)
   3. step3_warped.jpg       -> Perspektif Duzeltilmis (RENKLI)
   4. step4_threshold.jpg    -> Esiklenmis (Siyah-Beyaz)
   5. step5_final_clean.jpg  -> Temizlenmis Final Goruntu

   OCR Icin Onerilen:
   ---------------------
   * Renkli fis OCR'i icin  -> step3_warped.jpg
   * S/B dokuman OCR'i icin -> step5_final_clean.jpg

   Klasor: {output_dir}
""")

    return final, warped


# =============================================================================
# KOMUT SATIRI ARAYÜZÜ
# =============================================================================

def main():
    """Ana fonksiyon - komut satirindan calistirma."""

    if len(sys.argv) < 2:
        print("""
Kullanim:
    python preprocess_lab.py <resim_yolu> [cikti_klasoru]

Ornekler:
    python preprocess_lab.py ../data/fis13.jpeg
    python preprocess_lab.py ../data/fis13.jpeg ./sunum
    python preprocess_lab.py C:/resimler/fis.jpg D:/ciktilar

Aciklama:
    Bu script zorlu fis goruntulerini OCR icin hazirlar.
    Her islem adimi ayri dosya olarak kaydedilir.
        """)
        return

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        final, warped = preprocess_pipeline(image_path, output_dir)
        print("Islem basariyla tamamlandi!\n")
    except FileNotFoundError as e:
        print(f"\nHATA: {e}")
    except Exception as e:
        print(f"\nBeklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
