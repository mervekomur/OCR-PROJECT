"""
Fiş Ön İşleme Modülü (Sade Versiyon)
====================================
Sadece gerekli olanı yapar:
- Temiz fiş → resize + grayscale
- Kirli fiş → gölge silme + threshold
"""

import os
import sys
import cv2
import numpy as np

# RAM Optimizasyonu
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class ReceiptPreprocessor:
    """Basit ve etkili fiş ön işleme"""

    def __init__(self, target_width=800):
        self.target_width = target_width

    def process(self, image_path, force_clean=False):
        """
        Fişi işle ve OCR'a hazırla.

        Args:
            image_path: Resim dosya yolu
            force_clean: True ise sadece grayscale uygula

        Returns:
            processed_image veya None
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        # 1. Boyutlandır
        img = self._resize(img)

        # 2. Kalite kontrolü
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        is_clean = force_clean or self._is_clean(gray)

        # 3. İşlem seç
        if is_clean:
            # Temiz fiş: sadece hafif denoise
            result = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
        else:
            # Kirli fiş: gölge silme
            result = self._remove_shadow(gray)

        return result

    def _resize(self, img):
        """Boyutu ayarla (RAM koruması)"""
        h, w = img.shape[:2]
        if w > self.target_width:
            scale = self.target_width / w
            new_h = int(h * scale)
            img = cv2.resize(img, (self.target_width, new_h), interpolation=cv2.INTER_AREA)
        return img

    def _is_clean(self, gray):
        """Resim temiz mi kontrol et"""
        # Laplacian varyansı (detay/keskinlik ölçümü)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Yüksek varyans = detaylı/temiz resim
        return laplacian_var > 500

    def _remove_shadow(self, gray):
        """Gölge silme (klasik formül)"""
        # Arka plan tahmini
        blur = cv2.GaussianBlur(gray, (51, 51), 0)
        # Normalizasyon (gölgeleri siler)
        normalized = cv2.divide(gray, blur, scale=255)
        # Otsu threshold
        _, thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh


# Kullanım kolaylığı için fonksiyon
def preprocess(image_path, force_clean=False):
    """
    Tek satırda fiş işleme.

    Kullanım:
        from preprocessing_final import preprocess
        result = preprocess("fis.jpg")
    """
    processor = ReceiptPreprocessor()
    return processor.process(image_path, force_clean)


# Test
if __name__ == "__main__":
    import glob
    import gc

    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    files = glob.glob("data/*.jpg") + glob.glob("data/*.jpeg") + glob.glob("data/*.png")

    if not files:
        print("Resim bulunamadı!")
        exit()

    print(f"=== SADE PREPROCESSING ===")
    print(f"{len(files)} resim işlenecek\n")

    os.makedirs("results_simple", exist_ok=True)
    processor = ReceiptPreprocessor()

    temiz, kirli = 0, 0

    for i, f in enumerate(files, 1):
        name = os.path.basename(f)

        img = cv2.imread(f)
        if img is None:
            print(f"[{i}] {name} - ❌ okunamadı")
            continue

        img = processor._resize(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        is_clean = processor._is_clean(gray)

        result = processor.process(f)
        if result is not None:
            cv2.imwrite(f"results_simple/simple_{name}", result)
            status = "temiz" if is_clean else "işlendi"
            symbol = "✅" if is_clean else "🔧"
            print(f"[{i}] {name} - {symbol} {status}")
            if is_clean:
                temiz += 1
            else:
                kirli += 1

        del result
        gc.collect()

    print(f"\n=== SONUÇ ===")
    print(f"Temiz: {temiz}")
    print(f"İşlendi: {kirli}")
