import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# --- 1. AYARLAR: EZGİ HANIM'IN RAM KURALLARI ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import glob
import gc
import numpy as np

class SmartReceiptCleaner:
    def __init__(self):
        pass

    def detect_image_quality(self, gray):
        """
        Resmin kalitesini ölç.
        Yüksek kontrast + düşük gürültü = temiz resim
        """
        # 1. Kontrast ölçümü (standart sapma)
        contrast = np.std(gray)

        # 2. Gürültü ölçümü (Laplacian varyansı)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise = laplacian.var()

        # 3. Beyaz/siyah oranı (fiş genelde beyaz zemin)
        white_ratio = np.sum(gray > 200) / gray.size

        # Karar mantığı:
        # 1. Yüksek detay (noise > 1000) = iyi kalite, işleme bozmak
        # 2. Beyaz zemin (white_ratio > 0.4) = temiz arka plan
        # 3. Yüksek kontrast (> 60) = iyi okunabilirlik
        is_clean = (noise > 1000) or (white_ratio > 0.4) or (contrast > 60)

        return is_clean, contrast, noise

    def minimal_process(self, img):
        """Temiz resimler için: Sadece grayscale + hafif denoise"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Çok hafif denoise (yazıyı bozmaz)
        denoised = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
        return denoised

    def full_process(self, img):
        """Kirli/gölgeli resimler için: CLAHE + Adaptive Threshold"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Adaptive Threshold
        thresh = cv2.adaptiveThreshold(enhanced, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 15)

        # Auto-invert
        if np.sum(thresh == 255) < (thresh.size / 2):
            thresh = cv2.bitwise_not(thresh)

        # Denoise
        final = cv2.medianBlur(thresh, 3)
        return final

    def clean_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, "okunamadı"

        # 1. RESIZE
        target_width = 1000
        h, w = img.shape[:2]
        if w > target_width:
            scale = target_width / w
            new_h = int(h * scale)
            img = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)

        # 2. KALİTE TESPİTİ
        gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        is_clean, contrast, noise = self.detect_image_quality(gray_check)

        # 3. AKILLI İŞLEM SEÇİMİ
        if is_clean:
            # Temiz resim → minimal işlem
            result = self.minimal_process(img)
            status = f"temiz (c:{contrast:.0f}, n:{noise:.0f})"
        else:
            # Kirli resim → full işlem
            result = self.full_process(img)
            status = f"işlendi (c:{contrast:.0f}, n:{noise:.0f})"

        return result, status


# --- ÇALIŞTIRMA KISMI ---
def safe_batch_process():
    resimler = glob.glob("data/*.jpg") + glob.glob("data/*.jpeg") + glob.glob("data/*.png")
    if not resimler:
        resimler = glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png")

    if not resimler:
        print("❌ HATA: Hiç resim dosyası bulunamadı!")
        return

    processor = SmartReceiptCleaner()
    print(f"--- AKILLI İŞLEM MODU ---")
    print(f"Temiz resim → Minimal işlem (grayscale + denoise)")
    print(f"Kirli resim → Full işlem (CLAHE + threshold)")
    print(f"Toplam {len(resimler)} resim işlenecek...")
    print()

    output_folder = "results_final"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    temiz = 0
    islendi = 0

    for i, resim_yolu in enumerate(resimler):
        dosya_adi = os.path.basename(resim_yolu)
        print(f"[{i+1}/{len(resimler)}] {dosya_adi}...", end=" ")

        try:
            processed, status = processor.clean_image(resim_yolu)
            if processed is not None:
                cv2.imwrite(f"{output_folder}/final_{dosya_adi}", processed)
                if "temiz" in status:
                    print(f"✅ {status}")
                    temiz += 1
                else:
                    print(f"🔧 {status}")
                    islendi += 1
            else:
                print(f"❌ {status}")

            del processed
            gc.collect()
        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n--- SONUÇ ---")
    print(f"Temiz (minimal): {temiz}")
    print(f"İşlendi (full): {islendi}")

if __name__ == "__main__":
    safe_batch_process()
