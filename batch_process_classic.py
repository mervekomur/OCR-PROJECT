import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# --- 1. SİGORTA: EZGİ HANIM'IN RAM AYARLARI ---
# Kod eski olsa da bu ayarlar kalmalı, yoksa laptop donar.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import glob
import gc
import numpy as np

class OriginalCleaner:
    def __init__(self):
        pass

    def clean_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None

        # 1. RESIZE: RAM dostu boyut (800px yeterli ve net)
        target_width = 800
        h, w = img.shape[:2]
        if w > target_width:
            scale = target_width / w
            new_h = int(h * scale)
            img = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)

        # --- 2. ESKİ VE GÜZEL YÖNTEM (DIVISION NORMALIZATION) ---

        # Gri Yap
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gölge Haritasını Çıkar (Blur)
        # (51,51) gölgeleri silmek için en ideal "kernel" boyutudur.
        blur = cv2.GaussianBlur(gray, (51, 51), 0)

        # BÖLME İŞLEMİ (SİHİR BURADA)
        # Resmi, kendi bulanık haline böleriz.
        # Sonuç: Arka plan bembeyaz olur, yazılar kalır.
        normalized = cv2.divide(gray, blur, scale=255)

        # Otsu Threshold (Otomatik Siyah-Beyaz)
        # Adaptive Threshold yerine Otsu kullanıyoruz çünkü Otsu
        # kağıdın genel histogramına bakar, harf harf uğraşıp bozmaz.
        thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        return thresh

# --- ÇALIŞTIRMA KISMI ---
def safe_batch_process():
    resimler = glob.glob("data/*.jpg") + glob.glob("data/*.jpeg") + glob.glob("data/*.png")
    if not resimler:
        resimler = glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png")

    if not resimler:
        print("❌ HATA: Hiç resim dosyası bulunamadı!")
        return

    processor = OriginalCleaner()
    print(f"--- MOD: ESKİ FORMÜL (FABRİKA AYARLARI) ---")
    print(f"Toplam {len(resimler)} resim işlenecek...")

    # Karışmasın diye 'results_classic' yapalım
    output_folder = "results_classic"
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    for i, resim_yolu in enumerate(resimler):
        dosya_adi = os.path.basename(resim_yolu)
        print(f"[{i+1}/{len(resimler)}] {dosya_adi}...", end=" ")

        try:
            processed = processor.clean_image(resim_yolu)
            if processed is not None:
                cv2.imwrite(f"{output_folder}/classic_{dosya_adi}", processed)
                print("✅ OK")
            else:
                print("⚠️ Fail")

            del processed
            gc.collect()
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    safe_batch_process()
