import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# --- 1. RAM VE CPU KISITLAMALARI (EZGİ HANIM'IN EMRİ) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# PaddleOCR loglarını sustur
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

import cv2
import glob
import gc
import numpy as np
from paddleocr import PaddleOCR

# --- 2. BÖLÜM: AKILLI PREPROCESSING (SMART CLEANER) ---
class SmartReceiptCleaner:
    def __init__(self):
        pass

    def get_blur_score(self, gray_image):
        # Resmin netliğini ölçer (Laplacian Varyansı)
        return cv2.Laplacian(gray_image, cv2.CV_64F).var()

    def heavy_clean(self, img):
        """KİRLİ FİŞLER İÇİN: Division Normalization (Gölge Silme)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (51, 51), 0)
        normalized = cv2.divide(gray, blur, scale=255)
        # Otsu threshold ile kesin siyah-beyaz
        thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh

    def light_clean(self, img):
        """TEMİZ FİŞLER İÇİN: Hafif Adaptive Threshold"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Ezgi Hanım'ın parametrelerine uygun yumuşak bir geçiş
        thresh = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, 10)
        return thresh

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None, "okunamadı"

        # Resize (RAM Dostu)
        target_width = 800
        h, w = img.shape[:2]
        if w > target_width:
            scale = target_width / w
            new_h = int(h * scale)
            img = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)

        # Karar Mekanizması
        gray_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = self.get_blur_score(gray_temp)

        limit = 300 # Eşik değerimiz
        status = ""

        if blur_score < limit:
            status = f"KİRLİ (Skor: {int(blur_score)}) -> Ağır Temizlik"
            final_img = self.heavy_clean(img)
        else:
            status = f"TEMİZ (Skor: {int(blur_score)}) -> Hafif İşlem"
            final_img = self.light_clean(img)

        return final_img, status

# --- 3. BÖLÜM: ANA BORU HATTI (PIPELINE) ---
def run_full_pipeline():
    print("--- FULL PIPELINE BAŞLATILIYOR (CLEAN + OCR) ---")

    # A. PaddleOCR'ı Ezgi Hanım'ın Ayarlarıyla Başlat
    ocr = PaddleOCR(
        lang='en',
        det_db_thresh=0.25,        # Silik yazılar için düşük eşik
        det_db_box_thresh=0.6,
        det_db_unclip_ratio=1.75,  # Kutucukları genişlet
        det_limit_side_len=960,
        rec_batch_num=6,
        use_angle_cls=True,        # Yamuk fiş desteği
        use_gpu=False,
        show_log=False
    )

    cleaner = SmartReceiptCleaner()

    # Resimleri Bul
    resimler = glob.glob("data/*.jpg") + glob.glob("data/*.jpeg") + glob.glob("data/*.png")
    if not resimler:
        resimler = glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png")

    if not resimler:
        print("❌ Klasörde resim bulunamadı!")
        return

    print(f"Toplam {len(resimler)} resim bulundu.\n")

    # Çıktı Klasörü
    output_dir = "final_output"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    for i, resim_yolu in enumerate(resimler):
        dosya_adi = os.path.basename(resim_yolu)
        print(f"[{i+1}/{len(resimler)}] {dosya_adi}")

        try:
            # ADIM 1: PREPROCESSING (Temizle)
            result = cleaner.process_image(resim_yolu)
            if result[0] is None:
                print(f"   ⚠️ Atlandı: {result[1]}")
                continue

            processed_img, status_msg = result
            print(f"   {status_msg}")

            # İşlenmiş resmi geçici kaydet
            temp_path = f"{output_dir}/proc_{dosya_adi}"
            cv2.imwrite(temp_path, processed_img)

            # ADIM 2: OCR (Oku)
            ocr_result = ocr.ocr(temp_path, cls=True)

            # ADIM 3: SONUCU YAZDIR
            text_content = []

            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    text = line[1][0]
                    conf = line[1][1]
                    if conf > 0.5:
                        text_content.append(text)

                # Dosyaya yaz
                txt_path = f"{output_dir}/{dosya_adi}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(text_content))

                # Özet göster
                full_text = " ".join(text_content)
                print(f"   OCR: {full_text[:80]}...")
            else:
                print("   ⚠️ OCR sonuç yok")

            # RAM Temizliği
            del processed_img
            gc.collect()

        except Exception as e:
            print(f"   ❌ HATA: {e}")

    print(f"\n✅ Tamamlandı! Sonuçlar: {output_dir}/")

if __name__ == "__main__":
    run_full_pipeline()
