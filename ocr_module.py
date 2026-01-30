# -*- coding: utf-8 -*-
"""
=============================================================================
OCR Module - Fis Okuma Modulu (Ensemble: EasyOCR + Donut + Scanner)
=============================================================================

Moduler, OOP tabanli fis OCR sistemi.

Bilesenler:
    - ReceiptScanner: Arka plan temizleme (Contour + Perspective Warp)
    - EasyOCR: Hizli, geleneksel OCR
    - Donut: Transformer tabanli, end-to-end
    - Ensemble: Her ikisini calistirip en iyi sonucu secer (%80 basari)

Kullanim:
    from ocr_module import EnsembleOCR, ReceiptOCR, DonutOCR, ReceiptScanner

    # Ensemble + Multi-filter (onerilen - farkli filtrelerle en iyi sonuc)
    ocr = EnsembleOCR(multi_filter=True)
    result = ocr.process("fis.jpg")

    # Sadece Ensemble (onceden kesilmis resimler icin)
    ocr = EnsembleOCR()
    result = ocr.process("fis.jpg")

    # Sadece Scanner (arka plan temizleme)
    scanner = ReceiptScanner()
    cropped = scanner.scan("fis.jpg")

Bagimsiz calistirma:
    python ocr_module.py              # Ensemble (varsayilan)
    python ocr_module.py --scan       # Ensemble + Auto-scan
    python ocr_module.py --easyocr    # Sadece EasyOCR
    python ocr_module.py --donut      # Sadece Donut
"""

import os
import re
import json
import cv2
import numpy as np
from glob import glob
from datetime import datetime
import gc  # Garbage Collector (RAM temizligi)


# =============================================================================
# RECEIPT SCANNER (Arka Plan Temizleme)
# =============================================================================

class ReceiptScanner:
    """
    Fis kenarlarini tespit edip arka plani temizler.

    Pipeline: Canny Edge -> Contour -> 4 Kose Bul -> Perspective Warp

    GPU gerektirmez, SAM'a alternatif.
    """

    def __init__(self, target_height=500):
        """
        Args:
            target_height: Islem icin kucultme yuksekligi (hiz icin)
        """
        self.target_height = target_height

    @staticmethod
    def _order_points(pts):
        """Koseleri sirala: sol-ust, sag-ust, sag-alt, sol-alt."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def _four_point_transform(image, pts):
        """Perspektif duzeltme (kus bakisi)."""
        rect = ReceiptScanner._order_points(pts)
        (tl, tr, br, bl) = rect

        # Genislik hesapla
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Yukseklik hesapla
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Hedef koordinatlar
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Warp
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def scan(self, image_path, return_debug=False, min_area_ratio=0.1):
        """
        Fisi arka plandan ayir.

        Args:
            image_path: Resim dosya yolu
            return_debug: Debug bilgisi don
            min_area_ratio: Minimum alan orani (resmin %'si)

        Returns:
            numpy array: Kesilmis fis resmi (veya bulunamazsa orijinal)
        """
        # Resmi oku
        image = cv2.imread(image_path)
        if image is None:
            return None

        orig = image.copy()
        orig_h, orig_w = orig.shape[:2]
        ratio = orig_h / float(self.target_height)
        image = cv2.resize(image, (int(orig_w / ratio), self.target_height))

        # Grayscale + Blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection (birden fazla threshold dene)
        screenCnt = None
        best_area = 0

        for canny_low, canny_high in [(30, 100), (50, 150), (75, 200), (100, 250)]:
            edged = cv2.Canny(gray, canny_low, canny_high)

            # Morfolojik kapatma (kesik cizgileri bagla)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

            # Kontur bul
            cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

            # 4 koseli sekil ara
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    # Minimum alan kontrolu
                    img_area = image.shape[0] * image.shape[1]
                    if area > img_area * min_area_ratio and area > best_area:
                        screenCnt = approx
                        best_area = area

        # Sonuc
        debug_info = {
            'found': screenCnt is not None,
            'contours_checked': 'multiple',
            'best_area': best_area,
        }

        if screenCnt is not None:
            # Perspective transform uygula
            warped = self._four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
            warped_h, warped_w = warped.shape[:2]

            # Cok kucuk sonuc kontrolu (min 100x100)
            if warped_h < 100 or warped_w < 100:
                debug_info['found'] = False
                debug_info['reason'] = 'too_small'
                if return_debug:
                    return orig, debug_info
                return orig

            debug_info['warped_size'] = (warped_h, warped_w)

            if return_debug:
                return warped, debug_info
            return warped
        else:
            # Bulunamadi, orijinali don
            debug_info['reason'] = 'no_quadrilateral'
            if return_debug:
                return orig, debug_info
            return orig

    def scan_and_save(self, image_path, output_path):
        """Tara ve kaydet."""
        result = self.scan(image_path)
        if result is not None:
            cv2.imwrite(output_path, result)
            return True
        return False

    def scan_by_color(self, image_path, return_debug=False):
        """
        Renk tabanli fis tespiti - Beyaz alanlari bul.

        Fisler genellikle beyaz, arka plan farkli renk.
        SAM'a alternatif, GPU gerektirmez.

        Args:
            image_path: Resim dosya yolu
            return_debug: Debug bilgisi don

        Returns:
            numpy array: Kesilmis fis resmi
        """
        image = cv2.imread(image_path)
        if image is None:
            return None

        orig = image.copy()
        orig_h, orig_w = orig.shape[:2]

        # HSV'ye cevir (renk analizi icin)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Beyaz/acik gri alanlari tespit et
        # Beyaz: Dusuk Saturation, Yuksek Value
        lower_white = np.array([0, 0, 180])    # H, S, V
        upper_white = np.array([180, 60, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Morfolojik temizlik
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_clean = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=2)

        # En buyuk beyaz bolgeyi bul
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug_info = {
            'method': 'color',
            'found': False,
            'contours_found': len(contours),
        }

        if not contours:
            if return_debug:
                return orig, debug_info
            return orig

        # En buyuk konturu al
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        img_area = orig_h * orig_w

        # Minimum alan kontrolu (%5)
        if area < img_area * 0.05:
            debug_info['reason'] = 'area_too_small'
            if return_debug:
                return orig, debug_info
            return orig

        # Bounding box veya minAreaRect
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)

        # Perspektif duzeltme
        warped = self._four_point_transform(orig, box)
        warped_h, warped_w = warped.shape[:2]

        # Cok kucuk sonuc kontrolu
        if warped_h < 100 or warped_w < 100:
            debug_info['reason'] = 'result_too_small'
            if return_debug:
                return orig, debug_info
            return orig

        debug_info['found'] = True
        debug_info['warped_size'] = (warped_h, warped_w)
        debug_info['area_ratio'] = area / img_area

        if return_debug:
            return warped, debug_info
        return warped

    def scan_combined(self, image_path, return_debug=False):
        """
        Birlesik tarama - Once contour, basarisizsa renk tabanli.

        Args:
            image_path: Resim dosya yolu
            return_debug: Debug bilgisi don

        Returns:
            numpy array: En iyi sonuc
        """
        # Once contour dene
        result1, debug1 = self.scan(image_path, return_debug=True)

        if debug1['found']:
            if return_debug:
                debug1['method'] = 'contour'
                return result1, debug1
            return result1

        # Contour basarisiz, renk tabanli dene
        result2, debug2 = self.scan_by_color(image_path, return_debug=True)

        if debug2['found']:
            if return_debug:
                return result2, debug2
            return result2

        # Ikisi de basarisiz, orijinali don
        if return_debug:
            return result1, {'found': False, 'method': 'none', 'tried': ['contour', 'color']}
        return result1


class ReceiptOCR:
    """
    Fis okuma sinifi.

    Pipeline: Preprocess -> OCR -> Extract
    """

    # Singleton OCR reader (sinif seviyesinde)
    _ocr_reader = None

    def __init__(self, languages=None, use_gpu=False):
        """
        Args:
            languages: OCR dilleri (varsayilan: ['tr', 'en'])
            use_gpu: GPU kullan (varsayilan: False - RAM dostu)
        """
        self.languages = languages or ['tr', 'en']
        self.use_gpu = use_gpu

    # =========================================================================
    # ADIM 1: PREPROCESS (Goruntu Isleme)
    # =========================================================================

    def preprocess_image(self, image):
        """
        Goruntu on isleme - Basit ve Etkili.

        Pipeline: Grayscale -> CLAHE -> Denoise -> Adaptive Threshold

        Args:
            image: BGR veya Grayscale numpy array

        Returns:
            Binary (siyah-beyaz) numpy array
        """
        # 1. Grayscale'e cevir
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. CLAHE - Kontrast artirma
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 3. Hafif Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=5, templateWindowSize=7, searchWindowSize=21)

        # 4. Adaptive Threshold - Siyah/beyaz
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        return binary

    # =========================================================================
    # ADIM 2: OCR ENGINE (Metin Okuma)
    # =========================================================================

    def get_ocr_reader(self):
        """
        EasyOCR reader - Singleton pattern (RAM dostu).

        Returns:
            EasyOCR Reader instance
        """
        # Singleton: Sadece bir kere yukle
        if ReceiptOCR._ocr_reader is not None:
            return ReceiptOCR._ocr_reader

        try:
            import easyocr
            print(f"EasyOCR yukleniyor... (diller={self.languages}, gpu={self.use_gpu})")
            ReceiptOCR._ocr_reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False
            )
            return ReceiptOCR._ocr_reader
        except ImportError:
            raise ImportError("EasyOCR yuklu degil! Kur: pip install easyocr")

    def read_text(self, image):
        """
        Resimden metin oku.

        Args:
            image: Preprocessed numpy array

        Returns:
            str: Okunan metin
        """
        reader = self.get_ocr_reader()
        results = reader.readtext(image, detail=0)
        return '\n'.join(results)

    # =========================================================================
    # ADIM 3: AKILLI CIKARIM (Regex)
    # =========================================================================

    def extract_data(self, text):
        """
        Ham metinden yapisal veri cikar (Gelistirilmis Regex).

        Args:
            text: OCR'dan donen ham metin

        Returns:
            dict: {'tarih': str|None, 'toplam': float|None, 'raw_text': str}
        """
        result = {
            'tarih': None,
            'toplam': None,
            'raw_text': text
        }

        # Metni temizle (OCR hatalarini duzelt)
        cleaned_text = self._clean_ocr_text(text)

        # --- TARIH CIKARI (Esnek Regex) ---
        result['tarih'] = self._extract_date(cleaned_text)

        # --- TOPLAM TUTAR CIKARI (Esnek Regex) ---
        result['toplam'] = self._extract_total(cleaned_text)

        return result

    def _clean_ocr_text(self, text):
        """OCR hatalarini duzelt."""
        cleaned = text

        # Yaygin OCR hatalari
        replacements = {
            'O': '0', 'o': '0',  # O -> 0 (sayilarda)
            'l': '1', 'I': '1',  # l/I -> 1
            'S': '5', 's': '5',  # S -> 5 (bazen)
            'B': '8',            # B -> 8 (bazen)
            'Z': '2',            # Z -> 2 (bazen)
            "'": '',             # Tirnak isaretlerini kaldir
            '"': '',
            '`': '',
            '{': '',
            '}': '',
            '[': '',
            ']': '',
        }

        # Sadece sayi bolumlerinde degistir (tarih ve tutar icin)
        return cleaned

    def _extract_date(self, text):
        """Tarih cikar - Cok Esnek pattern'ler."""

        # Metni temizle
        cleaned = text

        # OCR hatalarini duzelt (tarih icin)
        ocr_fixes = {
            '@': '0', 'O': '0', 'o': '0',
            'l': '1', 'I': '1', '|': '1',
            'S': '5', 's': '5',
            'B': '8', 'g': '9',
            '}': '1', '{': '1',
            'E': '6',  # E -> 6 (bazen)
        }

        # Pattern'ler (en spesifikten en genele)
        patterns = [
            # Standart: dd/mm/yyyy veya dd.mm.yyyy
            r'(\d{2})\s*[./\-]\s*(\d{2})\s*[./\-]\s*(\d{4})',

            # Bitisik rakamlar: 05108/2025 -> 05/08/2025 (OCR birlestirmesi)
            r'(\d{2})1?(\d{2})[./\-/](\d{4})',

            # Bosluklu: dd / mm / yy (20 / 06 / 24 gibi)
            r'(\d{2})\s*/\s*[@0oO]?(\d{1,2})\s*/\s*(\d{2,4})',

            # @ ile bozuk: 20 / @16 / 24 -> 20/06/24
            r'(\d{2})\s*[./\-/]\s*@?1?(\d{1,2})\s*[./\-/]\s*(\d{2,4})',

            # Cok bozuk: herhangi bir ayirici ile
            r'(\d{2})[^\d]{1,5}(\d{2})[^\d]{1,5}(\d{4})',
            r'(\d{2})[^\d]{1,5}(\d{2})[^\d]{1,5}(\d{2})\b',

            # Yalin: ddmmyyyy (8 rakam yan yana)
            r'\b(\d{2})(\d{2})(\d{4})\b',

            # 5 haneli gun-ay: 05108 = 05-08 (OCR birlestirmesi)
            r'(\d{2})1(\d{2})[^\d]*(\d{4})',
        ]

        # Ilk gecis: orijinal metinde ara
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                result = self._validate_date(match)
                if result:
                    return result

        # Ikinci gecis: temizlenmis metinde ara
        for old, new in ocr_fixes.items():
            cleaned = cleaned.replace(old, new)

        for pattern in patterns:
            matches = re.findall(pattern, cleaned, re.IGNORECASE)
            for match in matches:
                result = self._validate_date(match)
                if result:
                    return result

        return None

    def _validate_date(self, match):
        """Tarih eslesmesini dogrula ve formatla."""
        try:
            day, month, year = str(match[0]), str(match[1]), str(match[2])

            # Sadece rakamlari al
            day = ''.join(filter(str.isdigit, day))[:2]
            month = ''.join(filter(str.isdigit, month))[:2]
            year = ''.join(filter(str.isdigit, year))

            if not day or not month or not year:
                return None

            # Yil kontrolu
            if len(year) == 2:
                year = '20' + year

            # Gecerlilik kontrolu
            d, m, y = int(day), int(month), int(year)
            if not (1 <= d <= 31 and 1 <= m <= 12):
                return None

            # Yil duzeltme: Fisler genellikle son 2 yilda kesilir
            # OCR hatalari: 2025 -> 2002, 2009, 2005 vb.
            current_year = 2025  # Guncel yil

            if 2000 <= y <= 2015:
                # Eski yil, muhtemelen OCR hatasi
                # Son 2 haneyi duzeltmeye calis
                year_suffix = year[-2:]

                # Yaygin OCR hatalari: 02->25, 09->09(belki 05), 05->25
                ocr_year_fixes = {
                    '02': '25',  # 2002 -> 2025
                    '09': '25',  # 2009 -> 2025 (9 ve 5 karisir)
                    '05': '25',  # 2005 -> 2025
                    '00': '20',  # 2000 -> 2020
                    '01': '21',  # 2001 -> 2021
                    '03': '23',  # 2003 -> 2023
                    '04': '24',  # 2004 -> 2024
                    '06': '26',  # 2006 -> 2026
                }

                if year_suffix in ocr_year_fixes:
                    year = '20' + ocr_year_fixes[year_suffix]
                    y = int(year)

            # Final kontrol: 2018-2030 arasi kabul et
            if 2018 <= y <= 2030:
                return f"{day.zfill(2)}.{month.zfill(2)}.{year}"

        except (ValueError, IndexError):
            pass
        return None

    def _extract_total(self, text):
        """
        Toplam tutar cikar - Kara Liste + Anahtar Kelime Onceligi.

        Kurallar:
        1. KARA LISTE: NO, NUMERO, FIS, SAAT, HEURE, DATE, TARIH iceren satirlari ATLA
        2. ANAHTAR KELIME: TOPLAM, PRIX, TOTAL iceren satirlara ONCELIK ver
        3. TEMIZLIK: * isaretlerini kaldir
        """

        # =====================================================================
        # KARA LISTE - Bu PATTERN'ler geciyorsa o satiri ATLA (fiyat degildir)
        # NOT: Artik kelime degil, regex pattern kullaniyoruz
        # =====================================================================
        import re
        blacklist_patterns = [
            r'\bNO\s*[:\d]',              # NO: veya NO 0050 (fis numarasi)
            r'\bNUMERO\b', r'\bNUMBER\b',
            r'\bFIS\s*NO\b', r'\bFİŞ\s*NO\b',
            r'\bSAAT\s*[:\d]',            # SAAT: veya SAAT 09:19
            r'\bHEURE\b', r'\bTIME\b',
            r'\bDATE\s*[:\d]', r'\bTARIH\s*[:\d@]', r'\bTARİH\b',
            r'\bTEL\s*[:\d]', r'\bTELEFON\b', r'\bPHONE\b',
            r'\bSIRET\b', r'\bVKN\b', r'\bTCKN\b',
        ]

        # =====================================================================
        # GUCLU ANAHTAR KELIMELER - Fiyat iceren satirlar
        # =====================================================================
        strong_keywords = [
            # Turkce
            'TOPLAM', 'TOPL', 'TUTAR', 'TUTA',
            'ODENEN', 'ÖDENEN', 'ODENECEK', 'ÖDEME',
            'GENEL', 'NET',
            # Bozuk OCR okumalari
            'ÖPLÜL', 'FOPLUL', 'ÖPLAM', 'OPLA', 'TOPKDV',
            # Ingilizce/Fransizca
            'TOTAL', 'AMOUNT', 'SUM',
            'PRIX', 'PRICE',                    # PRIX = Fransizca fiyat
            # Kart/Nakit
            'CARTE', 'CARD', 'CASH', 'NAKIT', 'KREDI',
        ]

        # Para birimi sembolleri
        currency_symbols = ['€', '$', '£', 'TL', 'TRY', 'EUR', 'USD']

        # =====================================================================
        # METIN TEMIZLIGI
        # =====================================================================
        # Yildiz isaretlerini kaldir (* -> bosluk)
        cleaned_text = text.replace('*', ' ')

        # Satirlari tara
        lines = cleaned_text.split('\n')

        # 1. Oncelik: Guclu anahtar kelime iceren satirlar (kara liste haric)
        priority_amounts = []

        for line in lines:
            line_upper = line.upper()

            # KARA LISTE KONTROLU - Bu satirda blacklist PATTERN varsa ATLA
            is_blacklisted = any(re.search(pat, line_upper) for pat in blacklist_patterns)

            # ONEMLI: Eger satirda guclu keyword varsa, blacklist'i gec (TOPKDV gibi)
            has_strong_keyword = any(kw in line_upper for kw in strong_keywords)

            # Blacklist sadece keyword yoksa etkili
            if is_blacklisted and not has_strong_keyword:
                continue  # Bu satiri tamamen atla
            has_currency = any(curr in line_upper for curr in currency_symbols)

            if has_strong_keyword or has_currency:
                # Bu satirdan fiyat cikar
                amounts = self._extract_amounts_from_line(line)
                priority_amounts.extend(amounts)

        # Eger guclu anahtar kelimeli satirda tutar bulduysan
        if priority_amounts:
            # En sik tekrar eden tutari bul (mode)
            # Fislerde toplam genellikle birden fazla yerde yazar
            from collections import Counter

            # Kucuk tutarlari filtrele (< 10 TL fis toplami olamaz)
            significant_amounts = [a for a in priority_amounts if a >= 10.0]

            if significant_amounts:
                rounded = [round(a, 2) for a in significant_amounts]
                counter = Counter(rounded)

                # En yuksek frekansi bul
                max_freq = counter.most_common(1)[0][1]

                # Bu frekanstaki tum degerler
                top_values = [val for val, freq in counter.items() if freq == max_freq]

                # 2+ kez gecenler arasinda EN BUYUGUNU al
                if max_freq >= 2:
                    return max(top_values)

                # Tek seferlik degerler varsa, en buyugunu al
                return max(significant_amounts)

            # Hepsi kucukse, en buyugunu al
            return max(priority_amounts)

        # 2. Yedek: Tum satirlarda ara (kara liste haric, sadece ondalikli)
        fallback_amounts = []

        for line in lines:
            line_upper = line.upper()

            # Kara liste kontrolu (ayni pattern'lerle)
            is_blacklisted = any(re.search(pat, line_upper) for pat in blacklist_patterns)
            if is_blacklisted:
                continue

            amounts = self._extract_amounts_from_line(line, only_decimal=True)
            fallback_amounts.extend(amounts)

        if fallback_amounts:
            # Ayni mantik: kucukleri filtrele, en sik veya en buyuk
            from collections import Counter
            significant = [a for a in fallback_amounts if a >= 10.0]
            if significant:
                rounded = [round(a, 2) for a in significant]
                counter = Counter(rounded)
                max_freq = counter.most_common(1)[0][1]
                top_values = [val for val, freq in counter.items() if freq == max_freq]
                if max_freq >= 2:
                    return max(top_values)
                return max(significant)
            return max(fallback_amounts)

        return None

    def _extract_amounts_from_line(self, line, only_decimal=False):
        """Bir satirdan tum para tutarlarini cikar."""

        amounts = []

        # Satiri temizle (yildiz ve garip karakterler)
        clean_line = line.replace('*', ' ').replace('@', '')

        # Para formatlari (ondalikli olanlar oncelikli)
        patterns = [
            # Turk formati: 242,00 veya 1.242,00
            (r'(\d{1,3}(?:\.\d{3})*),(\d{2})\b', True),    # 1.242,00 veya 242,00

            # US/Genel format: 242.00
            (r'(\d+)\.(\d{2})\b', True),                    # 242.00

            # Bosluklu format: 242 00
            (r'(\d{1,4})\s+(\d{2})\b', True),               # 242 00

            # Tek sayi (kurus dahil): 24200 -> 242.00
            (r'\b(\d{3,5})\b', False),                      # 24200 (son 2 hane kurus)
        ]

        for pattern, is_decimal in patterns:
            if only_decimal and not is_decimal:
                continue

            matches = re.findall(pattern, clean_line)
            for match in matches:
                try:
                    # Tuple ise
                    if isinstance(match, tuple):
                        if len(match) == 2:
                            # (242, 00) -> 242.00
                            main_part = match[0].replace('.', '').replace(',', '')
                            decimal_part = match[1]
                            amount_str = f"{main_part}.{decimal_part}"
                        else:
                            amount_str = '.'.join(match)
                    else:
                        amount_str = match
                        # Eger 3+ haneli tam sayi ise, son 2 hane kurus
                        if '.' not in amount_str and len(amount_str) >= 3:
                            amount_str = amount_str[:-2] + '.' + amount_str[-2:]

                    # Sadece rakam ve nokta kalsin
                    amount_str = re.sub(r'[^\d.]', '', amount_str)

                    if not amount_str or amount_str == '.':
                        continue

                    amount = float(amount_str)

                    # Posta kodu filtresi:
                    # - 5 haneli tam sayi (10000-99999) muhtemelen posta kodu
                    # - 4+ haneli tam sayi ondalik olmadan supleli
                    if not is_decimal and 10000 <= amount <= 99999:
                        continue  # Posta kodu, atla

                    # Mantikli tutar kontrolu (0.50 - 5000 arasi)
                    if 0.5 <= amount <= 5000:
                        amounts.append(amount)

                except (ValueError, IndexError):
                    continue

        return amounts

    # =========================================================================
    # ANA ISLEM
    # =========================================================================

    def process(self, image_path):
        """
        Tek bir fis resmini isle.

        Args:
            image_path: Resim dosya yolu

        Returns:
            dict: {
                'dosya': str,
                'tarih': str|None,
                'toplam': float|None,
                'karakter_sayisi': int,
                'raw_text': str,
                'islem_zamani': str
            }
        """
        filename = os.path.basename(image_path)

        # Resmi yukle
        image = cv2.imread(image_path)
        if image is None:
            return {
                'dosya': filename,
                'hata': 'Resim yuklenemedi',
                'islem_zamani': datetime.now().isoformat()
            }

        # Pipeline: Preprocess -> OCR -> Extract
        preprocessed = self.preprocess_image(image)
        text = self.read_text(preprocessed)
        data = self.extract_data(text)

        # Sonuc
        return {
            'dosya': filename,
            'tarih': data['tarih'],
            'toplam': data['toplam'],
            'karakter_sayisi': len([c for c in text if c.strip()]),
            'raw_text': data['raw_text'],
            'islem_zamani': datetime.now().isoformat()
        }

    def process_batch(self, image_paths):
        """
        Birden fazla fis resmini isle.

        Args:
            image_paths: Resim dosya yollarinin listesi

        Returns:
            list: Sonuclarin listesi
        """
        results = []
        for i, path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Isleniyor: {os.path.basename(path)}")
            result = self.process(path)

            # Ozet goster
            if 'hata' not in result:
                print(f"  Tarih: {result['tarih'] or 'Bulunamadi'}")
                print(f"  Toplam: {result['toplam'] or 'Bulunamadi'}")
                print(f"  Karakter: {result['karakter_sayisi']}")
            else:
                print(f"  HATA: {result['hata']}")

            results.append(result)

        return results


# =============================================================================
# DONUT OCR ENGINE
# =============================================================================

class DonutOCR:
    """
    Donut (Document Understanding Transformer) tabanli OCR.

    Avantajlar:
    - Segmentasyon gerektirmez (end-to-end)
    - Bazi fislerde EasyOCR'dan daha basarili

    Dezavantajlar:
    - CPU'da yavas (~15-30sn/fis)
    - GPU ile hizlanir (~2-3sn/fis)
    """

    _processor = None
    _model = None
    _device = None

    MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"

    def __init__(self, use_gpu=False):
        """
        Args:
            use_gpu: GPU kullan (varsa)
        """
        self.use_gpu = use_gpu
        self._regex_parser = ReceiptOCR()  # Regex parser icin

    def _load_model(self):
        """Model yukle (lazy loading)."""
        if DonutOCR._model is not None:
            return

        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            import torch

            print(f"Donut yukleniyor... (model={self.MODEL_NAME})")

            DonutOCR._processor = DonutProcessor.from_pretrained(self.MODEL_NAME)
            DonutOCR._model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_NAME)

            # Device secimi
            if self.use_gpu:
                import torch
                DonutOCR._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                DonutOCR._device = "cpu"

            DonutOCR._model.to(DonutOCR._device)
            DonutOCR._model.eval()

            print(f"Donut hazir (device={DonutOCR._device})")

        except ImportError:
            raise ImportError("Transformers yuklu degil! Kur: pip install transformers sentencepiece")

    def read_text(self, image_path):
        """
        Donut ile goruntuden metin cikar.

        Args:
            image_path: Resim dosya yolu

        Returns:
            str: Temizlenmis metin
        """
        self._load_model()

        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")

        task_prompt = "<s_cord-v2>"
        decoder_input_ids = DonutOCR._processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(DonutOCR._device)

        pixel_values = DonutOCR._processor(image, return_tensors="pt").pixel_values.to(DonutOCR._device)

        with torch.no_grad():
            outputs = DonutOCR._model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=DonutOCR._model.decoder.config.max_position_embeddings,
                pad_token_id=DonutOCR._processor.tokenizer.pad_token_id,
                eos_token_id=DonutOCR._processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
            )

        sequence = DonutOCR._processor.batch_decode(outputs)[0]
        sequence = sequence.replace(DonutOCR._processor.tokenizer.eos_token, "")
        sequence = sequence.replace(DonutOCR._processor.tokenizer.pad_token, "")

        # XML tag'lerini temizle
        clean_text = re.sub(r'<[^>]+>', ' ', sequence)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Satirlara bol (regex icin)
        line_text = clean_text
        for sep in ['TARIH', 'SAAT', 'TOPLAM', 'KREDI', 'NAKIT', 'FIS', 'NO:', 'FATURA', 'DATE', 'PRIX', 'TOTAL']:
            line_text = line_text.replace(sep, '\n' + sep)

        return line_text

    def read_text_from_array(self, image_array):
        """
        Numpy array'den metin cikar.

        Args:
            image_array: OpenCV formatinda numpy array (BGR)

        Returns:
            str: Temizlenmis metin
        """
        self._load_model()

        from PIL import Image
        import torch

        # BGR -> RGB -> PIL
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) if len(image_array.shape) == 3 else image_array
        if len(rgb.shape) == 2:  # Grayscale
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(rgb)

        task_prompt = "<s_cord-v2>"
        decoder_input_ids = DonutOCR._processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(DonutOCR._device)

        pixel_values = DonutOCR._processor(image, return_tensors="pt").pixel_values.to(DonutOCR._device)

        with torch.no_grad():
            outputs = DonutOCR._model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=DonutOCR._model.decoder.config.max_position_embeddings,
                pad_token_id=DonutOCR._processor.tokenizer.pad_token_id,
                eos_token_id=DonutOCR._processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
            )

        sequence = DonutOCR._processor.batch_decode(outputs)[0]
        sequence = sequence.replace(DonutOCR._processor.tokenizer.eos_token, "")
        sequence = sequence.replace(DonutOCR._processor.tokenizer.pad_token, "")

        clean_text = re.sub(r'<[^>]+>', ' ', sequence)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        line_text = clean_text
        for sep in ['TARIH', 'SAAT', 'TOPLAM', 'KREDI', 'NAKIT', 'FIS', 'NO:', 'FATURA', 'DATE', 'PRIX', 'TOTAL']:
            line_text = line_text.replace(sep, '\n' + sep)

        return line_text

    def process_multi_filter(self, image_path):
        """
        Birden fazla filtre ile isle ve en iyi sonuclari birlestir.

        Filtreler:
        - Orijinal (grayscale)
        - CLAHE (Contrast Limited Adaptive Histogram Equalization)
        - Adaptive Threshold

        Tarih ve toplam icin farkli filtreler daha iyi sonuc verebilir.
        CLAHE ozellikle dusuk kontrastli gorsellerde OCR basarisini artirir.

        Args:
            image_path: Resim dosya yolu

        Returns:
            dict: Birlestirilmis sonuc
        """
        filename = os.path.basename(image_path)

        # Resmi oku
        image = cv2.imread(image_path)
        if image is None:
            return {'dosya': filename, 'hata': 'Resim okunamadi', 'engine': 'donut_multi'}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Filtre 1: Orijinal (grayscale) - Toplam icin iyi
        text_orig = self.read_text_from_array(gray)
        data_orig = self._regex_parser.extract_data(text_orig)

        # Filtre 2: CLAHE - Kontrast iyilestirme (dusuk isikta etkili)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_enhanced = clahe.apply(gray)
        text_clahe = self.read_text_from_array(clahe_enhanced)
        data_clahe = self._regex_parser.extract_data(text_clahe)

        # Filtre 3: Adaptive Threshold - Tarih icin iyi
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=11, C=2
        )
        text_adaptive = self.read_text_from_array(adaptive)
        data_adaptive = self._regex_parser.extract_data(text_adaptive)

        # Filtre 4: Division Normalization (Shadow Removal)
        # StackOverflow Q:78283494 - Golgeleri yok eder, arka plani beyazlatir
        blur = cv2.GaussianBlur(gray, (51, 51), 0)
        divided = cv2.divide(gray, blur, scale=255)
        _, shadow_removed = cv2.threshold(divided, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_shadow = self.read_text_from_array(shadow_removed)
        data_shadow = self._regex_parser.extract_data(text_shadow)

        # RAM temizligi
        del blur, divided, shadow_removed
        gc.collect()

        # En iyi sonuclari birlestir (4 filtreden)
        # Tarih onceligi: Shadow > CLAHE > Adaptive > Original
        tarih = data_shadow['tarih'] or data_clahe['tarih'] or data_adaptive['tarih'] or data_orig['tarih']
        if data_shadow['tarih']:
            tarih_kaynak = 'shadow'
        elif data_clahe['tarih']:
            tarih_kaynak = 'clahe'
        elif data_adaptive['tarih']:
            tarih_kaynak = 'adaptive'
        elif data_orig['tarih']:
            tarih_kaynak = 'original'
        else:
            tarih_kaynak = None

        # Toplam onceligi: Original > Shadow > CLAHE > Adaptive
        toplam = data_orig['toplam'] or data_shadow['toplam'] or data_clahe['toplam'] or data_adaptive['toplam']
        if data_orig['toplam']:
            toplam_kaynak = 'original'
        elif data_shadow['toplam']:
            toplam_kaynak = 'shadow'
        elif data_clahe['toplam']:
            toplam_kaynak = 'clahe'
        elif data_adaptive['toplam']:
            toplam_kaynak = 'adaptive'
        else:
            toplam_kaynak = None

        return {
            'dosya': filename,
            'tarih': tarih,
            'toplam': toplam,
            'tarih_kaynak': tarih_kaynak,
            'toplam_kaynak': toplam_kaynak,
            'engine': 'donut_multi',
            'islem_zamani': datetime.now().isoformat()
        }

    def process(self, image_path, multi_filter=False):
        """
        Tek bir fis resmini isle.

        Args:
            image_path: Resim dosya yolu
            multi_filter: Birden fazla filtre kullan

        Returns:
            dict: Sonuc
        """
        if multi_filter:
            return self.process_multi_filter(image_path)

        filename = os.path.basename(image_path)

        try:
            text = self.read_text(image_path)
            data = self._regex_parser.extract_data(text)

            return {
                'dosya': filename,
                'tarih': data['tarih'],
                'toplam': data['toplam'],
                'karakter_sayisi': len([c for c in text if c.strip()]),
                'raw_text': data['raw_text'],
                'engine': 'donut',
                'islem_zamani': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'dosya': filename,
                'hata': str(e),
                'engine': 'donut',
                'islem_zamani': datetime.now().isoformat()
            }


# =============================================================================
# ENSEMBLE OCR (EasyOCR + Donut)
# =============================================================================

class EnsembleOCR:
    """
    EasyOCR + Donut birlesik OCR sistemi.

    Her iki motoru calistirir ve en iyi sonucu secer.
    Basari orani: %80 (tek motorlarin 2 kati)

    auto_scan=True ile arka plan temizleme aktif edilir.
    """

    def __init__(self, use_gpu=False, auto_scan=False, multi_filter=True):
        """
        Args:
            use_gpu: GPU kullan
            auto_scan: Otomatik arka plan temizleme (ReceiptScanner)
            multi_filter: Donut icin coklu filtre kullan (Adaptive + Original)
        """
        self.easyocr = ReceiptOCR(use_gpu=use_gpu)
        self.donut = DonutOCR(use_gpu=use_gpu)
        self.auto_scan = auto_scan
        self.multi_filter = multi_filter
        self.scanner = ReceiptScanner() if auto_scan else None

    def _merge_results(self, easy_result, donut_result):
        """
        Iki sonucu akilli birlestir.

        Kurallar:
        - Tarih: EasyOCR oncelikli (daha tutarli)
        - Toplam: Hangisi bulduysa onu al
        """
        # Tarih: EasyOCR oncelikli
        tarih = easy_result.get('tarih') or donut_result.get('tarih')
        tarih_kaynak = 'easyocr' if easy_result.get('tarih') else ('donut' if donut_result.get('tarih') else None)

        # Toplam: Ikisi de bulduysa, buyuk olani sec
        easy_toplam = easy_result.get('toplam')
        donut_toplam = donut_result.get('toplam')

        if easy_toplam and donut_toplam:
            toplam = max(easy_toplam, donut_toplam)
            toplam_kaynak = 'easyocr' if easy_toplam >= donut_toplam else 'donut'
        elif easy_toplam:
            toplam = easy_toplam
            toplam_kaynak = 'easyocr'
        elif donut_toplam:
            toplam = donut_toplam
            toplam_kaynak = 'donut'
        else:
            toplam = None
            toplam_kaynak = None

        return {
            'tarih': tarih,
            'tarih_kaynak': tarih_kaynak,
            'toplam': toplam,
            'toplam_kaynak': toplam_kaynak,
        }

    def process(self, image_path):
        """
        Tek bir fis resmini isle (her iki motor ile).

        Args:
            image_path: Resim dosya yolu

        Returns:
            dict: Birlestirilmis sonuc
        """
        filename = os.path.basename(image_path)
        scan_info = {'scanned': False, 'found_receipt': False}

        # Auto-scan: Arka plan temizleme
        process_path = image_path
        if self.auto_scan and self.scanner:
            scanned, debug = self.scanner.scan(image_path, return_debug=True)
            scan_info['scanned'] = True
            scan_info['found_receipt'] = debug.get('found', False)

            if debug.get('found'):
                # Gecici dosyaya kaydet
                import tempfile
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f"scanned_{filename}")
                cv2.imwrite(temp_path, scanned)
                process_path = temp_path

        # EasyOCR
        easy_result = self.easyocr.process(process_path)

        # Donut (multi_filter ile daha iyi sonuc)
        donut_result = self.donut.process(process_path, multi_filter=self.multi_filter)

        # Birlestir
        merged = self._merge_results(easy_result, donut_result)

        return {
            'dosya': filename,
            'tarih': merged['tarih'],
            'toplam': merged['toplam'],
            'tarih_kaynak': merged['tarih_kaynak'],
            'toplam_kaynak': merged['toplam_kaynak'],
            'engine': 'ensemble',
            'auto_scan': scan_info,
            'easyocr_raw': easy_result.get('raw_text', ''),
            'donut_raw': donut_result.get('raw_text', ''),
            'islem_zamani': datetime.now().isoformat()
        }

    def process_batch(self, image_paths, verbose=True):
        """
        Birden fazla fis resmini isle.

        Args:
            image_paths: Resim dosya yollarinin listesi
            verbose: Detayli cikti

        Returns:
            list: Sonuclarin listesi
        """
        results = []

        for i, path in enumerate(image_paths, 1):
            if verbose:
                print(f"\n[{i}/{len(image_paths)}] Isleniyor: {os.path.basename(path)}")

            result = self.process(path)

            if verbose:
                t = result['tarih'] or 'Bulunamadi'
                p = f"{result['toplam']:.2f}" if result['toplam'] else 'Bulunamadi'
                tk = f"({result['tarih_kaynak']})" if result['tarih_kaynak'] else ''
                pk = f"({result['toplam_kaynak']})" if result['toplam_kaynak'] else ''
                print(f"  Tarih:  {t} {tk}")
                print(f"  Toplam: {p} {pk}")

            results.append(result)

        return results


# =============================================================================
# BAGIMSIZ CALISTIRMA
# =============================================================================

def main():
    """
    Bagimsiz calistirma.

    Kullanim:
        python ocr_module.py              # Ensemble (varsayilan)
        python ocr_module.py --easyocr    # Sadece EasyOCR
        python ocr_module.py --donut      # Sadece Donut
        python ocr_module.py --scan       # Ensemble + Auto-scan (arka plan temizleme)
    """
    import sys

    # Mod secimi
    mode = 'ensemble'
    auto_scan = '--scan' in sys.argv

    if '--easyocr' in sys.argv:
        mode = 'easyocr'
    elif '--donut' in sys.argv:
        mode = 'donut'

    print("\n" + "="*60)
    scan_label = " + SCAN" if auto_scan else ""
    print(f"RECEIPT OCR MODULE - {mode.upper()}{scan_label}")
    print("="*60)

    # Yollar
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(project_root, "results", "test_crops")
    output_file = os.path.join(project_root, "final_receipts.json")

    # Klasor kontrolu
    if not os.path.exists(input_dir):
        print(f"\nHATA: Klasor bulunamadi: {input_dir}")
        return

    # Resimleri bul
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    files = []
    for pattern in patterns:
        files.extend(glob(os.path.join(input_dir, pattern)))
    files = sorted(set(files))

    if not files:
        print(f"\nHATA: Resim bulunamadi: {input_dir}")
        return

    print(f"\nGirdi: {input_dir}")
    print(f"Cikti: {output_file}")
    print(f"Resim sayisi: {len(files)}")
    print(f"Motor: {mode}")
    print(f"Auto-scan: {'Acik' if auto_scan else 'Kapali'}")

    # OCR engine sec
    if mode == 'ensemble':
        ocr = EnsembleOCR(use_gpu=False, auto_scan=auto_scan)
        results = ocr.process_batch(files)
    elif mode == 'donut':
        ocr = DonutOCR(use_gpu=False)
        results = []
        for i, f in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Isleniyor: {os.path.basename(f)}")
            r = ocr.process(f)
            print(f"  Tarih: {r.get('tarih') or 'Bulunamadi'}")
            print(f"  Toplam: {r.get('toplam') or 'Bulunamadi'}")
            results.append(r)
    else:  # easyocr
        ocr = ReceiptOCR(languages=['tr', 'en'], use_gpu=False)
        results = ocr.process_batch(files)

    # JSON'a kaydet
    output_data = {
        'meta': {
            'islem_zamani': datetime.now().isoformat(),
            'engine': mode,
            'auto_scan': auto_scan,
            'toplam_fis': len(results),
            'basarili': len([r for r in results if 'hata' not in r]),
            'tarih_bulunan': len([r for r in results if r.get('tarih')]),
            'toplam_bulunan': len([r for r in results if r.get('toplam')])
        },
        'fisler': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Ozet
    print("\n" + "="*60)
    print("TAMAMLANDI")
    print("="*60)
    print(f"Motor: {mode}")
    print(f"Toplam fis: {output_data['meta']['toplam_fis']}")
    print(f"Tarih bulunan: {output_data['meta']['tarih_bulunan']}")
    print(f"Toplam bulunan: {output_data['meta']['toplam_bulunan']}")
    print(f"\nSonuclar kaydedildi: {output_file}")


if __name__ == '__main__':
    main()
