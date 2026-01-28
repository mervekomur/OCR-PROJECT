# -*- coding: utf-8 -*-
"""
OCR Benchmark Matrix
====================
Kapsamli OCR benchmark sistemi - Ezgi Hanim ve Flo ekibi icin.

Bu script:
- Birden fazla OCR motoru (PaddleOCR, EasyOCR, Donut, GoogleVision)
- Birden fazla preprocessing pipeline (Raw, Grayscale, Binary, Smart)
- Ground truth karsilastirmasi
- CSV cikti ile detayli sonuclar

Kullanim:
    python benchmark.py
    python benchmark.py --engines paddle easyocr
    python benchmark.py --preprocessors raw smart
"""

import os
import sys
import json
import re
import csv
import cv2
import gc
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# RAM Ayarlari
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)


# =============================================================================
# PARSER FONKSIYONLARI
# =============================================================================

def parse_date(text):
    """Metinden tarih cikar"""
    priority_patterns = [
        r'TARIH\s*[:\s]*(\d{2})[./\-](\d{2})[./\-](\d{2,4})',
        r'DATE\s*[:\s]*(\d{2})[./\-](\d{2})[./\-](\d{2,4})',
        r'FECHA\s*[:\s]*(\d{2})[./\-](\d{2})[./\-](\d{2,4})',
    ]

    for pattern in priority_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result = _validate_and_format_date(match.groups())
            if result:
                return result

    general_patterns = [
        r'(\d{2})[./\-](\d{2})[./\-](20\d{2})',
        r'(20\d{2})[./\-](\d{2})[./\-](\d{2})',
        r'(\d{2})[./\-](\d{2})[./\-](\d{2})\b',
    ]

    for pattern in general_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            result = _validate_and_format_date(match)
            if result:
                return result
    return None


def _validate_and_format_date(match):
    """Tarih eslesmesini dogrula ve formatla"""
    try:
        if len(match[0]) == 4:
            year, month, day = match[0], match[1], match[2]
        else:
            day, month, year = match[0], match[1], match[2]

        d, m = int(day), int(month)
        if not (1 <= d <= 31 and 1 <= m <= 12):
            return None

        if len(year) == 2:
            year = '20' + year

        y = int(year)
        if 2018 <= y <= 2030:
            return f"{str(d).zfill(2)}/{str(m).zfill(2)}/{year}"
    except:
        pass
    return None


def parse_total(text):
    """Metinden toplam tutari cikar"""
    text_upper = text.upper()
    lines = text_upper.split('\n')

    def parse_amount(val_str):
        val_str = re.sub(r'^[\*\+\-]', '', val_str)
        val_str = re.sub(r'\.(?=\d{3})', '', val_str)
        val_str = val_str.replace(',', '.')
        if '.' not in val_str and len(val_str) >= 3:
            val_str = val_str[:-2] + '.' + val_str[-2:]
        return float(val_str)

    same_line_patterns = [
        r'TOPLAM[:\s]*[\*\+]?(\d{1,6}[.,]?\d{0,2})',
        r'TOTAL[:\s]*[EURHTLCHF€\s]*[\*\+]?(\d{1,6}[.,]?\d{0,2})',
        r'GENEL\s*TOPLAM[:\s]*[\*\+]?(\d+[.,]?\d{0,2})',
        r'A\s*PAYER[:\s]*(\d+[.,]\d{2})',
        r'TUTAR[:\s]*[\*\+]?(\d+[.,]?\d{0,2})',
    ]

    for line in lines:
        for pattern in same_line_patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    amount = parse_amount(match.group(1))
                    if 1 <= amount <= 50000:
                        return amount
                except:
                    pass

    for i, line in enumerate(lines):
        if line.strip() in ['TOPLAM', 'TOTAL', 'TUTAR']:
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                match = re.search(r'[\*\+]?(\d{1,6}[.,]?\d{0,2})', next_line)
                if match:
                    try:
                        amount = parse_amount(match.group(1))
                        if 1 <= amount <= 50000:
                            return amount
                    except:
                        pass

    currency_patterns = [
        r'€\s*(\d+[.,]\d{2})',
        r'(\d+[.,]\d{2})\s*€',
        r'(\d+[.,]\d{2})\s*EUR\b',
        r'(\d+[.,]\d{2})\s*TL\b',
    ]

    found_with_currency = []
    for line in lines:
        for pattern in currency_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    amount = float(match.group(1).replace(',', '.'))
                    if 1 <= amount <= 50000:
                        found_with_currency.append(amount)
                except:
                    pass

    if found_with_currency:
        return max(found_with_currency)

    return None


def parse_kdv(text):
    """Metinden KDV tutarini cikar"""
    text_upper = text.upper()
    lines = text_upper.split('\n')

    same_line_patterns = [
        r'TOPKDV[:\s]*[\*\+]?(\d+[.,]?\d{0,2})',
        r'TOP\s*KDV[:\s]*[\*\+]?(\d+[.,]?\d{0,2})',
        r'KDV[:\s]*[\*\+]?(\d+[.,]\d{2})',
        r'TVA[:\s]*[€]?\s*(\d+[.,]\d{2})',
        r'TOTAL\s*TVA[:\s]*[€]?\s*(\d+[.,]\d{2})',
    ]

    for line in lines:
        if 'TOPLAM' in line and 'KDV' not in line:
            continue

        for pattern in same_line_patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    val_str = match.group(1).replace(',', '.')
                    if '.' not in val_str and len(val_str) >= 3:
                        val_str = val_str[:-2] + '.' + val_str[-2:]
                    amount = float(val_str)
                    if 0.5 <= amount <= 500:
                        return amount
                except:
                    pass

    for i, line in enumerate(lines):
        if line.strip() in ['TOPKDV', 'TOP KDV', 'KDV', 'TVA']:
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                match = re.search(r'[\*\+]?(\d+[.,]?\d{0,2})', next_line)
                if match:
                    try:
                        val_str = match.group(1).replace(',', '.')
                        if '.' not in val_str and len(val_str) >= 3:
                            val_str = val_str[:-2] + '.' + val_str[-2:]
                        amount = float(val_str)
                        if 0.5 <= amount <= 500:
                            return amount
                    except:
                        pass

    return None


def parse_vergi_no(text):
    """Metinden vergi numarasi cikar"""
    text_upper = text.upper()
    lines = text_upper.split('\n')

    tr_patterns = [
        r'V\.?\s*D\.?\s*(\d{10,11})',
        r'[A-Z]+V\.?D\.?(\d{10,11})',
        r'VKN[:\s]*(\d{10,11})',
        r'TCKN[:\s]*(\d{11})',
        r'VERGI\s*NO[:\s]*(\d{10,11})',
    ]

    fr_patterns = [
        r'SIRET[:\s]*(\d{3}\s*\d{3}\s*\d{3}\s*\d{5})',
        r'SIRET[:\s]*(\d{14})',
        r'N[°o]?\s*TVA[:\s]*(FR\d{11})',
    ]

    all_patterns = tr_patterns + fr_patterns

    for line in lines:
        for pattern in all_patterns:
            match = re.search(pattern, line)
            if match:
                result = re.sub(r'[\s\-\.]', '', match.group(1))
                if len(result) >= 9:
                    return result

    for i, line in enumerate(lines):
        if 'V.D' in line or 'V D' in line or 'VD' in line:
            numbers = re.findall(r'(\d{10,11})', line)
            if numbers:
                return numbers[-1]
            if i + 1 < len(lines):
                numbers = re.findall(r'(\d{10,11})', lines[i + 1])
                if numbers:
                    return numbers[0]

    return None


# =============================================================================
# PREPROCESSOR CLASSES
# =============================================================================

class BasePreprocessor(ABC):
    """Temel Preprocessor sinifi"""

    name = "base"

    @abstractmethod
    def process(self, image_path):
        """Resmi isle ve numpy array dondur"""
        pass

    def _resize_if_needed(self, img, max_width=800):
        """Resmi boyutlandir (RAM dostu)"""
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_h = int(h * scale)
            img = cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)
        return img


class RawPreprocessor(BasePreprocessor):
    """Ham goruntu - islem yok"""

    name = "raw"

    def process(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        return self._resize_if_needed(img)


class GrayscalePreprocessor(BasePreprocessor):
    """Grayscale donusum"""

    name = "grayscale"

    def process(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = self._resize_if_needed(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class BinaryPreprocessor(BasePreprocessor):
    """Otsu binary threshold"""

    name = "binary"

    def process(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = self._resize_if_needed(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


class SmartPreprocessor(BasePreprocessor):
    """Akilli pipeline - blur score'a gore karar verir"""

    name = "smart"
    BLUR_THRESHOLD = 300

    def get_blur_score(self, gray_image):
        """Resmin netligini olc (Laplacian Varyansi)"""
        return cv2.Laplacian(gray_image, cv2.CV_64F).var()

    def heavy_clean(self, img):
        """Kirli fisler icin: Division Normalization + Otsu"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (51, 51), 0)
        normalized = cv2.divide(gray, blur, scale=255)
        thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def light_clean(self, img):
        """Temiz fisler icin: Adaptive Threshold"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 10)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def process(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None

        img = self._resize_if_needed(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = self.get_blur_score(gray)

        if blur_score < self.BLUR_THRESHOLD:
            return self.heavy_clean(img)
        else:
            return self.light_clean(img)


# =============================================================================
# OCR ENGINE CLASSES
# =============================================================================

class BaseOCREngine(ABC):
    """Temel OCR Engine sinifi"""

    name = "base"
    available = True

    @abstractmethod
    def read(self, image):
        """Resimden metin oku (numpy array veya path)"""
        pass


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR Engine"""

    name = "paddle"

    def __init__(self):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                lang='en',
                det_db_thresh=0.25,
                det_db_box_thresh=0.6,
                det_db_unclip_ratio=1.75,
                det_limit_side_len=960,
                use_angle_cls=True,
                use_gpu=False,
                show_log=False
            )
            self.available = True
        except Exception as e:
            print(f"PaddleOCR yuklenemedi: {e}")
            self.available = False

    def read(self, image):
        if not self.available:
            return ""
        try:
            result = self.ocr.ocr(image, cls=True)
            if result and result[0]:
                lines = [line[1][0] for line in result[0] if line[1][1] > 0.3]
                return '\n'.join(lines)
        except Exception as e:
            print(f"  PaddleOCR Hata: {e}")
        return ""


class EasyOCREngine(BaseOCREngine):
    """EasyOCR Engine"""

    name = "easyocr"

    def __init__(self):
        try:
            import easyocr
            self.reader = easyocr.Reader(['en', 'tr'], gpu=False, verbose=False)
            self.available = True
        except Exception as e:
            print(f"EasyOCR yuklenemedi: {e}")
            print("Kurmak icin: pip install easyocr")
            self.available = False

    def read(self, image):
        if not self.available:
            return ""
        try:
            result = self.reader.readtext(image)
            if result:
                lines = [item[1] for item in result if item[2] > 0.3]
                return '\n'.join(lines)
        except Exception as e:
            print(f"  EasyOCR Hata: {e}")
        return ""


class DonutOCREngine(BaseOCREngine):
    """Donut OCR Engine (Placeholder)"""

    name = "donut"

    def __init__(self):
        self.available = False
        print("Donut OCR: Placeholder - Henuz implemente edilmedi")

    def read(self, image):
        return ""


class GoogleOCREngine(BaseOCREngine):
    """Google Cloud Vision OCR Engine"""

    name = "google"

    def __init__(self):
        try:
            from google.cloud import vision
            self.client = vision.ImageAnnotatorClient()
            self.available = True
        except Exception as e:
            print(f"Google Vision yuklenemedi: {e}")
            print("Kurmak icin: pip install google-cloud-vision")
            print("Ve GOOGLE_APPLICATION_CREDENTIALS ayarlayin")
            self.available = False

    def read(self, image):
        if not self.available:
            return ""
        try:
            from google.cloud import vision

            # Numpy array ise dosyaya yaz
            if isinstance(image, np.ndarray):
                temp_path = "_temp_google_ocr.png"
                cv2.imwrite(temp_path, image)
                with open(temp_path, 'rb') as f:
                    content = f.read()
                os.remove(temp_path)
            else:
                with open(image, 'rb') as f:
                    content = f.read()

            img = vision.Image(content=content)
            response = self.client.text_detection(image=img)

            if response.text_annotations:
                return response.text_annotations[0].description
        except Exception as e:
            print(f"  Google OCR Hata: {e}")
        return ""


# =============================================================================
# COMPARISON & EVALUATION
# =============================================================================

def normalize_date(date_str):
    """Tarih formatini normalize et"""
    if not date_str or date_str == "???":
        return None

    date_str = str(date_str).strip()

    match = re.match(r'(\d{4})[./\-](\d{2})[./\-](\d{2})', date_str)
    if match:
        return f"{match.group(3)}/{match.group(2)}/{match.group(1)}"

    match = re.match(r'(\d{2})[./\-](\d{2})[./\-](\d{2,4})', date_str)
    if match:
        year = match.group(3)
        if len(year) == 2:
            year = '20' + year
        return f"{match.group(1)}/{match.group(2)}/{year}"

    return None


def compare_dates(gt_date, ocr_date):
    """Tarihleri karsilastir"""
    gt_norm = normalize_date(gt_date)
    ocr_norm = normalize_date(ocr_date)

    if gt_norm is None:
        return None
    if ocr_norm is None:
        return False
    return gt_norm == ocr_norm


def compare_amounts(gt_amount, ocr_amount, tolerance=0.01):
    """Tutarlari karsilastir"""
    if gt_amount is None or gt_amount == "???":
        return None
    if ocr_amount is None:
        return False

    try:
        gt = float(gt_amount)
        ocr = float(ocr_amount)
        return abs(gt - ocr) <= tolerance * max(gt, 1)
    except:
        return False


def compare_vergi_no(gt_vn, ocr_vn):
    """Vergi numaralarini karsilastir"""
    if gt_vn is None or gt_vn == "???" or gt_vn == "null":
        return None
    if ocr_vn is None:
        return False

    gt_clean = re.sub(r'\D', '', str(gt_vn))
    ocr_clean = re.sub(r'\D', '', str(ocr_vn))

    return gt_clean == ocr_clean or gt_clean in ocr_clean or ocr_clean in gt_clean


def evaluate_ocr_text(gt_data, ocr_text):
    """OCR metnini ground truth ile karsilastir"""
    ocr_date = parse_date(ocr_text)
    ocr_total = parse_total(ocr_text)
    ocr_kdv = parse_kdv(ocr_text)
    ocr_vergi_no = parse_vergi_no(ocr_text)

    return {
        'tarih': {
            'gt': gt_data.get('tarih'),
            'ocr': ocr_date,
            'match': compare_dates(gt_data.get('tarih'), ocr_date)
        },
        'toplam': {
            'gt': gt_data.get('toplam'),
            'ocr': ocr_total,
            'match': compare_amounts(gt_data.get('toplam'), ocr_total)
        },
        'kdv': {
            'gt': gt_data.get('kdv'),
            'ocr': ocr_kdv,
            'match': compare_amounts(gt_data.get('kdv'), ocr_kdv)
        },
        'vergi_no': {
            'gt': gt_data.get('vergi_no'),
            'ocr': ocr_vergi_no,
            'match': compare_vergi_no(gt_data.get('vergi_no'), ocr_vergi_no)
        }
    }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """Ana Benchmark sinifi"""

    def __init__(self, data_dir='data', gt_file='ground_truth.json'):
        self.data_dir = data_dir
        self.gt_file = gt_file

        # Ground truth yukle
        with open(gt_file, 'r', encoding='utf-8') as f:
            self.gt_data = json.load(f)
        self.fisler = self.gt_data.get('fisler', {})

        # Preprocessors
        self.preprocessors = {
            'raw': RawPreprocessor(),
            'grayscale': GrayscalePreprocessor(),
            'binary': BinaryPreprocessor(),
            'smart': SmartPreprocessor(),
        }

        # OCR Engines
        self.engines = {}

        # Sonuclar
        self.results = {}

    def add_engine(self, engine):
        """OCR engine ekle"""
        if engine.available:
            self.engines[engine.name] = engine
            print(f"  + {engine.name} engine eklendi")
        else:
            print(f"  - {engine.name} engine kulanilamiyor")

    def run_single_combination(self, filename, preprocessor_name, engine_name):
        """Tek bir kombinasyonu calistir"""
        image_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(image_path):
            return None

        preprocessor = self.preprocessors[preprocessor_name]
        engine = self.engines[engine_name]

        # Preprocess
        processed = preprocessor.process(image_path)
        if processed is None:
            return None

        # Temp dosya kaydet (bazi OCR'lar path istiyor)
        temp_path = f"_temp_benchmark_{preprocessor_name}_{engine_name}.png"
        cv2.imwrite(temp_path, processed)

        # OCR
        ocr_text = engine.read(temp_path)

        # Temizle
        os.remove(temp_path)
        del processed
        gc.collect()

        if not ocr_text:
            return None

        # Ground truth ile karsilastir
        gt = self.fisler.get(filename, {})
        eval_result = evaluate_ocr_text(gt, ocr_text)

        return eval_result

    def calculate_score(self, eval_result):
        """Evaluation sonucundan skor hesapla"""
        if eval_result is None:
            return 0

        score = 0
        for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
            if field in eval_result and eval_result[field]['match'] == True:
                score += 1
        return score

    def run_benchmark(self, preprocessor_names=None, engine_names=None):
        """Tam benchmark calistir"""

        if preprocessor_names is None:
            preprocessor_names = list(self.preprocessors.keys())

        if engine_names is None:
            engine_names = list(self.engines.keys())

        print("\n" + "=" * 70)
        print("OCR BENCHMARK MATRIX")
        print("=" * 70)
        print(f"Dosya sayisi: {len(self.fisler)}")
        print(f"Preprocessors: {', '.join(preprocessor_names)}")
        print(f"OCR Engines: {', '.join(engine_names)}")
        print("=" * 70 + "\n")

        # Sonuc yapisi
        all_results = []
        combo_stats = {}  # {combo_name: {tarih: {c, w}, toplam: {c, w}, ...}}

        for combo_name in [f"{e}_{p}" for e in engine_names for p in preprocessor_names]:
            combo_stats[combo_name] = {
                'tarih': {'correct': 0, 'wrong': 0, 'skip': 0},
                'toplam': {'correct': 0, 'wrong': 0, 'skip': 0},
                'kdv': {'correct': 0, 'wrong': 0, 'skip': 0},
                'vergi_no': {'correct': 0, 'wrong': 0, 'skip': 0},
            }

        # Her dosya icin
        for i, (filename, gt) in enumerate(self.fisler.items(), 1):
            print(f"[{i}/{len(self.fisler)}] {filename}")

            row = {
                'filename': filename,
                'market': gt.get('market_adi', '-')[:30],
                'category': self._get_category(filename, gt),
            }

            # Ground truth degerleri
            row['gt_tarih'] = gt.get('tarih', '-')
            row['gt_toplam'] = gt.get('toplam', '-')
            row['gt_kdv'] = gt.get('kdv', '-')
            row['gt_vergi_no'] = gt.get('vergi_no', '-')

            best_score = -1
            best_combo = None

            # Her kombinasyon icin
            for engine_name in engine_names:
                for prep_name in preprocessor_names:
                    combo_name = f"{engine_name}_{prep_name}"

                    eval_result = self.run_single_combination(filename, prep_name, engine_name)
                    score = self.calculate_score(eval_result)

                    # CSV sutunlari
                    row[f"{combo_name}_score"] = score

                    if eval_result:
                        for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
                            match = eval_result[field]['match']
                            if match is True:
                                row[f"{combo_name}_{field}"] = 'OK'
                                combo_stats[combo_name][field]['correct'] += 1
                            elif match is False:
                                row[f"{combo_name}_{field}"] = 'FAIL'
                                combo_stats[combo_name][field]['wrong'] += 1
                            else:
                                row[f"{combo_name}_{field}"] = 'SKIP'
                                combo_stats[combo_name][field]['skip'] += 1
                    else:
                        for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
                            row[f"{combo_name}_{field}"] = 'ERR'

                    # En iyi kombinasyonu bul
                    if score > best_score:
                        best_score = score
                        best_combo = combo_name

            row['winner'] = best_combo if best_combo else '-'
            row['winner_score'] = best_score

            all_results.append(row)
            print(f"  Winner: {best_combo} (score: {best_score}/4)")

        # CSV'ye kaydet
        self._save_csv(all_results, preprocessor_names, engine_names)

        # Ozet tablosu
        self._print_summary(combo_stats, preprocessor_names, engine_names)

        return all_results, combo_stats

    def _get_category(self, filename, gt):
        """Dosya kategorisini belirle"""
        if filename.startswith('makbuz'):
            return 'Makbuz'
        currency = gt.get('para_birimi', '')
        if currency == 'TL':
            return 'TR'
        elif currency == 'EUR':
            return 'FR'
        return 'Other'

    def _save_csv(self, results, preprocessor_names, engine_names):
        """Sonuclari CSV'ye kaydet"""
        if not results:
            return

        csv_path = 'benchmark_results.csv'

        # Sutun sirasi
        columns = ['filename', 'market', 'category',
                   'gt_tarih', 'gt_toplam', 'gt_kdv', 'gt_vergi_no']

        for engine_name in engine_names:
            for prep_name in preprocessor_names:
                combo = f"{engine_name}_{prep_name}"
                columns.append(f"{combo}_score")
                for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
                    columns.append(f"{combo}_{field}")

        columns.extend(['winner', 'winner_score'])

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f"\nSonuclar kaydedildi: {csv_path}")

    def _print_summary(self, combo_stats, preprocessor_names, engine_names):
        """Ozet tablosunu yazdir"""
        print("\n" + "=" * 70)
        print("BENCHMARK OZETI")
        print("=" * 70)

        # Tablo basligi
        header = f"{'Kombinasyon':<25}"
        for field in ['Tarih', 'Toplam', 'KDV', 'VergiNo', 'GENEL']:
            header += f" {field:<10}"
        print(header)
        print("-" * 75)

        # Her kombinasyon icin
        best_overall = None
        best_acc = 0

        for engine_name in engine_names:
            for prep_name in preprocessor_names:
                combo = f"{engine_name}_{prep_name}"
                stats = combo_stats[combo]

                row = f"{combo:<25}"
                total_c = 0
                total_w = 0

                for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
                    c = stats[field]['correct']
                    w = stats[field]['wrong']
                    t = c + w
                    total_c += c
                    total_w += w

                    if t > 0:
                        acc = (c / t) * 100
                        row += f" {c}/{t} %{acc:<4.0f}"
                    else:
                        row += f" {'N/A':<10}"

                # Genel
                total = total_c + total_w
                if total > 0:
                    overall_acc = (total_c / total) * 100
                    row += f" %{overall_acc:<5.1f}"

                    if overall_acc > best_acc:
                        best_acc = overall_acc
                        best_overall = combo
                else:
                    row += f" {'N/A':<10}"

                print(row)

        print("-" * 75)
        if best_overall:
            print(f"\nEN IYI KOMBINASYON: {best_overall} (%{best_acc:.1f})")
        print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='OCR Benchmark Matrix')
    parser.add_argument('--engines', nargs='+', default=['paddle'],
                        choices=['paddle', 'easyocr', 'donut', 'google'],
                        help='OCR engines to test')
    parser.add_argument('--preprocessors', nargs='+', default=['raw', 'smart'],
                        choices=['raw', 'grayscale', 'binary', 'smart'],
                        help='Preprocessors to test')
    parser.add_argument('--data', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--gt', type=str, default='ground_truth.json',
                        help='Ground truth file')

    args = parser.parse_args()

    print("OCR Benchmark Matrix")
    print("-" * 40)
    print("Engine'ler yukleniyor...")

    runner = BenchmarkRunner(args.data, args.gt)

    # Engine'leri yukle
    if 'paddle' in args.engines:
        runner.add_engine(PaddleOCREngine())
    if 'easyocr' in args.engines:
        runner.add_engine(EasyOCREngine())
    if 'donut' in args.engines:
        runner.add_engine(DonutOCREngine())
    if 'google' in args.engines:
        runner.add_engine(GoogleOCREngine())

    if not runner.engines:
        print("\nHICBIR OCR ENGINE KULANILAMIYOR!")
        return

    # Benchmark calistir
    runner.run_benchmark(args.preprocessors, list(runner.engines.keys()))


if __name__ == '__main__':
    main()
