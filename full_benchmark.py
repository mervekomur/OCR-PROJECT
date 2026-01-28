# -*- coding: utf-8 -*-
"""
Full OCR Benchmark Matrix
=========================
Her OCR motoru + Her preprocessing adimi kombinasyonu

Preprocessor'lar:
- raw: Islem yok
- grayscale: Sadece gri tonlama
- otsu: Otsu binary threshold
- adaptive: Adaptive threshold
- shadow: Golge kaldirma (Division Normalization)
- denoise: Gurultu azaltma (GaussianBlur + Median)
- sharpen: Keskinlestirme
- all_clean: Tum temizlik adimlari birlikte

OCR Motorlari:
- PaddleOCR
- EasyOCR
- (Donut - placeholder)
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

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)


# =============================================================================
# PARSER FONKSIYONLARI
# =============================================================================

def parse_date(text):
    if not text:
        return None
    patterns = [
        r'TARIH\s*[:\s]*(\d{2})[./\-](\d{2})[./\-](\d{2,4})',
        r'DATE\s*[:\s]*(\d{2})[./\-](\d{2})[./\-](\d{2,4})',
        r'(\d{2})[./\-](\d{2})[./\-](20\d{2})',
        r'(20\d{2})[./\-](\d{2})[./\-](\d{2})',
        r'(\d{2})[./\-](\d{2})[./\-](\d{2})\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            try:
                if len(groups[0]) == 4:
                    year, month, day = groups[0], groups[1], groups[2]
                else:
                    day, month, year = groups[0], groups[1], groups[2]
                d, m = int(day), int(month)
                if not (1 <= d <= 31 and 1 <= m <= 12):
                    continue
                if len(year) == 2:
                    year = '20' + year
                return f"{str(d).zfill(2)}/{str(m).zfill(2)}/{year}"
            except:
                continue
    return None


def parse_total(text):
    if not text:
        return None
    text_upper = text.upper()
    lines = text_upper.split('\n')

    def parse_amount(val_str):
        val_str = re.sub(r'^[\*\+\-]', '', val_str)
        val_str = re.sub(r'\.(?=\d{3})', '', val_str)
        val_str = val_str.replace(',', '.')
        if '.' not in val_str and len(val_str) >= 3:
            val_str = val_str[:-2] + '.' + val_str[-2:]
        return float(val_str)

    patterns = [
        r'TOPLAM[:\s]*[\*\+]?(\d{1,6}[.,]?\d{0,2})',
        r'TOTAL[:\s]*[EURHTLCHF€\s]*[\*\+]?(\d{1,6}[.,]?\d{0,2})',
        r'GENEL\s*TOPLAM[:\s]*[\*\+]?(\d+[.,]?\d{0,2})',
        r'A\s*PAYER[:\s]*(\d+[.,]\d{2})',
    ]

    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    amount = parse_amount(match.group(1))
                    if 1 <= amount <= 50000:
                        return amount
                except:
                    pass
    return None


def parse_kdv(text):
    if not text:
        return None
    text_upper = text.upper()
    lines = text_upper.split('\n')

    patterns = [
        r'TOPKDV[:\s]*[\*\+]?(\d+[.,]?\d{0,2})',
        r'TOP\s*KDV[:\s]*[\*\+]?(\d+[.,]?\d{0,2})',
        r'KDV[:\s]*[\*\+]?(\d+[.,]\d{2})',
        r'TVA[:\s]*[€]?\s*(\d+[.,]\d{2})',
    ]

    for line in lines:
        if 'TOPLAM' in line and 'KDV' not in line:
            continue
        for pattern in patterns:
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

    # Multi-line check
    for i, line in enumerate(lines):
        if line.strip() in ['TOPKDV', 'TOP KDV', 'KDV', 'TVA']:
            if i + 1 < len(lines):
                match = re.search(r'[\*\+]?(\d+[.,]?\d{0,2})', lines[i + 1])
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
    if not text:
        return None
    text_upper = text.upper()

    patterns = [
        r'V\.?\s*D\.?\s*(\d{10,11})',
        r'VKN[:\s]*(\d{10,11})',
        r'SIRET[:\s]*(\d{14})',
        r'SIRET[:\s]*(\d{3}\s*\d{3}\s*\d{3}\s*\d{5})',
    ]

    for pattern in patterns:
        match = re.search(pattern, text_upper)
        if match:
            result = re.sub(r'[\s\-\.]', '', match.group(1))
            if len(result) >= 9:
                return result

    # V.D. yakınında sayı ara
    lines = text_upper.split('\n')
    for line in lines:
        if 'V.D' in line or 'V D' in line:
            numbers = re.findall(r'(\d{10,11})', line)
            if numbers:
                return numbers[-1]
    return None


# =============================================================================
# PREPROCESSING FONKSIYONLARI
# =============================================================================

def preprocess_raw(img):
    """Islem yok - orijinal"""
    return img.copy()


def preprocess_grayscale(img):
    """Sadece grayscale"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def preprocess_otsu(img):
    """Otsu binary threshold"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def preprocess_adaptive(img):
    """Adaptive threshold"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, 10)
    return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)


def preprocess_shadow(img):
    """Golge kaldirma (Division Normalization)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    normalized = cv2.divide(gray, blur, scale=255)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def preprocess_denoise(img):
    """Gurultu azaltma"""
    denoised = cv2.GaussianBlur(img, (3, 3), 0)
    denoised = cv2.medianBlur(denoised, 3)
    return denoised


def preprocess_sharpen(img):
    """Keskinlestirme"""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened


def preprocess_shadow_otsu(img):
    """Golge kaldirma + Otsu"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    normalized = cv2.divide(gray, blur, scale=255)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def preprocess_all_clean(img):
    """Tum temizlik adimlari"""
    # 1. Denoise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # 2. Shadow removal
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    normalized = cv2.divide(gray, blur, scale=255)
    # 3. Otsu
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


PREPROCESSORS = {
    'raw': ('Ham (Islenmemis)', preprocess_raw),
    'grayscale': ('Grayscale', preprocess_grayscale),
    'otsu': ('Otsu Threshold', preprocess_otsu),
    'adaptive': ('Adaptive Threshold', preprocess_adaptive),
    'shadow': ('Golge Kaldirma', preprocess_shadow),
    'denoise': ('Gurultu Azaltma', preprocess_denoise),
    'sharpen': ('Keskinlestirme', preprocess_sharpen),
    'shadow_otsu': ('Golge + Otsu', preprocess_shadow_otsu),
    'all_clean': ('Tum Temizlik', preprocess_all_clean),
}


# =============================================================================
# OCR ENGINES
# =============================================================================

class PaddleEngine:
    name = "PaddleOCR"

    def __init__(self):
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

    def read(self, img_path):
        try:
            result = self.ocr.ocr(img_path, cls=True)
            if result and result[0]:
                lines = [line[1][0] for line in result[0] if line[1][1] > 0.3]
                return '\n'.join(lines)
        except Exception as e:
            pass
        return ""


class EasyEngine:
    name = "EasyOCR"

    def __init__(self):
        try:
            import easyocr
            self.reader = easyocr.Reader(['en', 'tr'], gpu=False, verbose=False)
            self.available = True
        except:
            self.available = False
            print("  ! EasyOCR yuklenemedi (pip install easyocr)")

    def read(self, img_path):
        if not self.available:
            return ""
        try:
            result = self.reader.readtext(img_path)
            if result:
                lines = [item[1] for item in result if item[2] > 0.3]
                return '\n'.join(lines)
        except:
            pass
        return ""


# =============================================================================
# COMPARISON
# =============================================================================

def normalize_date(date_str):
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


def compare_values(gt_val, ocr_val, field_type):
    """Degerleri karsilastir ve sonuc dondur: OK, FAIL, SKIP"""
    if gt_val is None or gt_val == "???" or gt_val == "null" or gt_val == "":
        return "SKIP"

    if ocr_val is None:
        return "FAIL"

    if field_type == 'date':
        gt_norm = normalize_date(gt_val)
        ocr_norm = normalize_date(ocr_val)
        if gt_norm and ocr_norm and gt_norm == ocr_norm:
            return "OK"
        return "FAIL"

    elif field_type == 'amount':
        try:
            gt_f = float(gt_val)
            ocr_f = float(ocr_val)
            if abs(gt_f - ocr_f) <= 0.01 * max(gt_f, 1):
                return "OK"
        except:
            pass
        return "FAIL"

    elif field_type == 'vergi_no':
        gt_clean = re.sub(r'\D', '', str(gt_val))
        ocr_clean = re.sub(r'\D', '', str(ocr_val))
        if gt_clean == ocr_clean or gt_clean in ocr_clean or ocr_clean in gt_clean:
            return "OK"
        return "FAIL"

    return "FAIL"


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_full_benchmark():
    print("=" * 70)
    print("FULL OCR BENCHMARK MATRIX")
    print("=" * 70)

    # Ground truth yukle
    with open('ground_truth.json', 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    fisler = gt_data.get('fisler', {})
    print(f"Toplam dosya: {len(fisler)}")

    # OCR Engine'leri baslat
    print("\nOCR Engine'ler yukleniyor...")
    engines = {}

    print("  - PaddleOCR yukleniyor...")
    engines['paddle'] = PaddleEngine()

    print("  - EasyOCR yukleniyor...")
    easy = EasyEngine()
    if easy.available:
        engines['easyocr'] = easy

    print(f"\nAktif engine'ler: {list(engines.keys())}")
    print(f"Preprocessing adimlari: {list(PREPROCESSORS.keys())}")

    # Sonuc yapisi
    all_results = []

    # Kombinasyon istatistikleri
    combo_stats = {}
    for eng_name in engines.keys():
        for prep_name in PREPROCESSORS.keys():
            combo = f"{eng_name}_{prep_name}"
            combo_stats[combo] = {'tarih': 0, 'toplam': 0, 'kdv': 0, 'vergi_no': 0, 'total': 0}

    total_files = len(fisler)

    # Her dosya icin
    for file_idx, (filename, gt) in enumerate(fisler.items(), 1):
        image_path = os.path.join('data', filename)

        if not os.path.exists(image_path):
            print(f"[{file_idx}/{total_files}] {filename} - DOSYA YOK")
            continue

        print(f"[{file_idx}/{total_files}] {filename}")

        # Resmi oku
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ! Resim okunamadi")
            continue

        # Resize
        h, w = img.shape[:2]
        if w > 800:
            scale = 800 / w
            img = cv2.resize(img, (800, int(h * scale)), interpolation=cv2.INTER_AREA)

        # Satir verisi
        row = {
            'dosya': filename,
            'market': gt.get('market_adi', '-')[:25],
            'kategori': 'Makbuz' if filename.startswith('makbuz') else ('TR' if gt.get('para_birimi') == 'TL' else 'FR'),
            'gt_tarih': gt.get('tarih', ''),
            'gt_toplam': gt.get('toplam', ''),
            'gt_kdv': gt.get('kdv', ''),
            'gt_vergi_no': gt.get('vergi_no', ''),
        }

        # Her engine + preprocessing kombinasyonu
        for eng_name, engine in engines.items():
            for prep_name, (prep_desc, prep_func) in PREPROCESSORS.items():
                combo = f"{eng_name}_{prep_name}"

                # Preprocess
                processed = prep_func(img)

                # Temp dosya kaydet
                temp_path = f"_temp_{combo}.png"
                cv2.imwrite(temp_path, processed)

                # OCR
                ocr_text = engine.read(temp_path)

                # Temizle
                os.remove(temp_path)

                # Parse
                ocr_tarih = parse_date(ocr_text)
                ocr_toplam = parse_total(ocr_text)
                ocr_kdv = parse_kdv(ocr_text)
                ocr_vergi_no = parse_vergi_no(ocr_text)

                # Karsilastir
                r_tarih = compare_values(gt.get('tarih'), ocr_tarih, 'date')
                r_toplam = compare_values(gt.get('toplam'), ocr_toplam, 'amount')
                r_kdv = compare_values(gt.get('kdv'), ocr_kdv, 'amount')
                r_vergi_no = compare_values(gt.get('vergi_no'), ocr_vergi_no, 'vergi_no')

                # Skor hesapla
                score = sum(1 for r in [r_tarih, r_toplam, r_kdv, r_vergi_no] if r == "OK")

                # Istatistik guncelle
                if r_tarih == "OK":
                    combo_stats[combo]['tarih'] += 1
                if r_toplam == "OK":
                    combo_stats[combo]['toplam'] += 1
                if r_kdv == "OK":
                    combo_stats[combo]['kdv'] += 1
                if r_vergi_no == "OK":
                    combo_stats[combo]['vergi_no'] += 1
                combo_stats[combo]['total'] += score

                # Satira ekle
                row[f"{combo}_tarih"] = r_tarih
                row[f"{combo}_toplam"] = r_toplam
                row[f"{combo}_kdv"] = r_kdv
                row[f"{combo}_vergi_no"] = r_vergi_no
                row[f"{combo}_skor"] = f"{score}/4"

        all_results.append(row)

        # RAM temizligi
        del img, processed
        gc.collect()

    # CSV kaydet
    print("\n" + "=" * 70)
    print("CSV DOSYASI OLUSTURULUYOR...")

    # Sutun sirasi
    columns = ['dosya', 'market', 'kategori', 'gt_tarih', 'gt_toplam', 'gt_kdv', 'gt_vergi_no']

    for eng_name in engines.keys():
        for prep_name in PREPROCESSORS.keys():
            combo = f"{eng_name}_{prep_name}"
            columns.extend([f"{combo}_tarih", f"{combo}_toplam", f"{combo}_kdv", f"{combo}_vergi_no", f"{combo}_skor"])

    csv_path = 'full_benchmark_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Kaydedildi: {csv_path}")

    # Ozet tablosu
    print("\n" + "=" * 70)
    print("BENCHMARK OZET TABLOSU")
    print("=" * 70)

    print(f"\n{'Kombinasyon':<25} {'Tarih':<8} {'Toplam':<8} {'KDV':<8} {'VergiNo':<8} {'TOPLAM':<8}")
    print("-" * 70)

    best_combo = None
    best_total = 0

    for eng_name in engines.keys():
        for prep_name in PREPROCESSORS.keys():
            combo = f"{eng_name}_{prep_name}"
            stats = combo_stats[combo]

            total = stats['total']
            if total > best_total:
                best_total = total
                best_combo = combo

            print(f"{combo:<25} {stats['tarih']:<8} {stats['toplam']:<8} {stats['kdv']:<8} {stats['vergi_no']:<8} {total:<8}")

    print("-" * 70)
    print(f"\nEN IYI KOMBINASYON: {best_combo} (Toplam skor: {best_total})")

    # Markdown rapor
    create_markdown_report(all_results, combo_stats, engines, fisler)

    return all_results, combo_stats


def create_markdown_report(results, combo_stats, engines, fisler):
    """Detayli markdown rapor olustur"""

    report = []
    report.append("# Full OCR Benchmark Matrix Raporu")
    report.append(f"\n**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Toplam Dosya:** {len(fisler)}")
    report.append(f"**OCR Motorlari:** {', '.join(engines.keys())}")
    report.append(f"**Preprocessing Adimlari:** {len(PREPROCESSORS)}")

    # Engine + Preprocessing aciklamalari
    report.append("\n---\n")
    report.append("## Preprocessing Adimlari Aciklamasi")
    report.append("")
    report.append("| Kod | Aciklama |")
    report.append("|-----|----------|")
    for code, (desc, _) in PREPROCESSORS.items():
        report.append(f"| {code} | {desc} |")

    # Ozet tablo
    report.append("\n---\n")
    report.append("## Kombinasyon Performans Tablosu")
    report.append("")

    header = "| Kombinasyon | Tarih | Toplam | KDV | Vergi No | TOPLAM |"
    report.append(header)
    report.append("|-------------|-------|--------|-----|----------|--------|")

    sorted_combos = sorted(combo_stats.items(), key=lambda x: x[1]['total'], reverse=True)

    for combo, stats in sorted_combos:
        row = f"| {combo} | {stats['tarih']} | {stats['toplam']} | {stats['kdv']} | {stats['vergi_no']} | **{stats['total']}** |"
        report.append(row)

    # En iyi kombinasyonlar
    report.append("\n---\n")
    report.append("## En Iyi 5 Kombinasyon")
    report.append("")
    for i, (combo, stats) in enumerate(sorted_combos[:5], 1):
        eng, prep = combo.split('_', 1)
        prep_desc = PREPROCESSORS.get(prep, (prep, None))[0]
        report.append(f"{i}. **{combo}** - Toplam: {stats['total']}")
        report.append(f"   - OCR: {eng}")
        report.append(f"   - Preprocessing: {prep_desc}")
        report.append(f"   - Tarih: {stats['tarih']}, Toplam: {stats['toplam']}, KDV: {stats['kdv']}, Vergi No: {stats['vergi_no']}")
        report.append("")

    # Dosya bazli detay
    report.append("\n---\n")
    report.append("## Dosya Bazli Sonuclar (En Iyi Kombinasyon: " + sorted_combos[0][0] + ")")
    report.append("")

    best_combo = sorted_combos[0][0]

    report.append("| Dosya | Kategori | Tarih | Toplam | KDV | Vergi No | Skor |")
    report.append("|-------|----------|-------|--------|-----|----------|------|")

    for row in results:
        dosya = row['dosya']
        kat = row['kategori']
        tarih = row.get(f"{best_combo}_tarih", "-")
        toplam = row.get(f"{best_combo}_toplam", "-")
        kdv = row.get(f"{best_combo}_kdv", "-")
        vergi = row.get(f"{best_combo}_vergi_no", "-")
        skor = row.get(f"{best_combo}_skor", "-")

        # Emoji
        t_icon = "✅" if tarih == "OK" else ("⏭️" if tarih == "SKIP" else "❌")
        to_icon = "✅" if toplam == "OK" else ("⏭️" if toplam == "SKIP" else "❌")
        k_icon = "✅" if kdv == "OK" else ("⏭️" if kdv == "SKIP" else "❌")
        v_icon = "✅" if vergi == "OK" else ("⏭️" if vergi == "SKIP" else "❌")

        report.append(f"| {dosya} | {kat} | {t_icon} | {to_icon} | {k_icon} | {v_icon} | {skor} |")

    # Kaydet
    md_path = 'FULL_BENCHMARK_REPORT.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"Markdown rapor: {md_path}")


if __name__ == '__main__':
    run_full_benchmark()
