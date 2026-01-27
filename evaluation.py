# -*- coding: utf-8 -*-
"""
OCR Evaluation Script
=====================
Ground truth ile OCR sonuclarini karsilastirir.
PaddleOCR ve Google Vision OCR icin performans olcumu yapar.

Kullanim:
    python evaluation.py --engine paddle
    python evaluation.py --engine google
    python evaluation.py --engine all
"""

import os
import sys
import json
import re
import glob
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# RAM Ayarlari
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)


# =============================================================================
# PARSER FONKSIYONLARI
# =============================================================================

def parse_date(text):
    """Metinden tarih cikar - Gelistirilmis"""

    # Oncelikli: TARIH/DATE kelimesi yanindaki tarih
    priority_patterns = [
        r'TARIH\s*[:\s]*(\d{2})[./\-](\d{2})[./\-](\d{2,4})',
        r'DATE\s*[:\s]*(\d{2})[./\-](\d{2})[./\-](\d{2,4})',
        r'FECHA\s*[:\s]*(\d{2})[./\-](\d{2})[./\-](\d{2,4})',
        r'Tarih\s*[:\s]*(\d{2})[./\-](\d{2})[./\-](\d{2,4})',
    ]

    for pattern in priority_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result = _validate_and_format_date(match.groups())
            if result:
                return result

    # Genel tarih pattern'leri
    general_patterns = [
        r'(\d{2})[./\-](\d{2})[./\-](20\d{2})',   # DD/MM/YYYY
        r'(20\d{2})[./\-](\d{2})[./\-](\d{2})',    # YYYY-MM-DD
        r'(\d{2})[./\-](\d{2})[./\-](\d{2})\b',    # DD/MM/YY
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
        if len(match[0]) == 4:  # YYYY-MM-DD
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
    """Metinden toplam tutari cikar - Gelistirilmis v2"""
    text_upper = text.upper()
    lines = text_upper.split('\n')

    def parse_amount(val_str):
        """Tutari parse et: *305,00 -> 305.00"""
        val_str = re.sub(r'^[\*\+\-]', '', val_str)  # Bas karakterleri temizle
        val_str = re.sub(r'\.(?=\d{3})', '', val_str)  # Binlik ayirici
        val_str = val_str.replace(',', '.')
        # Nokta yoksa ve 3+ hane varsa son 2 haneyi ondalik yap
        if '.' not in val_str and len(val_str) >= 3:
            val_str = val_str[:-2] + '.' + val_str[-2:]
        return float(val_str)

    # 1. AYNI SATIRDA TOPLAM ve DEGER
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

    # 2. ARDISIK SATIRLARDA: TOPLAM + sonraki satir deger
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

    # 3. EURO/TL/CHF belirteci olan satirlar
    currency_patterns = [
        r'€\s*(\d+[.,]\d{2})',
        r'(\d+[.,]\d{2})\s*€',
        r'(\d+[.,]\d{2})\s*EUR\b',
        r'(\d+[.,]\d{2})\s*TL\b',
        r'(\d+[.,]\d{2})\s*CHF\b',
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

    # 4. FALLBACK: En buyuk mantikli sayi
    all_amounts = re.findall(r'(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})', text)
    if all_amounts:
        values = []
        for a in all_amounts:
            try:
                v = parse_amount(a)
                if 5 <= v <= 50000:
                    values.append(v)
            except:
                pass
        if values:
            return max(values)

    return None


def parse_kdv(text):
    """Metinden KDV tutarini cikar - Gelistirilmis v2"""
    text_upper = text.upper()
    lines = text_upper.split('\n')

    # 1. AYNI SATIRDA KDV ve DEGER
    same_line_patterns = [
        r'TOPKDV[:\s]*[\*\+]?(\d+[.,]?\d{0,2})',
        r'TOP\s*KDV[:\s]*[\*\+]?(\d+[.,]?\d{0,2})',
        r'KDV[:\s]*[\*\+]?(\d+[.,]\d{2})',
        r'TVA[:\s]*[€]?\s*(\d+[.,]\d{2})',
        r'TOTAL\s*TVA[:\s]*[€]?\s*(\d+[.,]\d{2})',
    ]

    for line in lines:
        # TOPLAM satirini atla (KDV degil)
        if 'TOPLAM' in line and 'KDV' not in line:
            continue

        for pattern in same_line_patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    val_str = match.group(1).replace(',', '.')
                    # Nokta yoksa ekle (2886 -> 28.86)
                    if '.' not in val_str and len(val_str) >= 3:
                        val_str = val_str[:-2] + '.' + val_str[-2:]
                    amount = float(val_str)
                    if 0.5 <= amount <= 500:
                        return amount
                except:
                    pass

    # 2. ARDISIK SATIRLARDA: TOPKDV + sonraki satir deger
    for i, line in enumerate(lines):
        # TOPKDV veya KDV kelimesi tek basina
        if line.strip() in ['TOPKDV', 'TOP KDV', 'KDV', 'TVA']:
            # Sonraki satira bak
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # Deger pattern'i: *28.86, +859, 13.64 gibi
                match = re.search(r'[\*\+]?(\d+[.,]?\d{0,2})', next_line)
                if match:
                    try:
                        val_str = match.group(1).replace(',', '.')
                        # Nokta yoksa ekle
                        if '.' not in val_str and len(val_str) >= 3:
                            val_str = val_str[:-2] + '.' + val_str[-2:]
                        amount = float(val_str)
                        if 0.5 <= amount <= 500:
                            return amount
                    except:
                        pass

    # 3. FALLBACK: TVA % satirinda
    for line in lines:
        match = re.search(r'TVA\s*\d+[.,]?\d*\s*%[:\s]*[€]?\s*(\d+[.,]\d{2})', line)
        if match:
            try:
                amount = float(match.group(1).replace(',', '.'))
                if 0.5 <= amount <= 500:
                    return amount
            except:
                pass

    return None


def parse_vergi_no(text):
    """Metinden vergi numarasi cikar - Gelistirilmis v2"""
    text_upper = text.upper()
    lines = text_upper.split('\n')

    # Turkiye Vergi No pattern'leri
    tr_patterns = [
        # V.D. bitisik yazilmis: DOCANBEYV.D6490494407
        r'V\.?\s*D\.?\s*(\d{10,11})',
        r'[A-Z]+V\.?D\.?(\d{10,11})',  # KADIKOYV.D.1234567890
        r'V\.?\s*D\.?\s*[:\s]*[A-Z\s]*(\d{10,11})',

        # Standart formatlar
        r'VKN[:\s]*(\d{10,11})',
        r'TCKN[:\s]*(\d{11})',
        r'VERGI\s*NO[:\s]*(\d{10,11})',
        r'VERGI\s*NUMARASI[:\s]*(\d{10,11})',

        # Ticaret Sicil
        r'T\.?\s*SICIL\s*N?O?[:\s]*(\d{6,10})',
        r'TIC\.?\s*SICIL[:\s]*N?[:\s]*(\d{6,10})',
    ]

    # Fransa pattern'leri
    fr_patterns = [
        r'SIRET[:\s]*(\d{3}\s*\d{3}\s*\d{3}\s*\d{5})',
        r'SIRET[:\s]*(\d{14})',
        r'N[°o]?\s*TVA[:\s]*(FR\d{11})',
        r'TVA\s*INTRA[:\s]*(FR\d{11})',
    ]

    # Diger ulkeler
    other_patterns = [
        r'RNC[:\s\-]*(\d{3}[\-\s]?\d{3}[\-\s]?\d{3})',
        r'UID[:\s]*(CHE[\-\s]?\d{3}[\-\s]?\d{3}[\-\s]?\d{3})',
        r'MWST[:\s]*(\d{9})',
    ]

    all_patterns = tr_patterns + fr_patterns + other_patterns

    # Satir bazli arama
    for line in lines:
        for pattern in all_patterns:
            match = re.search(pattern, line)
            if match:
                result = re.sub(r'[\s\-\.]', '', match.group(1))
                if len(result) >= 9:
                    return result

    # Ozel durum: Satirda sadece 10-11 haneli sayi (V.D. iceren satirdan sonra)
    for i, line in enumerate(lines):
        if 'V.D' in line or 'V D' in line or 'VD' in line:
            # Ayni satirda sayi ara
            numbers = re.findall(r'(\d{10,11})', line)
            if numbers:
                return numbers[-1]  # Son sayi genellikle VKN

            # Sonraki satirda sayi ara
            if i + 1 < len(lines):
                numbers = re.findall(r'(\d{10,11})', lines[i + 1])
                if numbers:
                    return numbers[0]

    # Fallback: KURUMLAR V.D veya benzeri yakininda
    for line in lines:
        if any(kw in line for kw in ['KURUMLAR', 'VERGI', 'VKN']):
            numbers = re.findall(r'(\d{10,11})', line)
            if numbers:
                return numbers[-1]

    return None


# =============================================================================
# OCR ENGINE'LER
# =============================================================================

class PaddleOCREngine:
    """PaddleOCR Engine"""

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

    def read(self, image_path):
        """Resimden metin oku"""
        try:
            result = self.ocr.ocr(image_path, cls=True)
            if result and result[0]:
                lines = [line[1][0] for line in result[0] if line[1][1] > 0.3]
                return '\n'.join(lines)
        except Exception as e:
            print(f"  PaddleOCR Hata: {e}")
        return ""


class GoogleOCREngine:
    """Google Cloud Vision OCR Engine"""

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

    def read(self, image_path):
        """Resimden metin oku"""
        if not self.available:
            return ""

        try:
            from google.cloud import vision

            with open(image_path, 'rb') as f:
                content = f.read()

            image = vision.Image(content=content)
            response = self.client.text_detection(image=image)

            if response.text_annotations:
                return response.text_annotations[0].description
        except Exception as e:
            print(f"  Google OCR Hata: {e}")
        return ""


# =============================================================================
# EVALUATION
# =============================================================================

def normalize_date(date_str):
    """Tarih formatini normalize et (DD/MM/YYYY)"""
    if not date_str or date_str == "???":
        return None

    # Farkli formatlari dene
    date_str = str(date_str).strip()

    # YYYY-MM-DD -> DD/MM/YYYY
    match = re.match(r'(\d{4})[./\-](\d{2})[./\-](\d{2})', date_str)
    if match:
        return f"{match.group(3)}/{match.group(2)}/{match.group(1)}"

    # DD/MM/YYYY veya DD-MM-YYYY
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
        return None  # Ground truth yok, karsilastirma yapma

    if ocr_norm is None:
        return False  # OCR bulamadi

    return gt_norm == ocr_norm


def compare_amounts(gt_amount, ocr_amount, tolerance=0.01):
    """Tutarlari karsilastir (kuçuk toleransla)"""
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

    # Sadece rakamlari karsilastir
    gt_clean = re.sub(r'\D', '', str(gt_vn))
    ocr_clean = re.sub(r'\D', '', str(ocr_vn))

    # Tam eslesme veya icerme kontrolu
    return gt_clean == ocr_clean or gt_clean in ocr_clean or ocr_clean in gt_clean


def evaluate_single(gt_data, ocr_text):
    """Tek bir fis icin evaluation"""

    # OCR'dan parse et
    ocr_date = parse_date(ocr_text)
    ocr_total = parse_total(ocr_text)
    ocr_kdv = parse_kdv(ocr_text)
    ocr_vergi_no = parse_vergi_no(ocr_text)

    # Ground truth'dan al
    gt_date = gt_data.get('tarih')
    gt_total = gt_data.get('toplam')
    gt_kdv = gt_data.get('kdv')
    gt_vergi_no = gt_data.get('vergi_no')

    # Karsilastir
    results = {
        'tarih': {
            'gt': gt_date,
            'ocr': ocr_date,
            'match': compare_dates(gt_date, ocr_date)
        },
        'toplam': {
            'gt': gt_total,
            'ocr': ocr_total,
            'match': compare_amounts(gt_total, ocr_total)
        },
        'kdv': {
            'gt': gt_kdv,
            'ocr': ocr_kdv,
            'match': compare_amounts(gt_kdv, ocr_kdv)
        },
        'vergi_no': {
            'gt': gt_vergi_no,
            'ocr': ocr_vergi_no,
            'match': compare_vergi_no(gt_vergi_no, ocr_vergi_no)
        }
    }

    return results


def run_evaluation(engine_name='paddle', data_dir='data', gt_file='ground_truth.json'):
    """Evaluation calistir"""

    print("\n" + "=" * 60)
    print(f"OCR EVALUATION - {engine_name.upper()}")
    print("=" * 60)

    # Ground truth yukle
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    fisler = gt_data.get('fisler', {})
    print(f"Ground truth: {len(fisler)} fis\n")

    # OCR engine sec
    if engine_name == 'paddle':
        engine = PaddleOCREngine()
    elif engine_name == 'google':
        engine = GoogleOCREngine()
        if not engine.available:
            print("Google OCR kullanilamiyor!")
            return
    else:
        print(f"Bilinmeyen engine: {engine_name}")
        return

    # Sonuclar
    all_results = {}
    stats = {
        'tarih': {'correct': 0, 'wrong': 0, 'skipped': 0},
        'toplam': {'correct': 0, 'wrong': 0, 'skipped': 0},
        'kdv': {'correct': 0, 'wrong': 0, 'skipped': 0},
        'vergi_no': {'correct': 0, 'wrong': 0, 'skipped': 0},
    }

    # Her fis icin
    for i, (filename, gt) in enumerate(fisler.items(), 1):
        # Dosya yolunu bul
        image_path = os.path.join(data_dir, filename)
        if not os.path.exists(image_path):
            print(f"[{i}] {filename} - DOSYA BULUNAMADI")
            continue

        print(f"[{i}/{len(fisler)}] {filename}")

        # OCR oku
        ocr_text = engine.read(image_path)

        if not ocr_text:
            print(f"  OCR SONUC YOK")
            continue

        # Evaluate
        result = evaluate_single(gt, ocr_text)
        all_results[filename] = result

        # Sonuclari goster ve istatistik topla
        for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
            r = result[field]
            match = r['match']

            if match is None:
                stats[field]['skipped'] += 1
                status = "SKIP"
            elif match:
                stats[field]['correct'] += 1
                status = "OK"
            else:
                stats[field]['wrong'] += 1
                status = "FAIL"

            gt_val = r['gt'] if r['gt'] else '-'
            ocr_val = r['ocr'] if r['ocr'] else '-'

            if status == "FAIL":
                print(f"  {field}: {status} (GT: {gt_val}, OCR: {ocr_val})")

        print()

    # Ozet
    print("\n" + "=" * 60)
    print("OZET")
    print("=" * 60)

    for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
        s = stats[field]
        total = s['correct'] + s['wrong']
        if total > 0:
            accuracy = (s['correct'] / total) * 100
            print(f"{field:12}: {s['correct']:3}/{total:3} = %{accuracy:.1f} (skip: {s['skipped']})")
        else:
            print(f"{field:12}: N/A (tum veriler atlandı)")

    # Genel basari
    total_correct = sum(s['correct'] for s in stats.values())
    total_wrong = sum(s['wrong'] for s in stats.values())
    total_all = total_correct + total_wrong

    if total_all > 0:
        overall = (total_correct / total_all) * 100
        print(f"\n{'GENEL':12}: {total_correct:3}/{total_all:3} = %{overall:.1f}")

    # Sonuclari kaydet
    output_file = f"evaluation_results_{engine_name}.json"
    output_data = {
        'meta': {
            'engine': engine_name,
            'timestamp': datetime.now().isoformat(),
            'ground_truth_file': gt_file,
            'total_files': len(fisler)
        },
        'stats': stats,
        'results': all_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSonuclar kaydedildi: {output_file}")

    return stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='OCR Evaluation Script')
    parser.add_argument('--engine', type=str, default='paddle',
                        choices=['paddle', 'google', 'all'],
                        help='OCR engine (paddle, google, all)')
    parser.add_argument('--data', type=str, default='data',
                        help='Resim klasoru')
    parser.add_argument('--gt', type=str, default='ground_truth.json',
                        help='Ground truth JSON dosyasi')

    args = parser.parse_args()

    if args.engine == 'all':
        print("\n### PADDLE OCR ###")
        paddle_stats = run_evaluation('paddle', args.data, args.gt)

        print("\n\n### GOOGLE OCR ###")
        google_stats = run_evaluation('google', args.data, args.gt)

        # Karsilastirma
        if paddle_stats and google_stats:
            print("\n" + "=" * 60)
            print("KARSILASTIRMA")
            print("=" * 60)
            print(f"{'Alan':<12} {'PaddleOCR':<15} {'Google OCR':<15}")
            print("-" * 42)

            for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
                p = paddle_stats[field]
                g = google_stats[field]

                p_total = p['correct'] + p['wrong']
                g_total = g['correct'] + g['wrong']

                p_acc = f"{p['correct']}/{p_total}" if p_total > 0 else "N/A"
                g_acc = f"{g['correct']}/{g_total}" if g_total > 0 else "N/A"

                print(f"{field:<12} {p_acc:<15} {g_acc:<15}")
    else:
        run_evaluation(args.engine, args.data, args.gt)
