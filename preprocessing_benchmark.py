# -*- coding: utf-8 -*-
"""
Preprocessing Benchmark - Her adimin ayri ayri testi
Gelismis parser'larla
"""
import os, sys, json, re, csv, cv2, gc
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)

# === GELISMIS PARSERS (evaluation.py'dan) ===
def parse_date(text):
    """Metinden tarih cikar - Gelistirilmis"""
    if not text:
        return None

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
    """Metinden toplam tutari cikar - Gelistirilmis v2"""
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
    if not text:
        return None

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
    if not text:
        return None

    text_upper = text.upper()
    lines = text_upper.split('\n')

    tr_patterns = [
        r'V\.?\s*D\.?\s*(\d{10,11})',
        r'[A-Z]+V\.?D\.?(\d{10,11})',
        r'V\.?\s*D\.?\s*[:\s]*[A-Z\s]*(\d{10,11})',
        r'VKN[:\s]*(\d{10,11})',
        r'TCKN[:\s]*(\d{11})',
        r'VERGI\s*NO[:\s]*(\d{10,11})',
        r'VERGI\s*NUMARASI[:\s]*(\d{10,11})',
        r'T\.?\s*SICIL\s*N?O?[:\s]*(\d{6,10})',
        r'TIC\.?\s*SICIL[:\s]*N?[:\s]*(\d{6,10})',
    ]

    fr_patterns = [
        r'SIRET[:\s]*(\d{3}\s*\d{3}\s*\d{3}\s*\d{5})',
        r'SIRET[:\s]*(\d{14})',
        r'N[°o]?\s*TVA[:\s]*(FR\d{11})',
        r'TVA\s*INTRA[:\s]*(FR\d{11})',
    ]

    other_patterns = [
        r'RNC[:\s\-]*(\d{3}[\-\s]?\d{3}[\-\s]?\d{3})',
        r'UID[:\s]*(CHE[\-\s]?\d{3}[\-\s]?\d{3}[\-\s]?\d{3})',
        r'MWST[:\s]*(\d{9})',
    ]

    all_patterns = tr_patterns + fr_patterns + other_patterns

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

    for line in lines:
        if any(kw in line for kw in ['KURUMLAR', 'VERGI', 'VKN']):
            numbers = re.findall(r'(\d{10,11})', line)
            if numbers:
                return numbers[-1]

    return None


# === PREPROCESSING ===
def prep_raw(img):
    return img.copy()

def prep_grayscale(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def prep_otsu(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

def prep_adaptive(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)

def prep_shadow(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(g, (51, 51), 0)
    n = cv2.divide(g, blur, scale=255)
    return cv2.cvtColor(n, cv2.COLOR_GRAY2BGR)

def prep_denoise(img):
    return cv2.medianBlur(cv2.GaussianBlur(img, (3, 3), 0), 3)

def prep_sharpen(img):
    k = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    return cv2.filter2D(img, -1, k)

def prep_shadow_otsu(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(g, (51, 51), 0)
    n = cv2.divide(g, blur, scale=255)
    _, b = cv2.threshold(n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

def prep_all_clean(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(g, (51, 51), 0)
    n = cv2.divide(g, blur, scale=255)
    _, b = cv2.threshold(n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

PREPS = {
    'raw': ('Ham (Islenmemis)', prep_raw),
    'grayscale': ('Grayscale', prep_grayscale),
    'otsu': ('Otsu Threshold', prep_otsu),
    'adaptive': ('Adaptive Threshold', prep_adaptive),
    'shadow': ('Golge Kaldirma', prep_shadow),
    'denoise': ('Gurultu Azaltma', prep_denoise),
    'sharpen': ('Keskinlestirme', prep_sharpen),
    'shadow_otsu': ('Golge + Otsu', prep_shadow_otsu),
    'all_clean': ('Full Clean', prep_all_clean),
}

# === COMPARE ===
def normalize_date(date_str):
    if not date_str or date_str == "???" or date_str == "null":
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

def compare(gt, ocr, typ):
    if gt is None or gt == '???' or gt == 'null' or gt == '':
        return 'SKIP'
    if ocr is None:
        return 'FAIL'

    if typ == 'date':
        g = normalize_date(gt)
        o = normalize_date(ocr)
        return 'OK' if g and o and g == o else 'FAIL'

    if typ == 'amt':
        try:
            gt_f = float(gt)
            ocr_f = float(ocr)
            return 'OK' if abs(gt_f - ocr_f) <= 0.01 * max(gt_f, 1) else 'FAIL'
        except:
            return 'FAIL'

    if typ == 'vn':
        gt_c = re.sub(r'\D', '', str(gt))
        ocr_c = re.sub(r'\D', '', str(ocr))
        return 'OK' if gt_c == ocr_c or gt_c in ocr_c or ocr_c in gt_c else 'FAIL'

    return 'FAIL'

# === MAIN ===
def main():
    print('='*70)
    print('PREPROCESSING BENCHMARK - PaddleOCR (Gelismis Parser)')
    print('='*70)

    print('\nPaddleOCR yukleniyor...')
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang='en', det_db_thresh=0.25, det_db_box_thresh=0.6,
                   det_db_unclip_ratio=1.75, use_angle_cls=True, use_gpu=False, show_log=False)

    with open('ground_truth.json', 'r', encoding='utf-8') as f:
        fisler = json.load(f).get('fisler', {})

    print(f"Dosya sayisi: {len(fisler)}")
    print(f"Preprocessing adimlari: {len(PREPS)}")
    print('='*70 + '\n')

    results = []
    stats = {p: {'tarih': 0, 'toplam': 0, 'kdv': 0, 'vergi_no': 0, 'total': 0} for p in PREPS}

    for i, (fn, gt) in enumerate(fisler.items(), 1):
        path = f'data/{fn}'
        if not os.path.exists(path):
            print(f"[{i}] {fn} - DOSYA YOK")
            continue

        print(f"[{i}/{len(fisler)}] {fn}")
        img = cv2.imread(path)
        if img is None:
            print(f"  ! Okunamadi")
            continue

        h, w = img.shape[:2]
        if w > 800:
            img = cv2.resize(img, (800, int(h * 800 / w)))

        kat = 'Makbuz' if fn.startswith('makbuz') else ('TR' if gt.get('para_birimi') == 'TL' else 'FR')

        row = {
            'dosya': fn,
            'market': gt.get('market_adi', '-')[:20],
            'kategori': kat,
            'gt_tarih': gt.get('tarih', ''),
            'gt_toplam': gt.get('toplam', ''),
            'gt_kdv': gt.get('kdv', ''),
            'gt_vergi_no': gt.get('vergi_no', '')
        }

        for pn, (pdesc, pfunc) in PREPS.items():
            proc = pfunc(img)
            tmp = f'_tmp_{pn}.png'
            cv2.imwrite(tmp, proc)

            try:
                r = ocr.ocr(tmp, cls=True)
                txt = '\n'.join([l[1][0] for l in r[0] if l[1][1] > 0.3]) if r and r[0] else ''
            except:
                txt = ''

            os.remove(tmp)

            ot = parse_date(txt)
            otl = parse_total(txt)
            ok = parse_kdv(txt)
            ov = parse_vergi_no(txt)

            rt = compare(gt.get('tarih'), ot, 'date')
            rtl = compare(gt.get('toplam'), otl, 'amt')
            rk = compare(gt.get('kdv'), ok, 'amt')
            rv = compare(gt.get('vergi_no'), ov, 'vn')

            sc = sum(1 for x in [rt, rtl, rk, rv] if x == 'OK')

            if rt == 'OK': stats[pn]['tarih'] += 1
            if rtl == 'OK': stats[pn]['toplam'] += 1
            if rk == 'OK': stats[pn]['kdv'] += 1
            if rv == 'OK': stats[pn]['vergi_no'] += 1
            stats[pn]['total'] += sc

            row[f'{pn}_tarih'] = rt
            row[f'{pn}_toplam'] = rtl
            row[f'{pn}_kdv'] = rk
            row[f'{pn}_vergi'] = rv
            row[f'{pn}_skor'] = f'{sc}/4'

        results.append(row)
        del img, proc
        gc.collect()

    # CSV kaydet
    cols = ['dosya', 'market', 'kategori', 'gt_tarih', 'gt_toplam', 'gt_kdv', 'gt_vergi_no']
    for pn in PREPS:
        cols += [f'{pn}_tarih', f'{pn}_toplam', f'{pn}_kdv', f'{pn}_vergi', f'{pn}_skor']

    csv_path = 'preprocessing_benchmark_v2.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)

    print('\n' + '='*70)
    print('PREPROCESSING KARSILASTIRMA TABLOSU')
    print('='*70)
    print(f"{'Preprocessing':<18} {'Tarih':<8} {'Toplam':<8} {'KDV':<8} {'VergiNo':<8} {'TOTAL':<8}")
    print('-'*70)

    for pn, s in sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True):
        print(f"{pn:<18} {s['tarih']:<8} {s['toplam']:<8} {s['kdv']:<8} {s['vergi_no']:<8} {s['total']:<8}")

    print('-'*70)

    best = max(stats.items(), key=lambda x: x[1]['total'])
    print(f"\nEN IYI: {best[0]} ({PREPS[best[0]][0]}) - Toplam: {best[1]['total']}")

    # Markdown rapor
    md = []
    md.append('# Preprocessing Benchmark Raporu')
    md.append(f'\n**Tarih:** {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    md.append(f'**OCR Motoru:** PaddleOCR')
    md.append(f'**Dosya Sayisi:** {len(fisler)}')
    md.append('')
    md.append('## Preprocessing Adimlari')
    md.append('')
    md.append('| Kod | Aciklama |')
    md.append('|-----|----------|')
    for pn, (pdesc, _) in PREPS.items():
        md.append(f'| {pn} | {pdesc} |')

    md.append('')
    md.append('## Sonuc Tablosu')
    md.append('')
    md.append('| Preprocessing | Tarih | Toplam | KDV | Vergi No | **TOTAL** |')
    md.append('|---------------|-------|--------|-----|----------|-----------|')
    for pn, s in sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True):
        md.append(f"| {pn} | {s['tarih']} | {s['toplam']} | {s['kdv']} | {s['vergi_no']} | **{s['total']}** |")

    md.append('')
    md.append(f'## En Iyi Kombinasyon: **{best[0]}** ({PREPS[best[0]][0]})')

    with open('PREPROCESSING_BENCHMARK_REPORT.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f'\nCSV: {csv_path}')
    print('Markdown: PREPROCESSING_BENCHMARK_REPORT.md')
    print('='*70)

if __name__ == '__main__':
    main()
