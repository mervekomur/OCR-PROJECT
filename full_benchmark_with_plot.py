# -*- coding: utf-8 -*-
"""
Full Benchmark with Visual Report
==================================
Ablation Study + Ensemble + PNG Tablo Ciktisi

Ciktilar:
- detailed_benchmark.csv (Ham veri)
- benchmark_report.png (Gorsel tablo - sunum icin)
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
from collections import Counter

import matplotlib
matplotlib.use('Agg')  # GUI olmadan calistir
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)


# =============================================================================
# 1. IMAGE PROCESSOR
# =============================================================================

class ImageProcessor:
    """5 farkli preprocessing versiyonu uretir"""

    METHODS = ['original', 'gray', 'binary', 'denoise', 'morph']

    def __init__(self, max_width=1200):
        self.max_width = max_width

    def load_and_resize(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            img = cv2.resize(img, (self.max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    def process_original(self, img):
        """Orijinal - islem yok"""
        return img.copy()

    def process_gray(self, img):
        """Grayscale"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def process_binary(self, img):
        """Otsu Binary Threshold"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def process_denoise(self, img):
        """Fast NL Means Denoising"""
        try:
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            return denoised
        except:
            return img.copy()

    def process_morph(self, img):
        """Morphological Operations (Opening + Closing)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        return cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)

    def get_all_versions(self, image_path):
        img = self.load_and_resize(image_path)
        if img is None:
            return None

        return {
            'original': self.process_original(img),
            'gray': self.process_gray(img),
            'binary': self.process_binary(img),
            'denoise': self.process_denoise(img),
            'morph': self.process_morph(img),
        }


# =============================================================================
# 2. OCR ENGINE
# =============================================================================

class OCREngine:
    """PaddleOCR ve EasyOCR"""

    def __init__(self):
        self.engines = {}
        self._init_paddle()
        self._init_easyocr()

    def _init_paddle(self):
        try:
            from paddleocr import PaddleOCR
            self.engines['Paddle'] = PaddleOCR(
                lang='en',
                det_db_thresh=0.25,
                det_db_box_thresh=0.6,
                det_db_unclip_ratio=1.75,
                det_limit_side_len=960,
                use_angle_cls=True,
                use_gpu=False,
                show_log=False
            )
            print("  [OK] PaddleOCR")
        except Exception as e:
            print(f"  [X] PaddleOCR: {e}")

    def _init_easyocr(self):
        try:
            import easyocr
            self.engines['Easy'] = easyocr.Reader(['en', 'tr'], gpu=False, verbose=False)
            print("  [OK] EasyOCR")
        except Exception as e:
            print(f"  [X] EasyOCR: {e}")

    def read_paddle(self, image):
        if 'Paddle' not in self.engines:
            return ""
        try:
            temp_path = '_temp_paddle.png'
            cv2.imwrite(temp_path, image)
            result = self.engines['Paddle'].ocr(temp_path, cls=True)
            os.remove(temp_path)
            if result and result[0]:
                return '\n'.join([line[1][0] for line in result[0] if line[1][1] > 0.3])
        except:
            pass
        return ""

    def read_easy(self, image):
        if 'Easy' not in self.engines:
            return ""
        try:
            temp_path = '_temp_easy.png'
            cv2.imwrite(temp_path, image)
            result = self.engines['Easy'].readtext(temp_path)
            os.remove(temp_path)
            if result:
                return '\n'.join([item[1] for item in result if item[2] > 0.3])
        except:
            pass
        return ""

    def read_all(self, image_versions):
        results = {}
        for version_name, img in image_versions.items():
            results[f"Paddle_{version_name.capitalize()}"] = self.read_paddle(img)
            results[f"Easy_{version_name.capitalize()}"] = self.read_easy(img)
        return results


# =============================================================================
# 3. PARSER
# =============================================================================

class Parser:
    @staticmethod
    def parse_total(text):
        if not text:
            return None

        text_upper = text.upper()
        lines = text_upper.split('\n')

        def clean_amount(val_str):
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
            r'TUTAR[:\s]*[\*\+]?(\d+[.,]?\d{0,2})',
        ]

        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    try:
                        amount = clean_amount(match.group(1))
                        if 1 <= amount <= 50000:
                            return amount
                    except:
                        pass

        for i, line in enumerate(lines):
            if line.strip() in ['TOPLAM', 'TOTAL', 'TUTAR']:
                if i + 1 < len(lines):
                    match = re.search(r'[\*\+]?(\d{1,6}[.,]?\d{0,2})', lines[i + 1])
                    if match:
                        try:
                            amount = clean_amount(match.group(1))
                            if 1 <= amount <= 50000:
                                return amount
                        except:
                            pass

        currency_patterns = [r'€\s*(\d+[.,]\d{2})', r'(\d+[.,]\d{2})\s*€',
                            r'(\d+[.,]\d{2})\s*EUR\b', r'(\d+[.,]\d{2})\s*TL\b']
        found = []
        for line in lines:
            for pattern in currency_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        amount = float(match.group(1).replace(',', '.'))
                        if 1 <= amount <= 50000:
                            found.append(amount)
                    except:
                        pass
        if found:
            return max(found)

        return None


# =============================================================================
# 4. ENSEMBLE VOTER
# =============================================================================

class EnsembleVoter:
    @staticmethod
    def vote(values_dict, tolerance=0.5):
        valid_values = {k: v for k, v in values_dict.items() if v is not None}
        if not valid_values:
            return None, []

        groups = []
        for key, value in valid_values.items():
            found_group = False
            for group in groups:
                ref_value = list(group.values())[0]
                try:
                    if abs(float(value) - float(ref_value)) <= tolerance:
                        group[key] = value
                        found_group = True
                        break
                except:
                    if str(value) == str(ref_value):
                        group[key] = value
                        found_group = True
                        break
            if not found_group:
                groups.append({key: value})

        if not groups:
            return None, []

        largest_group = max(groups, key=len)
        winner_value = list(largest_group.values())[0]
        voters = list(largest_group.keys())
        return winner_value, voters


# =============================================================================
# 5. VISUALIZER - PNG TABLO OLUSTURMA
# =============================================================================

def visualize_results(csv_file, output_png='benchmark_report.png'):
    """
    CSV dosyasini okuyup gorsel tablo olusturur.
    Yesil = Dogru, Kirmizi = Yanlis
    """
    print("\n" + "="*70)
    print("GORSEL RAPOR OLUSTURULUYOR...")
    print("="*70)

    # CSV oku
    df = pd.read_csv(csv_file)

    # Yontem sutunlarini bul (Paddle_*, Easy_*, ENSEMBLE)
    method_cols = [col for col in df.columns if col.startswith('Paddle_') or col.startswith('Easy_')]
    method_cols.append('ENSEMBLE_SONUC')

    # Match sutunlarini olustur
    match_data = []

    for idx, row in df.iterrows():
        gt = row['gt_toplam']
        row_matches = {'dosya': row['dosya'][:20]}  # Dosya adi kisalt

        for col in method_cols:
            if col == 'ENSEMBLE_SONUC':
                display_col = 'ENSEMBLE'
            else:
                display_col = col

            ocr_val = row[col] if col in row else None

            # Karsilastir
            if pd.isna(gt) or gt == '' or gt == '???':
                row_matches[display_col] = 'skip'
            elif pd.isna(ocr_val) or ocr_val == '-' or ocr_val == '':
                row_matches[display_col] = 'fail'
            else:
                try:
                    gt_f = float(gt)
                    ocr_f = float(ocr_val)
                    if abs(gt_f - ocr_f) <= 0.01 * max(gt_f, 1):
                        row_matches[display_col] = 'ok'
                    else:
                        row_matches[display_col] = 'fail'
                except:
                    row_matches[display_col] = 'fail'

        # Best method
        row_matches['BEST'] = row['BEST_METHOD'] if 'BEST_METHOD' in row and pd.notna(row['BEST_METHOD']) else '-'

        match_data.append(row_matches)

    # DataFrame olustur
    match_df = pd.DataFrame(match_data)

    # Gorsel olustur
    fig, ax = plt.subplots(figsize=(20, len(match_df) * 0.5 + 2))
    ax.axis('off')

    # Renk paleti
    colors = {
        'ok': '#2ecc71',      # Yesil
        'fail': '#e74c3c',    # Kirmizi
        'skip': '#95a5a6',    # Gri
    }

    # Tablo verileri
    columns = list(match_df.columns)
    cell_text = []
    cell_colors = []

    for idx, row in match_df.iterrows():
        row_text = []
        row_colors = []
        for col in columns:
            val = row[col]
            if col == 'dosya' or col == 'BEST':
                row_text.append(str(val)[:15])
                row_colors.append('#ecf0f1')  # Acik gri
            elif val == 'ok':
                row_text.append('✓')
                row_colors.append(colors['ok'])
            elif val == 'fail':
                row_text.append('✗')
                row_colors.append(colors['fail'])
            else:
                row_text.append('-')
                row_colors.append(colors['skip'])

        cell_text.append(row_text)
        cell_colors.append(row_colors)

    # Tablo olustur
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellColours=cell_colors,
        colColours=['#3498db'] * len(columns),
        cellLoc='center',
        loc='center'
    )

    # Tablo stili
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Baslik renklerini ayarla
    for i, col in enumerate(columns):
        cell = table[0, i]
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_facecolor('#2c3e50')

    # Baslik
    plt.title('OCR Ablation Study & Ensemble Results\n(Toplam Alani Tespiti)',
              fontsize=16, fontweight='bold', pad=20)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['ok'], label='Dogru (✓)'),
        mpatches.Patch(facecolor=colors['fail'], label='Yanlis (✗)'),
        mpatches.Patch(facecolor=colors['skip'], label='Atlandı (-)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1.1))

    # Kaydet
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"PNG kaydedildi: {output_png}")

    # Istatistik tablosu da olustur
    create_summary_chart(match_df, columns)


def create_summary_chart(match_df, columns):
    """Ozet bar chart olustur"""

    method_cols = [c for c in columns if c not in ['dosya', 'BEST']]

    # Her yontem icin basari sayisi
    stats = {}
    for col in method_cols:
        ok_count = (match_df[col] == 'ok').sum()
        total = ((match_df[col] == 'ok') | (match_df[col] == 'fail')).sum()
        stats[col] = {'ok': ok_count, 'total': total}

    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 6))

    methods = list(stats.keys())
    ok_counts = [stats[m]['ok'] for m in methods]
    totals = [stats[m]['total'] for m in methods]
    percentages = [100 * ok / total if total > 0 else 0 for ok, total in zip(ok_counts, totals)]

    # Renkler - en yuksek yesil
    max_pct = max(percentages) if percentages else 0
    bar_colors = ['#2ecc71' if p == max_pct and p > 0 else '#3498db' for p in percentages]

    bars = ax.bar(methods, ok_counts, color=bar_colors, edgecolor='#2c3e50', linewidth=1.5)

    # Degerler
    for bar, ok, total, pct in zip(bars, ok_counts, totals, percentages):
        height = bar.get_height()
        ax.annotate(f'{ok}/{total}\n(%{pct:.0f})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Yontem', fontsize=12)
    ax.set_ylabel('Dogru Sonuc Sayisi', fontsize=12)
    ax.set_title('OCR Yontem Karsilastirmasi - Toplam Alani Basarisi', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(ok_counts) * 1.3 if ok_counts else 10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('benchmark_summary_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print("PNG kaydedildi: benchmark_summary_chart.png")


# =============================================================================
# 6. BENCHMARK RUNNER
# =============================================================================

class FullBenchmark:
    """Ana benchmark sinifi"""

    def __init__(self, data_dir='data', gt_file='ground_truth.json'):
        self.data_dir = data_dir
        self.gt_file = gt_file

        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        self.fisler = gt_data.get('fisler', {})

        print("\n" + "="*70)
        print("FULL BENCHMARK WITH VISUAL REPORT")
        print("="*70)
        print("\nBilesenler yukleniyor...")

        self.processor = ImageProcessor()
        self.ocr = OCREngine()
        self.parser = Parser()
        self.voter = EnsembleVoter()

    def compare_values(self, gt_val, ocr_val, tolerance=0.01):
        if gt_val is None or gt_val == '???' or gt_val == 'null' or gt_val == '':
            return None
        if ocr_val is None:
            return False
        try:
            gt_f = float(gt_val)
            ocr_f = float(ocr_val)
            return abs(gt_f - ocr_f) <= tolerance * max(gt_f, 1)
        except:
            return str(gt_val) == str(ocr_val)

    def run(self):
        print(f"\nToplam dosya: {len(self.fisler)}")
        print(f"Preprocessing: {ImageProcessor.METHODS}")
        print("\n" + "="*70 + "\n")

        results = []
        method_stats = {}

        for idx, (filename, gt) in enumerate(self.fisler.items(), 1):
            image_path = os.path.join(self.data_dir, filename)

            if not os.path.exists(image_path):
                print(f"[{idx}] {filename} - DOSYA YOK")
                continue

            print(f"[{idx}/{len(self.fisler)}] {filename}")

            gt_toplam = gt.get('toplam')

            versions = self.processor.get_all_versions(image_path)
            if versions is None:
                continue

            ocr_texts = self.ocr.read_all(versions)

            toplam_results = {}
            row = {
                'dosya': filename,
                'market': gt.get('market_adi', '-')[:25],
                'gt_toplam': gt_toplam,
            }

            best_method = None

            for method_name, text in ocr_texts.items():
                parsed = self.parser.parse_total(text)
                toplam_results[method_name] = parsed
                row[method_name] = parsed if parsed else "-"

                # Stats
                if method_name not in method_stats:
                    method_stats[method_name] = {'correct': 0, 'total': 0}

                is_correct = self.compare_values(gt_toplam, parsed)
                if is_correct is not None:
                    method_stats[method_name]['total'] += 1
                    if is_correct:
                        method_stats[method_name]['correct'] += 1
                        if best_method is None:
                            best_method = method_name

            # Ensemble
            ensemble_value, voters = self.voter.vote(toplam_results)
            row['ENSEMBLE_SONUC'] = ensemble_value if ensemble_value else "-"

            ensemble_match = self.compare_values(gt_toplam, ensemble_value)
            if ensemble_match is None:
                row['ENSEMBLE_MATCH'] = "SKIP"
            else:
                row['ENSEMBLE_MATCH'] = "TRUE" if ensemble_match else "FALSE"

                if 'ENSEMBLE' not in method_stats:
                    method_stats['ENSEMBLE'] = {'correct': 0, 'total': 0}
                method_stats['ENSEMBLE']['total'] += 1
                if ensemble_match:
                    method_stats['ENSEMBLE']['correct'] += 1

            row['BEST_METHOD'] = best_method if best_method else "-"
            results.append(row)

            del versions, ocr_texts
            gc.collect()

        # CSV kaydet
        csv_path = 'ablation_results.csv'
        self._save_csv(results, csv_path)

        # Istatistikleri yazdir
        self._print_stats(method_stats)

        # GORSEL RAPOR OLUSTUR
        visualize_results(csv_path, 'benchmark_report.png')

        return results, method_stats

    def _save_csv(self, results, csv_path):
        if not results:
            return

        columns = ['dosya', 'market', 'gt_toplam']

        # OCR method columns
        method_cols = set()
        for row in results:
            for key in row.keys():
                if key.startswith('Paddle_') or key.startswith('Easy_'):
                    method_cols.add(key)

        method_cols = sorted(list(method_cols))
        columns.extend(method_cols)
        columns.extend(['ENSEMBLE_SONUC', 'ENSEMBLE_MATCH', 'BEST_METHOD'])

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nCSV kaydedildi: {csv_path}")

    def _print_stats(self, stats):
        print("\n" + "="*70)
        print("SONUCLAR")
        print("="*70)
        print(f"\n{'Yontem':<25} {'Dogru':<8} {'Toplam':<8} {'Basari':<10}")
        print("-"*55)

        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['correct'], reverse=True)

        for key, s in sorted_stats:
            total = s['total']
            correct = s['correct']
            acc = f"%{(correct/total)*100:.1f}" if total > 0 else "N/A"
            print(f"{key:<25} {correct:<8} {total:<8} {acc:<10}")

        print("="*70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    benchmark = FullBenchmark()
    results, stats = benchmark.run()

    print("\n" + "="*70)
    print("TAMAMLANDI!")
    print("="*70)
    print("\nOlusan dosyalar:")
    print("  1. detailed_benchmark.csv    - Ham veri")
    print("  2. benchmark_report.png      - Gorsel tablo (sunum icin)")
    print("  3. benchmark_summary_chart.png - Ozet bar chart")
    print("="*70)
