# -*- coding: utf-8 -*-
"""
Ablation Study & Ensemble Benchmark
====================================
Her preprocessing adimini AYRI AYRI test eder.
PaddleOCR + EasyOCR ile 10 farkli sonuc uretir.
Ensemble (Majority Voting) ile en iyi sonucu belirler.

Cikti: detailed_benchmark.csv
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

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)


# =============================================================================
# 1. IMAGE PROCESSOR - Her preprocessing adimini AYRI AYRI uygular
# =============================================================================

class ImageProcessor:
    """
    Resme tek tek islemleri uygulayip versiyonlarini dondurur.
    Her adim BAGIMSIZ - birlestirilmez!
    """

    METHODS = ['original', 'grayscale', 'binary_otsu', 'adaptive_thresh', 'denoise']

    def __init__(self, max_width=1200):
        self.max_width = max_width

    def load_and_resize(self, image_path):
        """Resmi yukle ve boyutlandir"""
        img = cv2.imread(image_path)
        if img is None:
            return None

        h, w = img.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            img = cv2.resize(img, (self.max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

        return img

    def process_original(self, img):
        """Hicbir islem yok - orijinal"""
        return img.copy()

    def process_grayscale(self, img):
        """Sadece gri tonlama"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def process_binary_otsu(self, img):
        """Otsu Thresholding"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def process_adaptive_thresh(self, img):
        """Adaptive Gaussian Thresholding"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

    def process_denoise(self, img):
        """Fast NL Means Denoising"""
        # Renkli resim icin fastNlMeansDenoisingColored kullan
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return denoised

    def get_all_versions(self, image_path):
        """
        Resmin TUM versiyonlarini dondur.
        Returns: dict{'original': img, 'grayscale': img, ...}
        """
        img = self.load_and_resize(image_path)
        if img is None:
            return None

        versions = {
            'original': self.process_original(img),
            'grayscale': self.process_grayscale(img),
            'binary_otsu': self.process_binary_otsu(img),
            'adaptive_thresh': self.process_adaptive_thresh(img),
            'denoise': self.process_denoise(img),
        }

        return versions


# =============================================================================
# 2. OCR ENGINE - PaddleOCR ve EasyOCR
# =============================================================================

class OCREngine:
    """
    Birden fazla OCR motorunu yonetir.
    Her motoru ayri ayri calistirir.
    """

    def __init__(self):
        self.engines = {}
        self._init_paddle()
        self._init_easyocr()

    def _init_paddle(self):
        """PaddleOCR'i baslat"""
        try:
            from paddleocr import PaddleOCR
            self.engines['paddle'] = PaddleOCR(
                lang='en',
                det_db_thresh=0.25,
                det_db_box_thresh=0.6,
                det_db_unclip_ratio=1.75,
                det_limit_side_len=960,
                use_angle_cls=True,
                use_gpu=False,
                show_log=False
            )
            print("  [OK] PaddleOCR yuklendi")
        except Exception as e:
            print(f"  [X] PaddleOCR yuklenemedi: {e}")

    def _init_easyocr(self):
        """EasyOCR'i baslat"""
        try:
            import easyocr
            self.engines['easy'] = easyocr.Reader(['en', 'tr'], gpu=False, verbose=False)
            print("  [OK] EasyOCR yuklendi")
        except Exception as e:
            print(f"  [X] EasyOCR yuklenemedi: {e}")
            print("      pip install easyocr")

    def read_paddle(self, image):
        """PaddleOCR ile oku"""
        if 'paddle' not in self.engines:
            return ""

        try:
            # Numpy array ise gecici dosyaya kaydet
            if isinstance(image, np.ndarray):
                temp_path = '_temp_paddle.png'
                cv2.imwrite(temp_path, image)
                result = self.engines['paddle'].ocr(temp_path, cls=True)
                os.remove(temp_path)
            else:
                result = self.engines['paddle'].ocr(image, cls=True)

            if result and result[0]:
                lines = [line[1][0] for line in result[0] if line[1][1] > 0.3]
                return '\n'.join(lines)
        except Exception as e:
            pass

        return ""

    def read_easy(self, image):
        """EasyOCR ile oku"""
        if 'easy' not in self.engines:
            return ""

        try:
            # Numpy array ise gecici dosyaya kaydet
            if isinstance(image, np.ndarray):
                temp_path = '_temp_easy.png'
                cv2.imwrite(temp_path, image)
                result = self.engines['easy'].readtext(temp_path)
                os.remove(temp_path)
            else:
                result = self.engines['easy'].readtext(image)

            if result:
                lines = [item[1] for item in result if item[2] > 0.3]
                return '\n'.join(lines)
        except Exception as e:
            pass

        return ""

    def read_all(self, image_versions):
        """
        Tum versiyonlari tum motorlarla oku.
        Returns: dict{'paddle_original': text, 'paddle_grayscale': text, 'easy_original': text, ...}
        """
        results = {}

        for version_name, img in image_versions.items():
            # PaddleOCR
            paddle_key = f"paddle_{version_name}"
            results[paddle_key] = self.read_paddle(img)

            # EasyOCR
            easy_key = f"easy_{version_name}"
            results[easy_key] = self.read_easy(img)

        return results


# =============================================================================
# 3. PARSER - Metinden deger cikarma
# =============================================================================

class Parser:
    """Metinden tarih, toplam, kdv, vergi no cikarir"""

    @staticmethod
    def parse_total(text):
        """Toplam tutari cikar"""
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

        # Pattern 1: TOPLAM/TOTAL ayni satirda
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

        # Pattern 2: TOPLAM sonraki satirda
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

        # Pattern 3: Para birimi ile
        currency_patterns = [
            r'€\s*(\d+[.,]\d{2})',
            r'(\d+[.,]\d{2})\s*€',
            r'(\d+[.,]\d{2})\s*EUR\b',
            r'(\d+[.,]\d{2})\s*TL\b',
        ]

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

    @staticmethod
    def parse_date(text):
        """Tarih cikar"""
        if not text:
            return None

        patterns = [
            r'TARIH\s*[:\s]*(\d{2})[./\-](\d{2})[./\-](\d{2,4})',
            r'DATE\s*[:\s]*(\d{2})[./\-](\d{2})[./\-](\d{2,4})',
            r'(\d{2})[./\-](\d{2})[./\-](20\d{2})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                g = match.groups()
                d, m, y = g[0], g[1], g[2]
                if len(y) == 2:
                    y = '20' + y
                try:
                    if 1 <= int(d) <= 31 and 1 <= int(m) <= 12:
                        return f"{d}/{m}/{y}"
                except:
                    pass

        return None

    @staticmethod
    def parse_kdv(text):
        """KDV cikar"""
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

        return None

    @staticmethod
    def parse_vergi_no(text):
        """Vergi numarasi cikar"""
        if not text:
            return None

        text_upper = text.upper()

        patterns = [
            r'V\.?\s*D\.?\s*(\d{10,11})',
            r'VKN[:\s]*(\d{10,11})',
            r'SIRET[:\s]*(\d{14})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text_upper)
            if match:
                return match.group(1)

        return None


# =============================================================================
# 4. ENSEMBLE VOTER - Majority Voting
# =============================================================================

class EnsembleVoter:
    """
    Tum sonuclari toplayip en cok tekrar eden degeri secer.
    Majority Voting mantigi.
    """

    @staticmethod
    def vote(values_dict, tolerance=0.5):
        """
        values_dict: {'paddle_original': 305.0, 'easy_grayscale': 305.0, ...}
        En cok tekrar eden degeri dondurur.
        """
        # None olanlari filtrele
        valid_values = {k: v for k, v in values_dict.items() if v is not None}

        if not valid_values:
            return None, []

        # Sayisal tolerans ile gruplama
        # Benzer degerleri ayni gruba koy (305.0 ve 305.00 gibi)
        groups = []

        for key, value in valid_values.items():
            found_group = False
            for group in groups:
                # Grubun ilk elemaniyla karsilastir
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

        # En buyuk grubu bul
        if not groups:
            return None, []

        largest_group = max(groups, key=len)
        winner_value = list(largest_group.values())[0]
        voters = list(largest_group.keys())

        return winner_value, voters


# =============================================================================
# 5. BENCHMARK RUNNER
# =============================================================================

class AblationBenchmark:
    """Ana benchmark sinifi"""

    def __init__(self, data_dir='data', gt_file='ground_truth.json'):
        self.data_dir = data_dir
        self.gt_file = gt_file

        # Ground truth yukle
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        self.fisler = gt_data.get('fisler', {})

        # Bilesenleri olustur
        print("\n" + "="*70)
        print("ABLATION STUDY & ENSEMBLE BENCHMARK")
        print("="*70)
        print("\nBilesenler yukleniyor...")

        self.processor = ImageProcessor()
        self.ocr = OCREngine()
        self.parser = Parser()
        self.voter = EnsembleVoter()

        # Sonuc kolonlari
        self.preprocessing_methods = ImageProcessor.METHODS
        self.ocr_engines = ['paddle', 'easy']

    def compare_values(self, gt_val, ocr_val, tolerance=0.01):
        """Degerleri karsilastir"""
        if gt_val is None or gt_val == '???' or gt_val == 'null' or gt_val == '':
            return None  # Skip

        if ocr_val is None:
            return False

        try:
            gt_f = float(gt_val)
            ocr_f = float(ocr_val)
            return abs(gt_f - ocr_f) <= tolerance * max(gt_f, 1)
        except:
            return str(gt_val) == str(ocr_val)

    def run(self):
        """Benchmark calistir"""
        print(f"\nToplam dosya: {len(self.fisler)}")
        print(f"Preprocessing: {self.preprocessing_methods}")
        print(f"OCR Engines: {self.ocr_engines}")
        print(f"Toplam kombinasyon: {len(self.preprocessing_methods) * len(self.ocr_engines)}")
        print("\n" + "="*70 + "\n")

        results = []

        # Istatistikler
        method_stats = {}
        for engine in self.ocr_engines:
            for prep in self.preprocessing_methods:
                key = f"{engine}_{prep}"
                method_stats[key] = {'correct': 0, 'total': 0}
        method_stats['ensemble'] = {'correct': 0, 'total': 0}

        # Her dosya icin
        for idx, (filename, gt) in enumerate(self.fisler.items(), 1):
            image_path = os.path.join(self.data_dir, filename)

            if not os.path.exists(image_path):
                print(f"[{idx}] {filename} - DOSYA YOK")
                continue

            print(f"[{idx}/{len(self.fisler)}] {filename}")

            # Ground truth
            gt_toplam = gt.get('toplam')
            gt_tarih = gt.get('tarih')
            gt_kdv = gt.get('kdv')
            gt_vergi_no = gt.get('vergi_no')

            # Resim versiyonlarini olustur
            versions = self.processor.get_all_versions(image_path)
            if versions is None:
                print(f"  ! Resim okunamadi")
                continue

            # Tum OCR sonuclarini al
            ocr_texts = self.ocr.read_all(versions)

            # Her kombinasyon icin toplam degerini parse et
            toplam_results = {}

            row = {
                'dosya': filename,
                'market': gt.get('market_adi', '-')[:25],
                'gt_toplam': gt_toplam,
                'gt_tarih': gt_tarih,
                'gt_kdv': gt_kdv,
            }

            best_method = None

            for engine in self.ocr_engines:
                for prep in self.preprocessing_methods:
                    key = f"{engine}_{prep}"
                    text = ocr_texts.get(key, "")

                    # Toplam parse et
                    parsed_toplam = self.parser.parse_total(text)
                    toplam_results[key] = parsed_toplam

                    # Kolon adi: Paddle_Original, Easy_Grayscale vb.
                    col_name = f"{engine.capitalize()}_{prep.replace('_', ' ').title().replace(' ', '_')}"
                    row[col_name] = parsed_toplam if parsed_toplam else "-"

                    # Dogruluk kontrolu
                    is_correct = self.compare_values(gt_toplam, parsed_toplam)
                    if is_correct is not None:
                        method_stats[key]['total'] += 1
                        if is_correct:
                            method_stats[key]['correct'] += 1
                            if best_method is None:
                                best_method = col_name

            # Ensemble voting
            ensemble_value, voters = self.voter.vote(toplam_results)
            row['ENSEMBLE_SONUC'] = ensemble_value if ensemble_value else "-"

            # Ensemble dogruluk
            ensemble_match = self.compare_values(gt_toplam, ensemble_value)
            if ensemble_match is None:
                row['ENSEMBLE_MATCH'] = "SKIP"
            else:
                row['ENSEMBLE_MATCH'] = "TRUE" if ensemble_match else "FALSE"
                method_stats['ensemble']['total'] += 1
                if ensemble_match:
                    method_stats['ensemble']['correct'] += 1

            # Best method
            row['BEST_METHOD'] = best_method if best_method else "-"

            results.append(row)

            # RAM temizligi
            del versions, ocr_texts
            gc.collect()

        # CSV kaydet
        self._save_csv(results)

        # Istatistikleri yazdir
        self._print_stats(method_stats)

        # Markdown rapor
        self._save_markdown(results, method_stats)

        return results, method_stats

    def _save_csv(self, results):
        """CSV dosyasi kaydet"""
        if not results:
            return

        # Sutun sirasi
        columns = ['dosya', 'market', 'gt_toplam', 'gt_tarih', 'gt_kdv']

        # OCR sonuclari
        for engine in self.ocr_engines:
            for prep in self.preprocessing_methods:
                col_name = f"{engine.capitalize()}_{prep.replace('_', ' ').title().replace(' ', '_')}"
                columns.append(col_name)

        columns.extend(['ENSEMBLE_SONUC', 'ENSEMBLE_MATCH', 'BEST_METHOD'])

        csv_path = 'detailed_benchmark.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nCSV kaydedildi: {csv_path}")

    def _print_stats(self, stats):
        """Istatistikleri yazdir"""
        print("\n" + "="*70)
        print("ABLATION STUDY SONUCLARI")
        print("="*70)
        print(f"\n{'Yontem':<25} {'Dogru':<8} {'Toplam':<8} {'Basari':<10}")
        print("-"*55)

        # Sirala
        sorted_stats = sorted(
            [(k, v) for k, v in stats.items() if k != 'ensemble'],
            key=lambda x: x[1]['correct'],
            reverse=True
        )

        for key, s in sorted_stats:
            total = s['total']
            correct = s['correct']
            acc = f"%{(correct/total)*100:.1f}" if total > 0 else "N/A"
            print(f"{key:<25} {correct:<8} {total:<8} {acc:<10}")

        print("-"*55)

        # Ensemble
        ens = stats['ensemble']
        ens_acc = f"%{(ens['correct']/ens['total'])*100:.1f}" if ens['total'] > 0 else "N/A"
        print(f"{'ENSEMBLE':<25} {ens['correct']:<8} {ens['total']:<8} {ens_acc:<10}")

        print("="*70)

        # En iyi yontem
        if sorted_stats:
            best = sorted_stats[0]
            print(f"\nEN IYI TEK YONTEM: {best[0]} ({best[1]['correct']}/{best[1]['total']})")

    def _save_markdown(self, results, stats):
        """Markdown rapor kaydet"""
        md = []
        md.append("# Ablation Study & Ensemble Benchmark Raporu")
        md.append(f"\n**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        md.append(f"**Toplam Dosya:** {len(self.fisler)}")
        md.append("")

        md.append("## Preprocessing Adimlari")
        md.append("")
        md.append("| Adim | Aciklama |")
        md.append("|------|----------|")
        md.append("| original | Hicbir islem yok |")
        md.append("| grayscale | Sadece gri tonlama |")
        md.append("| binary_otsu | Otsu Thresholding |")
        md.append("| adaptive_thresh | Adaptive Gaussian Thresholding |")
        md.append("| denoise | Fast NL Means Denoising |")
        md.append("")

        md.append("## Yontem Bazli Sonuclar (Toplam Alani)")
        md.append("")
        md.append("| Yontem | Dogru | Toplam | Basari |")
        md.append("|--------|-------|--------|--------|")

        sorted_stats = sorted(
            [(k, v) for k, v in stats.items()],
            key=lambda x: x[1]['correct'],
            reverse=True
        )

        for key, s in sorted_stats:
            total = s['total']
            correct = s['correct']
            acc = f"%{(correct/total)*100:.1f}" if total > 0 else "N/A"
            marker = "**" if key == 'ensemble' else ""
            md.append(f"| {marker}{key}{marker} | {correct} | {total} | {acc} |")

        md.append("")

        # Preprocessing karsilastirma
        md.append("## Preprocessing Etkisi (PaddleOCR)")
        md.append("")
        md.append("| Preprocessing | Dogru | Fark (vs Original) |")
        md.append("|---------------|-------|---------------------|")

        paddle_original = stats.get('paddle_original', {}).get('correct', 0)

        for prep in self.preprocessing_methods:
            key = f"paddle_{prep}"
            s = stats.get(key, {})
            correct = s.get('correct', 0)
            diff = correct - paddle_original
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            md.append(f"| {prep} | {correct} | {diff_str} |")

        md.append("")

        # EasyOCR karsilastirma
        md.append("## Preprocessing Etkisi (EasyOCR)")
        md.append("")
        md.append("| Preprocessing | Dogru | Fark (vs Original) |")
        md.append("|---------------|-------|---------------------|")

        easy_original = stats.get('easy_original', {}).get('correct', 0)

        for prep in self.preprocessing_methods:
            key = f"easy_{prep}"
            s = stats.get(key, {})
            correct = s.get('correct', 0)
            diff = correct - easy_original
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            md.append(f"| {prep} | {correct} | {diff_str} |")

        md.append("")

        # Ensemble analizi
        md.append("## Ensemble Voting Analizi")
        md.append("")
        ens = stats['ensemble']
        if ens['total'] > 0:
            ens_acc = (ens['correct']/ens['total'])*100
            md.append(f"- **Ensemble Basari:** %{ens_acc:.1f}")

            # En iyi tek yontem ile karsilastir
            best_single = max([(k, v['correct']) for k, v in stats.items() if k != 'ensemble'], key=lambda x: x[1])
            md.append(f"- **En Iyi Tek Yontem:** {best_single[0]} ({best_single[1]} dogru)")

            if ens['correct'] > best_single[1]:
                md.append(f"- **Ensemble Kazanci:** +{ens['correct'] - best_single[1]} ekstra dogru sonuc")
            elif ens['correct'] < best_single[1]:
                md.append(f"- **Ensemble Kaybi:** {ens['correct'] - best_single[1]} (tek yontem daha iyi)")
            else:
                md.append("- **Ensemble:** Tek yontemle ayni performans")

        md_path = 'ABLATION_BENCHMARK_REPORT.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md))

        print(f"Markdown kaydedildi: {md_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    benchmark = AblationBenchmark()
    results, stats = benchmark.run()
