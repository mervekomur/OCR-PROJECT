# -*- coding: utf-8 -*-
"""
OCR Sonuç Analiz Script'i
=========================
Evaluation sonuçlarını analiz eder ve detaylı rapor üretir.
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def load_results(filename):
    """JSON sonuç dosyasını yükle"""
    if not os.path.exists(filename):
        print(f"HATA: {filename} bulunamadi!")
        return None

    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_ground_truth(filename='ground_truth.json'):
    """Ground truth dosyasını yükle"""
    if not os.path.exists(filename):
        print(f"HATA: {filename} bulunamadi!")
        return None

    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_results(results, gt_data):
    """Sonuçları detaylı analiz et"""

    if not results or not gt_data:
        return

    print("\n" + "="*60)
    print("DETAYLI ANALIZ RAPORU")
    print("="*60)

    stats = results.get('stats', {})
    details = results.get('results', {})

    # Genel istatistikler
    print("\n## GENEL ISTATISTIKLER ##\n")

    for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
        s = stats.get(field, {})
        correct = s.get('correct', 0)
        wrong = s.get('wrong', 0)
        skipped = s.get('skipped', 0)
        total = correct + wrong

        if total > 0:
            accuracy = (correct / total) * 100
            print(f"{field:12}: {correct:3}/{total:3} = %{accuracy:5.1f}  (atlandı: {skipped})")
        else:
            print(f"{field:12}: N/A")

    # Başarısız olanlar
    print("\n" + "-"*60)
    print("## BASARISIZ SONUCLAR ##\n")

    failed_files = {
        'tarih': [],
        'toplam': [],
        'kdv': [],
        'vergi_no': []
    }

    for filename, result in details.items():
        for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
            if field in result:
                r = result[field]
                if r.get('match') == False:
                    failed_files[field].append({
                        'file': filename,
                        'gt': r.get('gt'),
                        'ocr': r.get('ocr')
                    })

    for field, failures in failed_files.items():
        if failures:
            print(f"\n### {field.upper()} HATALARI ({len(failures)} adet) ###")
            for f in failures:
                print(f"  {f['file']}")
                print(f"    GT:  {f['gt']}")
                print(f"    OCR: {f['ocr']}")

    # Kategori bazlı analiz
    print("\n" + "-"*60)
    print("## KATEGORI BAZLI ANALIZ ##\n")

    categories = {
        'TR_Fis': [],      # Türk fişleri
        'FR_Fis': [],      # Fransız fişleri
        'Makbuz': [],      # Makbuzlar
        'Diger': []        # Diğerleri
    }

    fisler = gt_data.get('fisler', {})

    for filename in details.keys():
        gt = fisler.get(filename, {})
        currency = gt.get('para_birimi', '')

        if filename.startswith('makbuz'):
            categories['Makbuz'].append(filename)
        elif currency == 'TL':
            categories['TR_Fis'].append(filename)
        elif currency == 'EUR':
            categories['FR_Fis'].append(filename)
        else:
            categories['Diger'].append(filename)

    for cat_name, files in categories.items():
        if not files:
            continue

        print(f"\n### {cat_name} ({len(files)} dosya) ###")

        cat_stats = {f: {'correct': 0, 'wrong': 0} for f in ['tarih', 'toplam', 'kdv', 'vergi_no']}

        for filename in files:
            result = details.get(filename, {})
            for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
                if field in result:
                    match = result[field].get('match')
                    if match == True:
                        cat_stats[field]['correct'] += 1
                    elif match == False:
                        cat_stats[field]['wrong'] += 1

        for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
            c = cat_stats[field]['correct']
            w = cat_stats[field]['wrong']
            total = c + w
            if total > 0:
                acc = (c / total) * 100
                print(f"  {field:10}: {c}/{total} = %{acc:.0f}")

    # En sorunlu dosyalar
    print("\n" + "-"*60)
    print("## EN SORUNLU DOSYALAR ##\n")

    file_errors = {}
    for filename, result in details.items():
        errors = 0
        for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
            if field in result and result[field].get('match') == False:
                errors += 1
        if errors > 0:
            file_errors[filename] = errors

    sorted_errors = sorted(file_errors.items(), key=lambda x: x[1], reverse=True)

    for filename, err_count in sorted_errors[:10]:
        gt = fisler.get(filename, {})
        market = gt.get('market_adi', 'Bilinmiyor')[:30]
        print(f"  {filename:20} - {err_count} hata - {market}")

    print("\n" + "="*60)


def compare_engines(paddle_file, google_file):
    """İki OCR motorunu karşılaştır"""

    paddle = load_results(paddle_file)
    google = load_results(google_file)

    if not paddle:
        print("PaddleOCR sonuçları yüklenemedi!")
        return

    if not google:
        print("Google OCR sonuçları yüklenemedi!")
        print("Sadece PaddleOCR analizi yapılacak...")
        return paddle

    print("\n" + "="*60)
    print("OCR MOTOR KARSILASTIRMASI")
    print("="*60)

    paddle_stats = paddle.get('stats', {})
    google_stats = google.get('stats', {})

    print(f"\n{'Alan':<12} {'PaddleOCR':<15} {'Google OCR':<15} {'Kazanan':<10}")
    print("-" * 52)

    for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
        p = paddle_stats.get(field, {})
        g = google_stats.get(field, {})

        p_total = p.get('correct', 0) + p.get('wrong', 0)
        g_total = g.get('correct', 0) + g.get('wrong', 0)

        p_acc = (p.get('correct', 0) / p_total * 100) if p_total > 0 else 0
        g_acc = (g.get('correct', 0) / g_total * 100) if g_total > 0 else 0

        winner = "Paddle" if p_acc > g_acc else ("Google" if g_acc > p_acc else "Eşit")

        print(f"{field:<12} %{p_acc:<13.1f} %{g_acc:<13.1f} {winner:<10}")

    return paddle


def print_benchmark_summary():
    """Tüm benchmark sonuçlarını özetleyen tablo"""

    print("\n" + "="*70)
    print("BENCHMARK PERFORMANS OZETI")
    print("="*70)

    # Mevcut sonuçları yükle
    benchmarks = []

    # PaddleOCR sonuçları
    if os.path.exists('evaluation_results_paddle.json'):
        paddle = load_results('evaluation_results_paddle.json')
        if paddle:
            stats = paddle.get('stats', {})
            benchmarks.append({
                'engine': 'PaddleOCR',
                'tarih': stats.get('tarih', {}),
                'toplam': stats.get('toplam', {}),
                'kdv': stats.get('kdv', {}),
                'vergi_no': stats.get('vergi_no', {}),
                'preprocessing': 'Yok',
                'notes': 'Varsayilan parametreler'
            })

    # Google OCR sonuçları
    if os.path.exists('evaluation_results_google.json'):
        google = load_results('evaluation_results_google.json')
        if google:
            stats = google.get('stats', {})
            benchmarks.append({
                'engine': 'Google Vision',
                'tarih': stats.get('tarih', {}),
                'toplam': stats.get('toplam', {}),
                'kdv': stats.get('kdv', {}),
                'vergi_no': stats.get('vergi_no', {}),
                'preprocessing': 'Yok',
                'notes': 'Cloud API'
            })

    # EasyOCR sonuçları (varsa)
    if os.path.exists('evaluation_results_easyocr.json'):
        easy = load_results('evaluation_results_easyocr.json')
        if easy:
            stats = easy.get('stats', {})
            benchmarks.append({
                'engine': 'EasyOCR',
                'tarih': stats.get('tarih', {}),
                'toplam': stats.get('toplam', {}),
                'kdv': stats.get('kdv', {}),
                'vergi_no': stats.get('vergi_no', {}),
                'preprocessing': 'Yok',
                'notes': 'PyTorch tabanli'
            })

    # Tesseract sonuçları (varsa)
    if os.path.exists('evaluation_results_tesseract.json'):
        tess = load_results('evaluation_results_tesseract.json')
        if tess:
            stats = tess.get('stats', {})
            benchmarks.append({
                'engine': 'Tesseract',
                'tarih': stats.get('tarih', {}),
                'toplam': stats.get('toplam', {}),
                'kdv': stats.get('kdv', {}),
                'vergi_no': stats.get('vergi_no', {}),
                'preprocessing': 'Yok',
                'notes': 'Acik kaynak'
            })

    if not benchmarks:
        print("\nHenuz benchmark sonucu bulunamadi!")
        print("Once evaluation.py calistirin.")
        return

    # Tablo başlığı
    print("\n" + "-"*70)
    print(f"{'OCR Engine':<15} {'Tarih':<10} {'Toplam':<10} {'KDV':<10} {'Vergi No':<10} {'Genel':<10}")
    print("-"*70)

    def calc_acc(stat):
        c = stat.get('correct', 0)
        w = stat.get('wrong', 0)
        total = c + w
        if total > 0:
            return f"{c}/{total} %{(c/total)*100:.0f}"
        return "N/A"

    def calc_overall(stats_dict):
        total_c = 0
        total_w = 0
        for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
            s = stats_dict.get(field, {})
            total_c += s.get('correct', 0)
            total_w += s.get('wrong', 0)
        total = total_c + total_w
        if total > 0:
            return f"%{(total_c/total)*100:.1f}"
        return "N/A"

    for b in benchmarks:
        tarih = calc_acc(b['tarih'])
        toplam = calc_acc(b['toplam'])
        kdv = calc_acc(b['kdv'])
        vergi = calc_acc(b['vergi_no'])
        overall = calc_overall({'tarih': b['tarih'], 'toplam': b['toplam'],
                                'kdv': b['kdv'], 'vergi_no': b['vergi_no']})

        print(f"{b['engine']:<15} {tarih:<10} {toplam:<10} {kdv:<10} {vergi:<10} {overall:<10}")

    print("-"*70)

    # Kategori bazlı özet
    print("\n" + "-"*70)
    print("KATEGORI BAZLI PERFORMANS (PaddleOCR)")
    print("-"*70)
    print(f"{'Kategori':<15} {'Tarih':<10} {'Toplam':<10} {'KDV':<10} {'Vergi No':<10} {'Toplam Fis':<10}")
    print("-"*70)

    # Ground truth'dan kategori bilgisi al
    gt_data = load_ground_truth()
    if gt_data and benchmarks:
        paddle = load_results('evaluation_results_paddle.json')
        if paddle:
            details = paddle.get('results', {})
            fisler = gt_data.get('fisler', {})

            categories = {
                'TR Fisleri': {'files': [], 'currency': 'TL'},
                'FR Fisleri': {'files': [], 'currency': 'EUR'},
                'Makbuzlar': {'files': [], 'prefix': 'makbuz'},
                'Diger': {'files': []}
            }

            for filename in details.keys():
                gt = fisler.get(filename, {})
                currency = gt.get('para_birimi', '')

                if filename.startswith('makbuz'):
                    categories['Makbuzlar']['files'].append(filename)
                elif currency == 'TL':
                    categories['TR Fisleri']['files'].append(filename)
                elif currency == 'EUR':
                    categories['FR Fisleri']['files'].append(filename)
                else:
                    categories['Diger']['files'].append(filename)

            for cat_name, cat_data in categories.items():
                files = cat_data['files']
                if not files:
                    continue

                cat_stats = {f: {'c': 0, 'w': 0} for f in ['tarih', 'toplam', 'kdv', 'vergi_no']}

                for filename in files:
                    result = details.get(filename, {})
                    for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
                        if field in result:
                            match = result[field].get('match')
                            if match == True:
                                cat_stats[field]['c'] += 1
                            elif match == False:
                                cat_stats[field]['w'] += 1

                def fmt(s):
                    t = s['c'] + s['w']
                    if t > 0:
                        return f"{s['c']}/{t} %{(s['c']/t)*100:.0f}"
                    return "N/A"

                print(f"{cat_name:<15} {fmt(cat_stats['tarih']):<10} {fmt(cat_stats['toplam']):<10} "
                      f"{fmt(cat_stats['kdv']):<10} {fmt(cat_stats['vergi_no']):<10} {len(files):<10}")

    print("-"*70)

    # Öneriler
    print("\n" + "-"*70)
    print("IYILESTIRME ONERILERI")
    print("-"*70)

    if benchmarks:
        paddle_stats = benchmarks[0] if benchmarks[0]['engine'] == 'PaddleOCR' else None

        if paddle_stats:
            kdv_acc = paddle_stats['kdv'].get('correct', 0) / max(1, paddle_stats['kdv'].get('correct', 0) + paddle_stats['kdv'].get('wrong', 0))

            if kdv_acc < 0.5:
                print("! KDV tespiti dusuk - Parser pattern'leri genisletilmeli")

            vergi_acc = paddle_stats['vergi_no'].get('correct', 0) / max(1, paddle_stats['vergi_no'].get('correct', 0) + paddle_stats['vergi_no'].get('wrong', 0))

            if vergi_acc < 0.5:
                print("! Vergi No tespiti dusuk - V.D. pattern'leri iyilestirilmeli")

            print("\nGenel Oneriler:")
            print("  1. Makbuzlar icin preprocessing (golge kaldirma) eklenmeli")
            print("  2. Fransiz fisleri icin TVA pattern'leri genisletilmeli")
            print("  3. Farkli OCR motorlari (Google, EasyOCR) denenebilir")
            print("  4. Ensemble yaklasim (birden fazla OCR) dusunulebilir")

    print("\n" + "="*70)


if __name__ == '__main__':
    print("OCR Sonuç Analizi")
    print("-" * 40)

    # Ground truth yükle
    gt_data = load_ground_truth()

    # Sonuçları yükle ve analiz et
    paddle_results = load_results('evaluation_results_paddle.json')

    if paddle_results:
        analyze_results(paddle_results, gt_data)

    # Benchmark özet tablosu
    print_benchmark_summary()

    # Google sonuçları varsa karşılaştır
    if os.path.exists('evaluation_results_google.json'):
        compare_engines('evaluation_results_paddle.json', 'evaluation_results_google.json')
