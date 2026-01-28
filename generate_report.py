# -*- coding: utf-8 -*-
"""
OCR Benchmark Rapor Üreticisi
=============================
Detaylı analiz ve benchmark sonuçlarını markdown tablo olarak üretir.
"""

import json
import os
from datetime import datetime

def load_json(filename):
    if not os.path.exists(filename):
        return None
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_report():
    """Markdown formatında rapor üret"""

    # Verileri yükle
    gt_data = load_json('ground_truth.json')
    paddle_results = load_json('evaluation_results_paddle.json')
    google_results = load_json('evaluation_results_google.json')

    if not paddle_results:
        print("HATA: evaluation_results_paddle.json bulunamadi!")
        return

    stats = paddle_results.get('stats', {})
    details = paddle_results.get('results', {})
    fisler = gt_data.get('fisler', {}) if gt_data else {}

    # Rapor oluştur
    report = []

    # Başlık
    report.append("# OCR Benchmark Raporu")
    report.append(f"\n**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Toplam Dosya:** {len(details)}")
    report.append("")

    # ============================================
    # GENEL PERFORMANS TABLOSU
    # ============================================
    report.append("## 1. Genel Performans Özeti")
    report.append("")
    report.append("| Metrik | Doğru | Toplam | Başarı | Atlanan |")
    report.append("|--------|-------|--------|--------|---------|")

    total_correct = 0
    total_wrong = 0

    for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
        s = stats.get(field, {})
        c = s.get('correct', 0)
        w = s.get('wrong', 0)
        skip = s.get('skipped', 0)
        t = c + w
        total_correct += c
        total_wrong += w

        acc = f"%{(c/t)*100:.1f}" if t > 0 else "N/A"
        field_tr = {'tarih': 'Tarih', 'toplam': 'Toplam', 'kdv': 'KDV', 'vergi_no': 'Vergi No'}
        report.append(f"| {field_tr[field]} | {c} | {t} | {acc} | {skip} |")

    overall = (total_correct / (total_correct + total_wrong)) * 100 if (total_correct + total_wrong) > 0 else 0
    report.append(f"| **GENEL** | **{total_correct}** | **{total_correct + total_wrong}** | **%{overall:.1f}** | - |")
    report.append("")

    # ============================================
    # KATEGORİ BAZLI PERFORMANS
    # ============================================
    report.append("## 2. Kategori Bazlı Performans")
    report.append("")

    # Kategorilere ayır
    categories = {
        'TR Fişleri': [],
        'FR Fişleri': [],
        'Makbuzlar': [],
        'Diğer': []
    }

    for filename in details.keys():
        gt = fisler.get(filename, {})
        currency = gt.get('para_birimi', '')

        if filename.startswith('makbuz'):
            categories['Makbuzlar'].append(filename)
        elif currency == 'TL':
            categories['TR Fişleri'].append(filename)
        elif currency == 'EUR':
            categories['FR Fişleri'].append(filename)
        else:
            categories['Diğer'].append(filename)

    report.append("| Kategori | Dosya | Tarih | Toplam | KDV | Vergi No |")
    report.append("|----------|-------|-------|--------|-----|----------|")

    for cat_name, files in categories.items():
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
                return f"{s['c']}/{t} (%{(s['c']/t)*100:.0f})"
            return "N/A"

        report.append(f"| {cat_name} | {len(files)} | {fmt(cat_stats['tarih'])} | {fmt(cat_stats['toplam'])} | {fmt(cat_stats['kdv'])} | {fmt(cat_stats['vergi_no'])} |")

    report.append("")

    # ============================================
    # OCR MOTOR KARŞILAŞTIRMASI
    # ============================================
    report.append("## 3. OCR Motor Karşılaştırması")
    report.append("")
    report.append("| Motor | Tarih | Toplam | KDV | Vergi No | Genel |")
    report.append("|-------|-------|--------|-----|----------|-------|")

    # PaddleOCR
    def format_stat(s):
        c = s.get('correct', 0)
        w = s.get('wrong', 0)
        t = c + w
        if t > 0:
            return f"{c}/{t} (%{(c/t)*100:.0f})"
        return "N/A"

    p_overall = (total_correct / (total_correct + total_wrong)) * 100 if (total_correct + total_wrong) > 0 else 0
    report.append(f"| PaddleOCR | {format_stat(stats.get('tarih', {}))} | {format_stat(stats.get('toplam', {}))} | {format_stat(stats.get('kdv', {}))} | {format_stat(stats.get('vergi_no', {}))} | %{p_overall:.1f} |")

    # Google OCR (varsa)
    if google_results:
        g_stats = google_results.get('stats', {})
        g_total_c = sum(g_stats.get(f, {}).get('correct', 0) for f in ['tarih', 'toplam', 'kdv', 'vergi_no'])
        g_total_w = sum(g_stats.get(f, {}).get('wrong', 0) for f in ['tarih', 'toplam', 'kdv', 'vergi_no'])
        g_overall = (g_total_c / (g_total_c + g_total_w)) * 100 if (g_total_c + g_total_w) > 0 else 0
        report.append(f"| Google Vision | {format_stat(g_stats.get('tarih', {}))} | {format_stat(g_stats.get('toplam', {}))} | {format_stat(g_stats.get('kdv', {}))} | {format_stat(g_stats.get('vergi_no', {}))} | %{g_overall:.1f} |")
    else:
        report.append("| Google Vision | - | - | - | - | Test edilmedi |")

    report.append("")

    # ============================================
    # DETAYLI DOSYA BAZLI SONUÇLAR
    # ============================================
    report.append("## 4. Dosya Bazlı Detaylı Sonuçlar")
    report.append("")
    report.append("| Dosya | Market | Tarih | Toplam | KDV | Vergi No | Hata |")
    report.append("|-------|--------|-------|--------|-----|----------|------|")

    for filename, result in sorted(details.items()):
        gt = fisler.get(filename, {})
        market = gt.get('market_adi', '-')[:25]

        errors = 0
        status = []

        for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
            if field in result:
                match = result[field].get('match')
                if match == True:
                    status.append("✓")
                elif match == False:
                    status.append("✗")
                    errors += 1
                else:
                    status.append("-")
            else:
                status.append("-")

        report.append(f"| {filename} | {market} | {status[0]} | {status[1]} | {status[2]} | {status[3]} | {errors} |")

    report.append("")

    # ============================================
    # BAŞARISIZ DETAYLAR
    # ============================================
    report.append("## 5. Başarısız Sonuçların Detayı")
    report.append("")

    for field in ['tarih', 'toplam', 'kdv', 'vergi_no']:
        field_tr = {'tarih': 'Tarih', 'toplam': 'Toplam', 'kdv': 'KDV', 'vergi_no': 'Vergi No'}
        failures = []

        for filename, result in details.items():
            if field in result and result[field].get('match') == False:
                failures.append({
                    'file': filename,
                    'gt': result[field].get('gt'),
                    'ocr': result[field].get('ocr')
                })

        if failures:
            report.append(f"### {field_tr[field]} Hataları ({len(failures)} adet)")
            report.append("")
            report.append("| Dosya | Beklenen (GT) | OCR Sonucu |")
            report.append("|-------|---------------|------------|")

            for f in failures:
                gt_val = f['gt'] if f['gt'] else '-'
                ocr_val = f['ocr'] if f['ocr'] else '-'
                report.append(f"| {f['file']} | {gt_val} | {ocr_val} |")

            report.append("")

    # ============================================
    # EN SORUNLU DOSYALAR
    # ============================================
    report.append("## 6. En Sorunlu Dosyalar (Top 10)")
    report.append("")
    report.append("| Sıra | Dosya | Hata Sayısı | Market |")
    report.append("|------|-------|-------------|--------|")

    file_errors = {}
    for filename, result in details.items():
        errors = sum(1 for f in ['tarih', 'toplam', 'kdv', 'vergi_no']
                    if f in result and result[f].get('match') == False)
        if errors > 0:
            file_errors[filename] = errors

    sorted_errors = sorted(file_errors.items(), key=lambda x: x[1], reverse=True)[:10]

    for i, (filename, err_count) in enumerate(sorted_errors, 1):
        gt = fisler.get(filename, {})
        market = gt.get('market_adi', '-')[:30]
        report.append(f"| {i} | {filename} | {err_count} | {market} |")

    report.append("")

    # ============================================
    # ÖNERİLER
    # ============================================
    report.append("## 7. İyileştirme Önerileri")
    report.append("")

    # KDV analizi
    kdv_stats = stats.get('kdv', {})
    kdv_acc = kdv_stats.get('correct', 0) / max(1, kdv_stats.get('correct', 0) + kdv_stats.get('wrong', 0))

    if kdv_acc < 0.5:
        report.append("- [ ] **KDV Parser İyileştirmesi**: TVA pattern'leri genişletilmeli (FR fişleri için)")

    # Vergi No analizi
    vn_stats = stats.get('vergi_no', {})
    vn_acc = vn_stats.get('correct', 0) / max(1, vn_stats.get('correct', 0) + vn_stats.get('wrong', 0))

    if vn_acc < 0.5:
        report.append("- [ ] **Vergi No Parser İyileştirmesi**: V.D. pattern'leri düzeltilmeli")

    # Makbuz analizi
    makbuz_count = len([f for f in details.keys() if f.startswith('makbuz')])
    if makbuz_count > 0:
        report.append("- [ ] **Makbuz Preprocessing**: Gölge kaldırma ve kırpma eklenmeli")

    report.append("- [ ] **Google Vision OCR**: Karşılaştırma için test edilmeli")
    report.append("- [ ] **Ensemble Yaklaşım**: Birden fazla OCR motoru birleştirilebilir")
    report.append("")

    # Dosyaya yaz
    report_text = "\n".join(report)

    with open('BENCHMARK_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("Rapor oluşturuldu: BENCHMARK_REPORT.md")
    print("\n" + "="*50)
    print(report_text)

    return report_text


if __name__ == '__main__':
    generate_report()
