# -*- coding: utf-8 -*-
"""
Ensemble OCR Test - EasyOCR + Donut Hybrid
En iyi sonucu secen akilli birlestirme
"""

import os
import sys
import re
import time
import cv2
from PIL import Image

# Windows console encoding fix
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from ocr_module import ReceiptOCR

print("="*60)
print("ENSEMBLE OCR - EasyOCR + Donut")
print("="*60)

# ============================================================
# MODEL YUKLEMELERI
# ============================================================
print("\n[1/5] Modeller yukleniyor...")

# EasyOCR
print("      EasyOCR yukleniyor...")
easyocr_engine = ReceiptOCR(languages=['tr', 'en'], use_gpu=False)
_ = easyocr_engine.get_ocr_reader()  # Pre-load
print("      EasyOCR hazir")

# Donut
print("      Donut yukleniyor (~1.5GB)...")
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

DONUT_MODEL = "naver-clova-ix/donut-base-finetuned-cord-v2"
donut_processor = DonutProcessor.from_pretrained(DONUT_MODEL)
donut_model = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
donut_model.to(device)
donut_model.eval()
print(f"      Donut hazir (device={device})")

# ============================================================
# OCR FONKSIYONLARI
# ============================================================

def run_easyocr(image_path):
    """EasyOCR ile isle."""
    start = time.time()

    image = cv2.imread(image_path)
    preprocessed = easyocr_engine.preprocess_image(image)
    text = easyocr_engine.read_text(preprocessed)
    data = easyocr_engine.extract_data(text)

    return {
        'tarih': data['tarih'],
        'toplam': data['toplam'],
        'text': text,
        'sure': time.time() - start
    }

def run_donut(image_path):
    """Donut ile isle."""
    start = time.time()

    image = Image.open(image_path).convert("RGB")

    task_prompt = "<s_cord-v2>"
    decoder_input_ids = donut_processor.tokenizer(
        task_prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids.to(device)

    pixel_values = donut_processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = donut_model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=donut_model.decoder.config.max_position_embeddings,
            pad_token_id=donut_processor.tokenizer.pad_token_id,
            eos_token_id=donut_processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
        )

    sequence = donut_processor.batch_decode(outputs)[0]
    sequence = sequence.replace(donut_processor.tokenizer.eos_token, "")
    sequence = sequence.replace(donut_processor.tokenizer.pad_token, "")

    # XML tag'lerini temizle
    clean_text = re.sub(r'<[^>]+>', ' ', sequence)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # Satirlara bol
    line_text = clean_text
    for sep in ['TARIH', 'SAAT', 'TOPLAM', 'KREDI', 'NAKIT', 'FIS', 'NO:', 'FATURA', 'DATE', 'PRIX', 'TOTAL']:
        line_text = line_text.replace(sep, '\n' + sep)

    # Regex ile parse et
    data = easyocr_engine.extract_data(line_text)

    return {
        'tarih': data['tarih'],
        'toplam': data['toplam'],
        'text': clean_text,
        'sure': time.time() - start
    }

def ensemble_merge(easyocr_result, donut_result):
    """Iki sonucu akilli birlestir."""

    # Tarih: Oncelik EasyOCR'a (daha tutarli), yoksa Donut
    tarih = easyocr_result['tarih'] or donut_result['tarih']

    # Toplam: Hangisi bulduysa onu al, ikisi de bulduysa buyuk olani sec
    # (genelde toplam en buyuk tutardir)
    easy_toplam = easyocr_result['toplam']
    donut_toplam = donut_result['toplam']

    if easy_toplam and donut_toplam:
        # Ikisi de buldu - buyuk olani al (veya ortalama yakin mi kontrol et)
        if abs(easy_toplam - donut_toplam) < 1:
            # Ayni deger, EasyOCR'a guven
            toplam = easy_toplam
            kaynak = "both"
        else:
            # Farkli degerler - buyuk olani sec (genelde dogru toplam buyuktur)
            toplam = max(easy_toplam, donut_toplam)
            kaynak = "easy" if easy_toplam > donut_toplam else "donut"
    elif easy_toplam:
        toplam = easy_toplam
        kaynak = "easy"
    elif donut_toplam:
        toplam = donut_toplam
        kaynak = "donut"
    else:
        toplam = None
        kaynak = None

    return {
        'tarih': tarih,
        'tarih_kaynak': 'easy' if easyocr_result['tarih'] else ('donut' if donut_result['tarih'] else None),
        'toplam': toplam,
        'toplam_kaynak': kaynak,
        'sure': easyocr_result['sure'] + donut_result['sure']
    }

# ============================================================
# TEST
# ============================================================
print("\n[2/5] Test goruntuleri...")

test_images = []
test_crops_dir = os.path.join(project_root, "results", "test_crops")
for f in sorted(os.listdir(test_crops_dir)):
    if f.endswith(('.jpg', '.png', '.jpeg')):
        test_images.append(os.path.join(test_crops_dir, f))

print(f"      {len(test_images)} goruntu bulundu")

print("\n[3/5] EasyOCR calistiriliyor...")
print("-"*60)

easyocr_results = {}
for img_path in test_images:
    fname = os.path.basename(img_path)
    result = run_easyocr(img_path)
    easyocr_results[fname] = result
    t = result['tarih'] or '-'
    p = f"{result['toplam']:.2f}" if result['toplam'] else '-'
    print(f"  {fname:<20} Tarih: {t:<12} Toplam: {p:<10} ({result['sure']:.1f}sn)")

print("\n[4/5] Donut calistiriliyor...")
print("-"*60)

donut_results = {}
for img_path in test_images:
    fname = os.path.basename(img_path)
    result = run_donut(img_path)
    donut_results[fname] = result
    t = result['tarih'] or '-'
    p = f"{result['toplam']:.2f}" if result['toplam'] else '-'
    print(f"  {fname:<20} Tarih: {t:<12} Toplam: {p:<10} ({result['sure']:.1f}sn)")

# ============================================================
# ENSEMBLE BIRLESTIRME
# ============================================================
print("\n[5/5] Ensemble birlestirme...")
print("="*60)

ensemble_results = []
for img_path in test_images:
    fname = os.path.basename(img_path)
    merged = ensemble_merge(easyocr_results[fname], donut_results[fname])
    merged['dosya'] = fname
    ensemble_results.append(merged)

# ============================================================
# SONUC TABLOSU
# ============================================================
print("\n" + "="*60)
print("KARSILASTIRMA TABLOSU")
print("="*60)

print(f"\n{'Dosya':<18} | {'EasyOCR':<22} | {'Donut':<22} | {'ENSEMBLE':<22}")
print(f"{'':<18} | {'Tarih':<10} {'Toplam':<10} | {'Tarih':<10} {'Toplam':<10} | {'Tarih':<10} {'Toplam':<10}")
print("-"*90)

for r in ensemble_results:
    fname = r['dosya']

    # EasyOCR
    et = easyocr_results[fname]['tarih'] or '-'
    ep = f"{easyocr_results[fname]['toplam']:.0f}" if easyocr_results[fname]['toplam'] else '-'

    # Donut
    dt = donut_results[fname]['tarih'] or '-'
    dp = f"{donut_results[fname]['toplam']:.0f}" if donut_results[fname]['toplam'] else '-'

    # Ensemble
    ent = r['tarih'] or '-'
    enp = f"{r['toplam']:.0f}" if r['toplam'] else '-'

    # Kaynak isaretleri
    if r['tarih_kaynak'] == 'donut' and not easyocr_results[fname]['tarih']:
        ent = ent + "*"
    if r['toplam_kaynak'] == 'donut' and not easyocr_results[fname]['toplam']:
        enp = enp + "*"

    print(f"{fname:<18} | {et:<10} {ep:<10} | {dt:<10} {dp:<10} | {ent:<10} {enp:<10}")

print("-"*90)
print("* = Donut'tan alindi (EasyOCR bulamadi)")

# Basari oranlari
print("\n" + "="*60)
print("BASARI ORANLARI")
print("="*60)

def calc_stats(results, key):
    return len([r for r in results if r[key]])

easy_tarih = len([r for r in easyocr_results.values() if r['tarih']])
easy_toplam = len([r for r in easyocr_results.values() if r['toplam']])
donut_tarih = len([r for r in donut_results.values() if r['tarih']])
donut_toplam = len([r for r in donut_results.values() if r['toplam']])
ens_tarih = len([r for r in ensemble_results if r['tarih']])
ens_toplam = len([r for r in ensemble_results if r['toplam']])

total = len(test_images)

print(f"\n{'Metrik':<15} | {'EasyOCR':<15} | {'Donut':<15} | {'ENSEMBLE':<15}")
print("-"*65)
print(f"{'Tarih':<15} | {easy_tarih}/{total} ({100*easy_tarih/total:.0f}%){'':<5} | {donut_tarih}/{total} ({100*donut_tarih/total:.0f}%){'':<5} | {ens_tarih}/{total} ({100*ens_tarih/total:.0f}%)")
print(f"{'Toplam':<15} | {easy_toplam}/{total} ({100*easy_toplam/total:.0f}%){'':<5} | {donut_toplam}/{total} ({100*donut_toplam/total:.0f}%){'':<5} | {ens_toplam}/{total} ({100*ens_toplam/total:.0f}%)")

# Sure
easy_time = sum(r['sure'] for r in easyocr_results.values())
donut_time = sum(r['sure'] for r in donut_results.values())
total_time = easy_time + donut_time

print(f"\n{'Toplam Sure':<15} | {easy_time:.1f}sn{'':<9} | {donut_time:.1f}sn{'':<8} | {total_time:.1f}sn (paralel: {max(easy_time, donut_time):.1f}sn)")

print("\n" + "="*60)
print("ENSEMBLE TAMAMLANDI")
print("="*60)
