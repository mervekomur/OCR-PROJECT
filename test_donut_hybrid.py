# -*- coding: utf-8 -*-
"""
Donut Hybrid Test - Donut OCR + Bizim Regex Parser
Segmentasyonsuz end-to-end fis okuma
"""

import os
import sys
import re
import time
from PIL import Image

# Windows console encoding fix
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Kendi regex modülümüzü import et
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
from ocr_module import ReceiptOCR

print("="*60)
print("DONUT HYBRID - Donut OCR + Regex Parser")
print("="*60)

# ============================================================
# DONUT MODEL YUKLE
# ============================================================
print("\n[1/4] Donut model yukleniyor...")
start_load = time.time()

from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"

processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

load_time = time.time() - start_load
print(f"      Model hazir ({load_time:.1f} sn, device={device})")

# Regex parser
receipt_ocr = ReceiptOCR()

# ============================================================
# DONUT RAW TEXT EXTRACTION
# ============================================================
def donut_extract_text(image_path):
    """Donut ile goruntuden raw text cikar."""
    image = Image.open(image_path).convert("RGB")

    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids.to(device)

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
        )

    sequence = processor.batch_decode(outputs)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "")
    sequence = sequence.replace(processor.tokenizer.pad_token, "")

    # XML tag'lerini temizle, sadece text birak
    clean_text = re.sub(r'<[^>]+>', ' ', sequence)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # Ayrica satirlara bol (daha iyi regex match icin)
    # Her ana bolumu ayri satira koy
    line_text = clean_text
    separators = ['TARIH', 'SAAT', 'TOPLAM', 'KREDI', 'NAKIT', 'FIS', 'NO:', 'FATURA']
    for sep in separators:
        line_text = line_text.replace(sep, '\n' + sep)

    return sequence, clean_text, line_text

# ============================================================
# TEST
# ============================================================
print("\n[2/4] Test goruntuleri hazirlaniyor...")

test_images = []
test_crops_dir = os.path.join(project_root, "results", "test_crops")
for f in os.listdir(test_crops_dir):
    if f.endswith(('.jpg', '.png', '.jpeg')):
        test_images.append(os.path.join(test_crops_dir, f))

test_images = sorted(test_images)
print(f"      {len(test_images)} goruntu bulundu")

print("\n[3/4] Islem basladi...")
print("="*60)

results = []

for img_path in test_images:
    filename = os.path.basename(img_path)
    print(f"\n>>> {filename}")
    print("-"*40)

    start = time.time()

    # Donut ile text cikar
    raw_output, clean_text, line_text = donut_extract_text(img_path)

    donut_time = time.time() - start

    # Bizim regex ile parse et
    data = receipt_ocr.extract_data(line_text)

    print(f"  Donut suresi: {donut_time:.1f} sn")
    print(f"  Temiz text ({len(clean_text)} kar):")
    print(f"    {clean_text[:200]}...")
    print(f"\n  SONUC:")
    print(f"    Tarih:  {data['tarih'] or 'Bulunamadi'}")
    print(f"    Toplam: {data['toplam'] or 'Bulunamadi'}")

    results.append({
        'dosya': filename,
        'tarih': data['tarih'],
        'toplam': data['toplam'],
        'sure': donut_time,
        'clean_text': clean_text
    })

# ============================================================
# OZET
# ============================================================
print("\n" + "="*60)
print("[4/4] SONUC TABLOSU")
print("="*60)

print(f"\n{'Dosya':<20} {'Tarih':<15} {'Toplam':<12} {'Sure':<8}")
print("-"*60)

for r in results:
    tarih = r['tarih'] or '-'
    toplam = f"{r['toplam']:.2f}" if r['toplam'] else '-'
    print(f"{r['dosya']:<20} {tarih:<15} {toplam:<12} {r['sure']:.1f}sn")

# Basari oranlari
tarih_ok = len([r for r in results if r['tarih']])
toplam_ok = len([r for r in results if r['toplam']])
total = len(results)

print("-"*60)
print(f"Tarih bulma:  {tarih_ok}/{total} ({100*tarih_ok/total:.0f}%)")
print(f"Toplam bulma: {toplam_ok}/{total} ({100*toplam_ok/total:.0f}%)")
print(f"Toplam sure:  {sum(r['sure'] for r in results):.1f} sn")
print(f"Ortalama:     {sum(r['sure'] for r in results)/total:.1f} sn/fis")

print("\n" + "="*60)
print("HYBRID TEST TAMAMLANDI")
print("="*60)
