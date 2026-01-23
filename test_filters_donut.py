# -*- coding: utf-8 -*-
"""
Filtreli gorselleri Donut ile oku ve karsilastir
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

print("="*60)
print("DONUT ILE FILTRE KARSILASTIRMASI")
print("="*60)

# Donut yukle
print("\nDonut yukleniyor...")
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re

MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
device = "cpu"
model.to(device)
model.eval()
print("Donut hazir!\n")

def donut_read(image_path):
    """Donut ile oku, tarih ve toplam cikar."""
    image = Image.open(image_path).convert("RGB")

    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
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

    # Tarih ara
    tarih_match = re.search(r'(\d{2}[./-]\d{2}[./-]\d{4})', sequence)
    tarih = tarih_match.group(1) if tarih_match else None

    # Toplam ara (TOPLAM, TOTAL, vb yakininda sayi)
    toplam_match = re.search(r'[*]?(\d+[.,]\d{2})', sequence)
    toplam = toplam_match.group(1) if toplam_match else None

    return tarih, toplam, len(sequence)

# Test dosyalari
filter_dir = "results/filter_test_fis1"
filters = [
    ("filter1_gray.jpg", "Grayscale"),
    ("filter2_binary.jpg", "Binary Threshold"),
    ("filter3_adaptive.jpg", "Adaptive Threshold"),
    ("filter4_morph.jpg", "Adaptive + Dilation"),
]

# Orijinal de test et
print("="*60)
results = []

# Orijinal
print("\n[0] Orijinal (data/fis1.jpeg)...")
tarih, toplam, chars = donut_read("data/fis1.jpeg")
print(f"    Tarih: {tarih}, Toplam: {toplam}, Karakter: {chars}")
results.append(("Orijinal", tarih, toplam, chars))

# Filtreler
for i, (filename, name) in enumerate(filters, 1):
    filepath = os.path.join(filter_dir, filename)
    print(f"\n[{i}] {name}...")
    tarih, toplam, chars = donut_read(filepath)
    print(f"    Tarih: {tarih}, Toplam: {toplam}, Karakter: {chars}")
    results.append((name, tarih, toplam, chars))

# Ozet tablo
print("\n" + "="*60)
print("SONUC TABLOSU")
print("="*60)
print(f"\n{'Filtre':<22} | {'Tarih':<12} | {'Toplam':<10} | {'Kar.':<6}")
print("-"*60)
for name, tarih, toplam, chars in results:
    t = tarih or "-"
    p = toplam or "-"
    print(f"{name:<22} | {t:<12} | {p:<10} | {chars:<6}")

print("\n" + "="*60)
print("Beklenen: Tarih=05/08/2025, Toplam=242.00")
print("="*60)
