# -*- coding: utf-8 -*-
"""
Donut OCR Test - Segmentasyonsuz Fis Okuma
"""

import os
import sys
import time
from PIL import Image

# Windows console encoding fix
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

print("="*60)
print("DONUT TEST - End-to-End Document Understanding")
print("="*60)

# Model yükleme
print("\n[1/3] Model yükleniyor (~1.5GB, ilk seferde indirecek)...")
print("      Bu işlem birkaç dakika sürebilir...\n")

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
print(f"      Model yüklendi! ({load_time:.1f} saniye, device={device})")

# Test görüntüsü
print("\n[2/3] Görüntü işleniyor...")

project_root = os.path.dirname(os.path.abspath(__file__))
test_images = [
    os.path.join(project_root, "results", "test_crops", "fis1_crop.jpg"),
    os.path.join(project_root, "results", "test_crops", "fis11_crop.jpg"),
]

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"  HATA: {img_path} bulunamadı")
        continue

    print(f"\n  Dosya: {os.path.basename(img_path)}")

    # Görüntüyü yükle
    image = Image.open(img_path).convert("RGB")
    print(f"  Boyut: {image.size}")

    # İşle
    start_infer = time.time()

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

    # Decode (outputs is tensor directly)
    sequence = processor.batch_decode(outputs)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "")
    sequence = sequence.replace(processor.tokenizer.pad_token, "")

    infer_time = time.time() - start_infer
    print(f"  İşlem süresi: {infer_time:.1f} saniye")

    # Sonucu göster
    print(f"\n  --- RAW OUTPUT ---")
    print(f"  {sequence[:500]}")
    if len(sequence) > 500:
        print("  ...")

    # Basit parse
    import re

    # Total
    total_match = re.search(r'<s_total_price>(.*?)</s_total_price>', sequence)
    if total_match:
        print(f"\n  TOPLAM: {total_match.group(1)}")

    # Menu items
    items = re.findall(r'<s_nm>(.*?)</s_nm>.*?<s_price>(.*?)</s_price>', sequence)
    if items:
        print(f"  ÜRÜNLER:")
        for name, price in items[:5]:
            print(f"    - {name}: {price}")
        if len(items) > 5:
            print(f"    ... ve {len(items)-5} ürün daha")

print("\n" + "="*60)
print("TEST TAMAMLANDI")
print("="*60)
