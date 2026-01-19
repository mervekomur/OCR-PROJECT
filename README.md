# OCR-PROJECT

Fatura ve fişlerden otomatik veri çıkarma yapan ensemble OCR servisi.

> Bu proje staj kapsamında geliştirilmektedir.

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### SAM Model (Fiş Tespiti)

Fiş/belge segmentasyonu için SAM modelini indirin (~375MB):

```bash
curl -L -o sam_vit_b.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Veya PowerShell ile:

```powershell
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "sam_vit_b.pth"
```

### Ek OCR Motorları

```bash
# PaddleOCR
pip install paddlepaddle paddleocr

# Donut & GOT-OCR
pip install torch transformers

# SAM (Segment Anything)
pip install segment-anything
```

## Kullanım

### Fiş Tespiti (SAM)

```python
from src import detect_receipt, detect_and_save

# Hızlı tespit
result = detect_receipt("data/fis1.jpeg")
if result['success']:
    print(f"Köşeler: {result['corners']}")
    print(f"Alan: {result['area']}")

# Tespit ve kaydet
result = detect_and_save("data/fis1.jpeg", "output/")
# Çıktılar: fis1_sam_result.jpg, fis1_sam_cropped.jpg, fis1_sam_mask.jpg
```

### Tek Motor

```python
from src.engines import EasyOCREngine

engine = EasyOCREngine()
result = engine.extract("fis.jpg")

print(f"Merchant: {result.fields['merchant']}")
print(f"Total: {result.fields['total']}")
print(f"Confidence: {result.confidence:.1%}")
```

### Ensemble Karşılaştırma

```python
from src.engines import compare_engines

# Tüm motorları karşılaştır
result = compare_engines("fis.jpg", show_table=True)

# Belirli motorları karşılaştır
result = compare_engines("fis.jpg", engines=['easyocr', 'paddleocr'])

# En iyi sonuç
print(f"Best engine: {result.best_engine}")
```

### Komut Satırı

```bash
python examples/ensemble_demo.py data/fis1.jpg
python examples/ensemble_demo.py data/fis1.jpg easyocr,paddleocr
```

## OCR Motorları

| Motor | Durum | Açıklama |
|-------|-------|----------|
| EasyOCR | ✅ Aktif | Türkçe karakter desteği, genel amaçlı |
| PaddleOCR | ✅ Aktif | Yüksek doğruluk, çok dilli destek |
| Donut | ✅ Aktif | Document Understanding Transformer |
| GOT-OCR | ✅ Aktif | General OCR Theory, end-to-end |

## Mimari

```
src/
├── engines/              # OCR motor implementasyonları
│   ├── base.py          # Abstract base class
│   ├── easyocr_engine.py
│   ├── paddleocr_engine.py
│   ├── donut_engine.py
│   ├── got_ocr_engine.py
│   └── ensemble.py      # Karşılaştırma yöneticisi
├── parser/              # Metin ayrıştırma
│   ├── cleaner.py
│   ├── extractor.py
│   └── item_parser.py
├── sam_detector.py      # SAM tabanlı fiş/belge tespiti
├── constants.py         # Sabitler
├── preprocessing.py     # Görüntü ön işleme
└── batch_processor.py   # Toplu işleme
```

## Teknolojiler

- Python 3.10+
- SAM (Segment Anything Model) - Fiş/belge tespiti
- EasyOCR / PaddleOCR / Donut / GOT-OCR
- OpenCV
- Pillow
- PyTorch & Transformers
