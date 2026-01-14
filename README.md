# OCR-PROJECT

Fatura ve fişlerden otomatik veri çıkarma yapan ensemble OCR servisi.

> Bu proje staj kapsamında geliştirilmektedir.

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Kullanım

```python
from src.ocr_engine import OCREngine
from src.receipt_parser import ReceiptParser

engine = OCREngine()
parser = ReceiptParser()

result = engine.extract_text("fis.jpg")
parsed = parser.parse(result)
```

## OCR Motorları

| Motor | Durum | Açıklama |
|-------|-------|----------|
| EasyOCR | ✅ Aktif | Türkçe karakter desteği, genel amaçlı |
| PaddleOCR | 🔄 Planlı | Yüksek doğruluk, çok dilli destek |
| GOT-OCR | 🔄 Planlı | Transformer tabanlı, end-to-end |

## Mimari

```
src/
├── ocr_engine.py       # OCR motor yönetimi
├── preprocessing.py    # Görüntü ön işleme
├── receipt_parser.py   # Fiş/fatura ayrıştırma
└── batch_processor.py  # Toplu işleme
```

## Teknolojiler

- Python 3.10+
- EasyOCR / PaddleOCR / GOT-OCR
- OpenCV
- Pillow
