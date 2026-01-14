# OCR-PROJECT

Fatura ve fişlerden otomatik veri çıkarma yapan OCR servisi.

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

## Desteklenen OCR Motorları

- [x] EasyOCR
- [ ] PaddleOCR
- [ ] GOT-OCR

## Teknolojiler

- Python 3.10+
- EasyOCR
- OpenCV
- Pillow
