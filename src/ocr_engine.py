"""
OCR Engine Modulu
EasyOCR kullanarak goruntu uzerinden metin cikarma islemleri.
"""

import easyocr
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


class OCREngine:
    """EasyOCR tabanli OCR motoru."""

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        """
        OCR motorunu baslatir.

        Args:
            languages: Desteklenecek diller listesi (varsayilan: Turkce ve Ingilizce)
            gpu: GPU kullanimi (varsayilan: False)
        """
        if languages is None:
            languages = ['tr', 'en']

        self.languages = languages
        self.gpu = gpu
        self.reader = None

    def _ensure_reader(self):
        """Reader'in yuklenmesini saglar (lazy loading)."""
        if self.reader is None:
            print(f"EasyOCR yukleniyor (diller: {self.languages})...")
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            print("EasyOCR hazir.")

    def extract_text(self, image: np.ndarray, detail: bool = False) -> Any:
        """
        Goruntu uzerinden metin cikarir.

        Args:
            image: numpy array formatinda goruntu
            detail: True ise konum bilgisi ile birlikte dondurur

        Returns:
            detail=False: Sadece metin (string)
            detail=True: Liste [(bbox, text, confidence), ...]
        """
        self._ensure_reader()

        results = self.reader.readtext(image)

        if detail:
            return results

        # Sadece metinleri birlestir
        texts = [item[1] for item in results]
        return '\n'.join(texts)

    def extract_text_from_file(
        self,
        image_path: str,
        detail: bool = False
    ) -> Any:
        """
        Dosyadan goruntu yukleyip metin cikarir.

        Args:
            image_path: Goruntu dosyasinin yolu
            detail: True ise konum bilgisi ile birlikte dondurur

        Returns:
            detail=False: Sadece metin (string)
            detail=True: Liste [(bbox, text, confidence), ...]
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Dosya bulunamadi: {image_path}")

        self._ensure_reader()

        results = self.reader.readtext(str(path))

        if detail:
            return results

        texts = [item[1] for item in results]
        return '\n'.join(texts)

    def extract_structured(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Goruntuden yapilandirilmis veri cikarir.

        Args:
            image: numpy array formatinda goruntu

        Returns:
            Dict: {
                'raw_text': str,
                'lines': List[Dict],  # Her satir icin {text, confidence, bbox}
                'confidence_avg': float
            }
        """
        self._ensure_reader()

        results = self.reader.readtext(image)

        lines = []
        total_confidence = 0

        for bbox, text, confidence in results:
            lines.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
            total_confidence += confidence

        avg_confidence = total_confidence / len(results) if results else 0

        return {
            'raw_text': '\n'.join([item[1] for item in results]),
            'lines': lines,
            'confidence_avg': avg_confidence
        }


# Modul seviyesinde kullanim icin varsayilan instance
_default_engine: Optional[OCREngine] = None


def get_engine() -> OCREngine:
    """Varsayilan OCR motorunu dondurur."""
    global _default_engine
    if _default_engine is None:
        _default_engine = OCREngine()
    return _default_engine


def extract_text(image_path: str) -> str:
    """
    Basit kullanim icin fonksiyon.

    Args:
        image_path: Goruntu dosyasinin yolu

    Returns:
        str: Cikartilan metin
    """
    engine = get_engine()
    return engine.extract_text_from_file(image_path)


# Ornek kullanim
if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("OCR Engine - EasyOCR")
    print("=" * 50)

    if len(sys.argv) > 1:
        image_file = sys.argv[1]

        try:
            # OCR motoru olustur
            engine = OCREngine(languages=['tr', 'en'], gpu=False)

            # Detayli sonuc al
            result = engine.extract_text_from_file(image_file, detail=True)

            print(f"\nDosya: {image_file}")
            print("-" * 50)
            print("CIKARTILAN METIN:")
            print("-" * 50)

            for bbox, text, confidence in result:
                conf_percent = confidence * 100
                print(f"[%{conf_percent:.1f}] {text}")

            print("-" * 50)
            print("\nTAM METIN:")
            print("-" * 50)
            full_text = '\n'.join([item[1] for item in result])
            print(full_text)

        except Exception as e:
            print(f"Hata: {e}")
            sys.exit(1)
    else:
        print("\nKullanim:")
        print("-" * 50)
        print("python ocr_engine.py <goruntu_dosyasi>")
        print("\nOrnek:")
        print("  python ocr_engine.py fatura.jpg")
        print("  python ocr_engine.py fis_processed.png")

        print("\n" + "=" * 50)
        print("Kod Icinden Kullanim:")
        print("=" * 50)
        print("""
# Basit kullanim
from ocr_engine import extract_text

text = extract_text("fatura.jpg")
print(text)

# Detayli kullanim
from ocr_engine import OCREngine

engine = OCREngine(languages=['tr', 'en'])
result = engine.extract_structured(image)

print(f"Ortalama guven: {result['confidence_avg']:.2%}")
for line in result['lines']:
    print(f"{line['text']} ({line['confidence']:.2%})")
""")
