"""
Claude Vision OCR Engine - FLO Masraf Modülü v3
Direct image-to-structured-data using Claude's multimodal capabilities.

Features:
- Multi-language support (TR, EN, DE, FR, etc.)
- TESK rule: KDV %0 for taxi receipts with TESK
- Account code matching (66 keywords)
- Withholding tax (Tevkifat) detection
- KKEG calculation (vehicle expenses 70/30 split)
- VKN checksum validation
- Document type detection (Fatura, Fiş, Gider Pusulası, Bilgi Fişi)
- Personal invoice rejection rule
- TODO: Employee name fuzzy matching (Levenshtein Distance) - gelecekte eklenecek
"""

import os
import base64
import json
import re
import time
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

from .base import BaseOCREngine, OCRResult


# =============================================================================
# FLO GROUP COMPANIES
# =============================================================================
FLO_COMPANIES = {
    "3880239429": {"name": "FLO Mağazacılık", "code": "FLO_MAG"},
    "8721503797": {"name": "Turuncu Ayakkabı", "code": "TURUNCU"},
    "3881618492": {"name": "FLO Teknoloji", "code": "FLO_TEK"},
    "3881765897": {"name": "FLO İç Dış Ticaret", "code": "FLO_IHR"},
}

# =============================================================================
# VALID KDV RATES IN TURKEY
# =============================================================================
VALID_KDV_RATES = [0, 1, 8, 10, 18, 20]

# =============================================================================
# WITHHOLDING TAX (TEVKİFAT) RATES
# =============================================================================
TEVKIFAT_RATES = {
    "9/10": 0.90,  # İşgücü temin hizmetleri
    "7/10": 0.70,  # Yapım işleri
    "5/10": 0.50,  # Temizlik, bahçe bakım
    "3/10": 0.30,  # Makine/teçhizat bakım
    "2/10": 0.20,  # Etüt, plan, proje
}

# =============================================================================
# ACCOUNT CODE KEYWORDS (66 entries from FLO document)
# =============================================================================
ACCOUNT_KEYWORDS = {
    # Yemek Giderleri
    "yemek": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001"},
    "restoran": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001"},
    "lokanta": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001"},
    "cafe": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001"},
    "kahvaltı": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001"},
    "öğle yemeği": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001"},
    "akşam yemeği": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001"},
    "menü": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001"},
    "yiyecek": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001"},
    "içecek": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001"},

    # Taksi Giderleri
    "taksi": {"hesap_adi": "Taksi Gideri", "hesap_kodu": "760.02.001"},
    "taxi": {"hesap_adi": "Taksi Gideri", "hesap_kodu": "760.02.001"},
    "tesk": {"hesap_adi": "Taksi Gideri (TESK)", "hesap_kodu": "760.02.001", "kdv_oran": 0},

    # Diğer Ulaşım
    "otobüs": {"hesap_adi": "Ulaşım Gideri", "hesap_kodu": "760.02.002"},
    "havaş": {"hesap_adi": "Ulaşım Gideri", "hesap_kodu": "760.02.002"},
    "havaist": {"hesap_adi": "Ulaşım Gideri", "hesap_kodu": "760.02.002"},
    "dolmuş": {"hesap_adi": "Ulaşım Gideri", "hesap_kodu": "760.02.002"},
    "vapur": {"hesap_adi": "Ulaşım Gideri", "hesap_kodu": "760.02.002"},
    "iett": {"hesap_adi": "Ulaşım Gideri", "hesap_kodu": "760.02.002"},
    "metro": {"hesap_adi": "Ulaşım Gideri", "hesap_kodu": "760.02.002"},
    "metrobüs": {"hesap_adi": "Ulaşım Gideri", "hesap_kodu": "760.02.002"},
    "istanbulkart": {"hesap_adi": "Ulaşım Gideri", "hesap_kodu": "760.02.002"},
    "marmaray": {"hesap_adi": "Ulaşım Gideri", "hesap_kodu": "760.02.002"},

    # Araç Kiralama / Binek Araç
    "araç kiralama": {"hesap_adi": "Binek Araç Gideri", "hesap_kodu": "760.02.003", "kkeg": True},
    "rent a car": {"hesap_adi": "Binek Araç Gideri", "hesap_kodu": "760.02.003", "kkeg": True},
    "europcar": {"hesap_adi": "Binek Araç Gideri", "hesap_kodu": "760.02.003", "kkeg": True},
    "avis": {"hesap_adi": "Binek Araç Gideri", "hesap_kodu": "760.02.003", "kkeg": True},
    "hertz": {"hesap_adi": "Binek Araç Gideri", "hesap_kodu": "760.02.003", "kkeg": True},
    "budget": {"hesap_adi": "Binek Araç Gideri", "hesap_kodu": "760.02.003", "kkeg": True},
    "enterprise": {"hesap_adi": "Binek Araç Gideri", "hesap_kodu": "760.02.003", "kkeg": True},

    # Akaryakıt
    "benzin": {"hesap_adi": "Akaryakıt Gideri", "hesap_kodu": "760.02.004", "kkeg": True},
    "motorin": {"hesap_adi": "Akaryakıt Gideri", "hesap_kodu": "760.02.004", "kkeg": True},
    "lpg": {"hesap_adi": "Akaryakıt Gideri", "hesap_kodu": "760.02.004", "kkeg": True},
    "akaryakıt": {"hesap_adi": "Akaryakıt Gideri", "hesap_kodu": "760.02.004", "kkeg": True},
    "petrol": {"hesap_adi": "Akaryakıt Gideri", "hesap_kodu": "760.02.004", "kkeg": True},
    "shell": {"hesap_adi": "Akaryakıt Gideri", "hesap_kodu": "760.02.004", "kkeg": True},
    "opet": {"hesap_adi": "Akaryakıt Gideri", "hesap_kodu": "760.02.004", "kkeg": True},
    "bp": {"hesap_adi": "Akaryakıt Gideri", "hesap_kodu": "760.02.004", "kkeg": True},
    "total": {"hesap_adi": "Akaryakıt Gideri", "hesap_kodu": "760.02.004", "kkeg": True},
    "petrol ofisi": {"hesap_adi": "Akaryakıt Gideri", "hesap_kodu": "760.02.004", "kkeg": True},

    # Kırtasiye
    "kırtasiye": {"hesap_adi": "Kırtasiye Gideri", "hesap_kodu": "760.03.001"},
    "kalem": {"hesap_adi": "Kırtasiye Gideri", "hesap_kodu": "760.03.001"},
    "defter": {"hesap_adi": "Kırtasiye Gideri", "hesap_kodu": "760.03.001"},
    "kağıt": {"hesap_adi": "Kırtasiye Gideri", "hesap_kodu": "760.03.001"},
    "toner": {"hesap_adi": "Kırtasiye Gideri", "hesap_kodu": "760.03.001"},
    "kartuş": {"hesap_adi": "Kırtasiye Gideri", "hesap_kodu": "760.03.001"},

    # Konaklama
    "otel": {"hesap_adi": "Konaklama Gideri", "hesap_kodu": "760.04.001"},
    "hotel": {"hesap_adi": "Konaklama Gideri", "hesap_kodu": "760.04.001"},
    "konaklama": {"hesap_adi": "Konaklama Gideri", "hesap_kodu": "760.04.001"},
    "pansiyon": {"hesap_adi": "Konaklama Gideri", "hesap_kodu": "760.04.001"},

    # Temizlik
    "temizlik": {"hesap_adi": "Temizlik Gideri", "hesap_kodu": "760.05.001"},
    "hijyen": {"hesap_adi": "Temizlik Gideri", "hesap_kodu": "760.05.001"},
    "deterjan": {"hesap_adi": "Temizlik Gideri", "hesap_kodu": "760.05.001"},

    # İçme Suyu
    "damacana": {"hesap_adi": "İçme Suyu Gideri", "hesap_kodu": "760.06.001"},
    "su": {"hesap_adi": "İçme Suyu Gideri", "hesap_kodu": "760.06.001"},

    # Bakım Onarım
    "bakım": {"hesap_adi": "Bakım Onarım Gideri", "hesap_kodu": "760.07.001"},
    "onarım": {"hesap_adi": "Bakım Onarım Gideri", "hesap_kodu": "760.07.001"},
    "tamir": {"hesap_adi": "Bakım Onarım Gideri", "hesap_kodu": "760.07.001"},
    "servis": {"hesap_adi": "Bakım Onarım Gideri", "hesap_kodu": "760.07.001"},

    # Kargo / Posta
    "kargo": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001"},
    "ptt": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001"},
    "mng": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001"},
    "yurtiçi": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001"},
    "aras": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001"},
    "ups": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001"},
    "dhl": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001"},
    "fedex": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001"},
}

# =============================================================================
# DOCUMENT TYPE KEYWORDS
# =============================================================================
DOC_TYPE_KEYWORDS = {
    "FATURA": ["e-arşiv fatura", "e-fatura", "fatura no", "invoice", "rechnung", "facture"],
    "FIS": ["fiş no", "yazar kasa", "perakende satış", "receipt", "quittung", "bon"],
    "GIDER_PUSULASI": ["gider pusulası", "expense voucher"],
    "BILGI_FISI": ["bilgi fişi", "e-arşiv bilgi"],
    "TEVKIFATLI": ["kdv tevkifat", "tevkifatlı", "withholding"],
}

# =============================================================================
# LANGUAGE DETECTION PATTERNS
# =============================================================================
LANGUAGE_PATTERNS = {
    "TR": ["toplam", "kdv", "tarih", "fatura", "fiş", "tutar", "ödeme"],
    "EN": ["total", "vat", "date", "invoice", "receipt", "amount", "payment"],
    "DE": ["gesamt", "mwst", "datum", "rechnung", "quittung", "betrag", "zahlung"],
    "FR": ["total", "tva", "date", "facture", "reçu", "montant", "paiement"],
    "IT": ["totale", "iva", "data", "fattura", "ricevuta", "importo", "pagamento"],
    "ES": ["total", "iva", "fecha", "factura", "recibo", "importe", "pago"],
}

# =============================================================================
# TODO: EMPLOYEE LIST (Çalışan Eşleştirme - Gelecekte Eklenecek)
# =============================================================================
# Çalışan isim eşleştirme fonksiyonu ileride eklenecek.
# Gerekli: SAP/HR sisteminden çalışan listesi çekilmeli
# Algoritma: Levenshtein Distance ile %80 benzerlik eşiği


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_personal_name(name: str) -> bool:
    """
    Check if a string looks like a personal name (not a company name).

    Personal names typically:
    - Don't contain company suffixes (A.Ş., LTD., ŞTİ., etc.)
    - Are shorter
    - Contain common Turkish name patterns
    """
    if not name:
        return False

    name_upper = name.upper()

    # Company suffixes that indicate it's NOT a personal name
    company_indicators = [
        "A.Ş.", "A.Ş", "AŞ", "A.S.",
        "LTD.", "LTD", "LİMİTED",
        "ŞTİ.", "ŞTİ", "STI.", "STI",
        "SAN.", "SAN", "SANAYİ",
        "TİC.", "TİC", "TİCARET",
        "GIDA", "TEKSTİL", "OTOMOTİV",
        "MARKET", "MAĞAZA", "RESTORAN",
        "INC.", "LLC", "GMBH", "AG",
        "CORPORATION", "CORP.",
    ]

    for indicator in company_indicators:
        if indicator in name_upper:
            return False

    # If name is very long, probably a company
    if len(name) > 50:
        return False

    # If name has more than 4 words, probably a company
    words = name.split()
    if len(words) > 4:
        return False

    # If all checks pass, assume it could be a personal name
    return True


@dataclass
class KKEGCalculation:
    """KKEG (Kanunen Kabul Edilmeyen Gider) calculation result."""
    brut_tutar: float
    matrah: float  # brut / 1.20
    gider_tutar: float  # brut * 0.70
    kkeg_gider: float  # matrah * 0.30
    kkeg_kdv: float  # kkeg_gider * 0.20

    def to_dict(self) -> Dict[str, float]:
        return {
            "brut_tutar": round(self.brut_tutar, 2),
            "matrah": round(self.matrah, 2),
            "gider_tutar": round(self.gider_tutar, 2),
            "kkeg_gider": round(self.kkeg_gider, 2),
            "kkeg_kdv": round(self.kkeg_kdv, 2)
        }


class ClaudeVisionEngine(BaseOCREngine):
    """
    Claude Vision-based OCR engine with FLO business rules.

    Features:
    - Multi-language OCR (TR, EN, DE, FR, IT, ES)
    - TESK rule: KDV %0 for taxi receipts
    - Account code matching from 66 keywords
    - Withholding tax detection
    - KKEG calculation for vehicle expenses
    - VKN/TCKN checksum validation
    - Personal invoice rejection rule
    - TODO: Employee name fuzzy matching (gelecekte eklenecek)
    """

    name = "claude_vision"
    description = "Claude Vision - FLO Masraf Modülü with full business rules"

    SYSTEM_PROMPT = """Sen FLO Grup şirketleri için çok dilli fiş/fatura OCR ve analiz uzmanısın.

## GÖREV
Görüntüdeki fiş/fatura metnini oku ve SAP'ye aktarılabilecek yapılandırılmış JSON üret.

## ÇOK ÖNEMLİ: SATICI ve ALICI AYRIMI

Faturada İKİ FARKLI VKN olabilir - bunları KARIŞTIRMA:

### SATICI (Üstteki/Faturayı Kesen):
- Faturanın EN ÜSTÜNDE yazan firma
- "Satıcı", "Firma", "Şirket" başlığı altında
- Mal/hizmet SATAN taraf

### ALICI (Alttaki/Müşteri - BU ÖNEMLİ):
- "SAYIN", "ALICI", "MÜŞTERİ", "Customer", "To:", "Bill To:" başlıkları altında
- Faturanın KESİLDİĞİ taraf
- BU VKN'Yİ KONTROL EDECEĞİZ!

## FLO GRUP ŞİRKETLERİ (Sadece ALICI VKN olabilir)
- 3880239429 → FLO Mağazacılık
- 8721503797 → Turuncu Ayakkabı
- 3881618492 → FLO Teknoloji
- 3881765897 → FLO İç Dış Ticaret

ALICI bölümündeki VKN bu 4 şirketten biri DEĞİLSE veya orada şahıs ismi yazıyorsa → RED

## DİL TESPİTİ
- Türkçe: "toplam", "kdv", "tarih", "fatura"
- İngilizce: "total", "vat", "date", "invoice"
- Almanca: "gesamt", "mwst", "datum", "rechnung"
- Fransızca: "total", "tva", "date", "facture"

## BELGE TÜRÜ TESPİTİ
- "e-arşiv fatura" veya "e-fatura" veya "invoice" → FATURA
- "fiş no" veya "yazar kasa" veya "receipt" → FİŞ
- "gider pusulası" → GIDER_PUSULASI
- "bilgi fişi" → BILGI_FISI
- "kdv tevkifat" ibaresi varsa → TEVKİFATLI olarak işaretle

## KRİTİK KURALLAR

### 1. TESK KURALI (TAKSİ FİŞLERİ)
Taksi fişinde "TESK" (Türkiye Esnaf ve Sanatkarları Konfederasyonu) ibaresi varsa:
- KDV oranı = %0
- Eğer TESK yoksa KDV = %20

### 2. ALICI KONTROLÜ (EN ÖNEMLİ KURAL)
SADECE "Sayın/Alıcı/Müşteri" bölümündeki bilgiyi kontrol et:
- Orada VKN varsa → 4'lü FLO listesiyle karşılaştır
- Orada TCKN varsa (11 hane) → Şahıs faturası → RED
- Orada sadece kişi ismi varsa → Şahıs faturası → RED
- VKN yoksa ve alıcı bilgisi yoksa → Perakende fiş

### 3. TEVKİFAT TESPİTİ
Aşağıdakilerden biri varsa tevkifatlı:
- "KDV Tevkifatlıdır" ibaresi
- Oranlı bölünmüş KDV satırı
- Hizmet açıklaması tevkifat listesinde

### 4. KKEG (BİNEK ARAÇ)
Araç kiralama, akaryakıt gibi binek araç giderlerinde:
- Gider: %70
- KKEG: %30

## PARA BİRİMİ TESPİTİ
- TL, TRY, ₺ → TRY
- EUR, € → EUR
- USD, $ → USD
- CHF → CHF
- GBP, £ → GBP

## OCR TALİMATLARI
- Görüntüdeki TÜM metni dikkatli oku
- Slashed Zero düzeltmeleri: 0↔O, 8↔B, 5↔S, 1↔I/l
- VKN/TCKN'leri dikkatli oku (10 veya 11 hane)
- Tutarları nokta/virgül formatına dikkat ederek oku

SADECE JSON FORMATINDA YANIT VER."""

    USER_PROMPT = """Bu fiş/fatura görüntüsünü analiz et.

## ADIMLAR:
1. Belgenin dilini tespit et
2. Tüm metni oku
3. TESK ibaresi var mı kontrol et (taksi fişi ise)
4. SATICI bilgilerini bul (faturayı KESEN firma - genelde üstte)
5. ALICI bilgilerini bul (faturanın KESİLDİĞİ taraf):
   - "SAYIN", "ALICI", "MÜŞTERİ", "Customer", "To:", "Bill To:" başlıklarını ara
   - Bu başlıkların ALTINDA veya YANINDA yazan VKN/TCKN/ismi al
   - SATICI VKN'yi ALICI olarak yazma!

## ÖNEMLİ:
- Faturada 2 VKN olabilir: biri SATICI, biri ALICI
- Sadece ALICI bölümündeki VKN'yi "alici.vkn_veya_tckn" alanına yaz
- Eğer ALICI bölümünde VKN yoksa ve sadece kişi ismi varsa, onu "alici.unvan_veya_ad" alanına yaz

JSON formatı:
{
    "raw_text": "Görüntüden okunan ham metin",
    "detected_language": "TR|EN|DE|FR|IT|ES",
    "header": {
        "belge_turu": "FATURA|FIS|GIDER_PUSULASI|BILGI_FISI",
        "tevkifatli": true/false,
        "seri_no": "Seri numarası veya null",
        "belge_no": "Belge/Fiş numarası",
        "belge_tarihi": "DD.MM.YYYY formatında",
        "saat": "HH:MM veya null",
        "satici": {
            "vkn": "SATICI VKN (faturayı kesen firma)",
            "firma_adi": "Satıcı firma adı",
            "adres": "Adres veya null",
            "ulke": "Ülke kodu (TR, DE, CH, vb.)"
        },
        "alici": {
            "vkn_veya_tckn": "ALICI bölümündeki VKN/TCKN (Sayın/Müşteri altındaki)",
            "tip": "TUZEL_KISI|GERCEK_KISI|null",
            "unvan_veya_ad": "ALICI bölümündeki şirket adı veya kişi ismi",
            "adres": "Alıcı adresi veya null"
        }
    },
    "financials": {
        "brut_tutar": Brüt toplam (sayı),
        "kdv_haric_tutar": KDV hariç tutar veya null,
        "kdv_orani": KDV yüzdesi (0, 1, 8, 10, 18, 20),
        "kdv_tutari": KDV miktarı veya null,
        "para_birimi": "TRY|EUR|USD|CHF|GBP",
        "odeme_sekli": "NAKIT|KREDI_KARTI|HAVALE|null",
        "tevkifat_orani": "9/10|7/10|5/10|3/10|2/10|null",
        "tevkifat_tutari": Tevkifat tutarı veya null
    },
    "items": [
        {
            "aciklama": "Ürün/hizmet açıklaması",
            "miktar": Adet,
            "birim_fiyat": Birim fiyat,
            "tutar": Satır tutarı,
            "kdv_orani": Satır KDV oranı
        }
    ],
    "tesk_detected": true/false,
    "keywords_found": ["tespit edilen anahtar kelimeler"],
    "ocr_notes": ["OCR düzeltmeleri ve notlar"]
}"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514"
    ):
        super().__init__()
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    @classmethod
    def _check_availability(cls) -> bool:
        try:
            import anthropic
            return True
        except ImportError:
            return False

    def _initialize(self) -> None:
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. "
                "Set ANTHROPIC_API_KEY environment variable."
            )
        import anthropic
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def _image_to_base64(self, image_path: str) -> Tuple[str, str]:
        """Convert image to base64 and detect media type."""
        path = Path(image_path)
        suffix = path.suffix.lower()

        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }

        media_type = media_type_map.get(suffix, 'image/jpeg')

        with open(image_path, 'rb') as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')

        return image_data, media_type

    def _validate_vkn_checksum(self, vkn: str) -> bool:
        """
        Validate Turkish VKN (Vergi Kimlik Numarası) checksum.

        VKN is 10 digits. The last digit is a check digit.
        Algorithm:
        1. For each of first 9 digits, multiply by position weight
        2. Sum all results
        3. Check digit = sum mod 10
        """
        if not vkn or len(vkn) != 10:
            return False

        if not vkn.isdigit():
            return False

        try:
            digits = [int(d) for d in vkn]

            # Turkish VKN checksum algorithm
            total = 0
            for i in range(9):
                v1 = (digits[i] + 10 - (i + 1)) % 10
                v2 = v1 * (2 ** (10 - (i + 1))) % 9
                if v1 != 0 and v2 == 0:
                    v2 = 9
                total += v2

            check_digit = (10 - (total % 10)) % 10

            return digits[9] == check_digit
        except:
            return False

    def _validate_tckn_checksum(self, tckn: str) -> bool:
        """
        Validate Turkish TCKN (TC Kimlik Numarası) checksum.

        TCKN is 11 digits. First digit cannot be 0.
        """
        if not tckn or len(tckn) != 11:
            return False

        if not tckn.isdigit() or tckn[0] == '0':
            return False

        try:
            digits = [int(d) for d in tckn]

            # 10th digit check
            odd_sum = digits[0] + digits[2] + digits[4] + digits[6] + digits[8]
            even_sum = digits[1] + digits[3] + digits[5] + digits[7]
            d10 = (odd_sum * 7 - even_sum) % 10

            if digits[9] != d10:
                return False

            # 11th digit check
            total = sum(digits[:10])
            d11 = total % 10

            return digits[10] == d11
        except:
            return False

    def _detect_tesk(self, text: str) -> bool:
        """Detect TESK (Türkiye Esnaf ve Sanatkarları Konfederasyonu) in text."""
        tesk_patterns = [
            r'\bTESK\b',
            r'T\.E\.S\.K\.',
            r'Türkiye\s+Esnaf',
            r'Esnaf\s+ve\s+Sanatkar',
            r'ESNAF\s+VE\s+SANATKAR',
            r'Konfederasyon',
            r'KONFEDERASYON'
        ]

        for pattern in tesk_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_tevkifat(self, text: str, items: List[Dict]) -> Dict[str, Any]:
        """Detect withholding tax (tevkifat) in document."""
        result = {
            "tevkifatli": False,
            "tevkifat_orani": None,
            "tevkifat_tutari": None,
            "tespit_yontemi": None
        }

        # Check for tevkifat keywords
        tevkifat_patterns = [
            r'KDV\s+Tevkifat',
            r'Tevkifatlı',
            r'TEVKİFAT',
            r'withholding',
            r'9/10|7/10|5/10|3/10|2/10'
        ]

        for pattern in tevkifat_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["tevkifatli"] = True
                result["tespit_yontemi"] = "keyword"

                # Try to extract rate
                rate_match = re.search(r'(\d)/10', text)
                if rate_match:
                    result["tevkifat_orani"] = f"{rate_match.group(1)}/10"
                break

        return result

    def _match_account_code(self, text: str, items: List[Dict]) -> Dict[str, Any]:
        """Match account code from keywords."""
        text_lower = text.lower()
        matched_keywords = []
        account_info = None
        requires_kkeg = False
        tesk_kdv_override = False

        for keyword, info in ACCOUNT_KEYWORDS.items():
            if keyword.lower() in text_lower:
                matched_keywords.append(keyword)
                if account_info is None:
                    account_info = info.copy()

                # Check for KKEG requirement
                if info.get("kkeg"):
                    requires_kkeg = True

                # Check for TESK KDV override
                if keyword.lower() == "tesk":
                    tesk_kdv_override = True

        return {
            "matched_keywords": matched_keywords,
            "account_info": account_info,
            "requires_kkeg": requires_kkeg,
            "tesk_kdv_override": tesk_kdv_override
        }

    def _calculate_kkeg(self, brut_tutar: float, kdv_orani: int = 20) -> KKEGCalculation:
        """
        Calculate KKEG (Kanunen Kabul Edilmeyen Gider) for vehicle expenses.

        Formula from FLO document:
        - Matrah = Brüt / 1.20 (assuming 20% KDV)
        - Gider = Brüt * 0.70
        - KKEG Gider = Matrah * 0.30
        - KKEG KDV = KKEG Gider * 0.20
        """
        kdv_multiplier = 1 + (kdv_orani / 100)
        matrah = brut_tutar / kdv_multiplier
        gider_tutar = brut_tutar * 0.70
        kkeg_gider = matrah * 0.30
        kkeg_kdv = kkeg_gider * 0.20

        return KKEGCalculation(
            brut_tutar=brut_tutar,
            matrah=matrah,
            gider_tutar=gider_tutar,
            kkeg_gider=kkeg_gider,
            kkeg_kdv=kkeg_kdv
        )

    def _apply_category_kdv(self, financials: Dict[str, Any], account_match: Dict[str, Any]) -> None:
        """
        Apply category-based default KDV rates for Turkish receipts.

        KDV Rates by Category (from FLO document):
        - Yemek: %10 (restoran, lokanta)
        - Konaklama: %10 (otel, pansiyon)
        - Ulaşım: %20 (otobüs, metro - except TESK taxi)
        - Kırtasiye: %20
        - Temizlik: %20
        - Bakım/Onarım: %20
        - Kargo: %20
        - Akaryakıt: %20 (+ KKEG)
        - Binek Araç: %20 (+ KKEG)
        """
        # If KDV is already set from OCR, keep it
        if financials.get("kdv_orani") is not None and financials.get("kdv_orani") != 0:
            return

        account_info = account_match.get("account_info", {})
        if not account_info:
            return

        hesap_adi = account_info.get("hesap_adi", "").lower()

        # Category-based default KDV rates
        kdv_map = {
            "yemek": 10,
            "konaklama": 10,
            "otel": 10,
            "ulaşım": 20,
            "taksi": 20,  # Default if no TESK
            "kırtasiye": 20,
            "temizlik": 20,
            "bakım": 20,
            "onarım": 20,
            "kargo": 20,
            "akaryakıt": 20,
            "binek araç": 20,
            "içme suyu": 10,
        }

        for category, rate in kdv_map.items():
            if category in hesap_adi:
                if financials.get("kdv_orani") is None:
                    financials["kdv_orani"] = rate
                    financials["kdv_notu"] = f"{category.title()} kategorisi - varsayılan KDV %{rate}"
                break

    def _validate_buyer(self, alici: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate buyer against FLO companies.

        Rules:
        1. If VKN exists and matches FLO company -> APPROVED
        2. If TCKN exists (11 digits) -> Personal invoice -> REJECTED
        3. If name exists but no VKN and looks like personal name -> REJECTED
        4. No buyer info -> Retail receipt -> APPROVED

        TODO: Çalışan isim eşleştirme (Levenshtein Distance) ileride eklenecek
        """
        if not alici:
            return {
                "alici_gecerli": True,
                "alici_tipi": "PERAKENDE",
                "islem_durumu": "ONAYLANDI",
                "not": "Perakende satış - alıcı bilgisi yok"
            }

        vkn_or_tckn = alici.get("vkn_veya_tckn") or alici.get("vkn")
        alici_adi = alici.get("unvan_veya_ad") or alici.get("ad_soyad") or alici.get("firma_adi")

        # Case 1: VKN/TCKN exists
        if vkn_or_tckn:
            clean_id = re.sub(r'\D', '', str(vkn_or_tckn))

            if len(clean_id) == 11:
                # TCKN - Personal ID number
                tckn_valid = self._validate_tckn_checksum(clean_id)
                return {
                    "alici_gecerli": False,
                    "alici_tipi": "GERCEK_KISI",
                    "tckn_checksum_valid": tckn_valid,
                    "islem_durumu": "REDDEDILDI",
                    "red_sebebi": "Şahıs adına kesilmiş fatura - FLO masraf politikasına aykırı"
                }

            elif len(clean_id) == 10:
                # VKN - Company tax number
                vkn_valid = self._validate_vkn_checksum(clean_id)

                if clean_id in FLO_COMPANIES:
                    return {
                        "alici_gecerli": True,
                        "alici_flo_sirketi": FLO_COMPANIES[clean_id]["name"],
                        "alici_flo_kodu": FLO_COMPANIES[clean_id]["code"],
                        "alici_tipi": "TUZEL_KISI",
                        "vkn_checksum_valid": vkn_valid,
                        "islem_durumu": "ONAYLANDI"
                    }
                else:
                    return {
                        "alici_gecerli": False,
                        "alici_tipi": "TUZEL_KISI",
                        "vkn_checksum_valid": vkn_valid,
                        "islem_durumu": "REDDEDILDI",
                        "red_sebebi": "Alıcı FLO Grup şirketlerinden biri değil"
                    }
            else:
                return {
                    "alici_gecerli": False,
                    "islem_durumu": "REDDEDILDI",
                    "red_sebebi": f"Geçersiz VKN/TCKN formatı: {len(clean_id)} hane"
                }

        # Case 2: No VKN but name exists - Check if it's a personal name
        elif alici_adi:
            if is_personal_name(alici_adi):
                return {
                    "alici_gecerli": False,
                    "alici_tipi": "GERCEK_KISI",
                    "islem_durumu": "REDDEDILDI",
                    "red_sebebi": "Şahıs adına kesilmiş fatura (VKN yok, kişisel isim tespit edildi)"
                }
            else:
                # Looks like a company name but no VKN
                return {
                    "alici_gecerli": False,
                    "alici_tipi": "TUZEL_KISI",
                    "islem_durumu": "REDDEDILDI",
                    "red_sebebi": "Alıcı VKN bilgisi eksik"
                }

        # Case 3: No VKN and no name - Retail receipt
        else:
            return {
                "alici_gecerli": True,
                "alici_tipi": "PERAKENDE",
                "islem_durumu": "ONAYLANDI",
                "not": "Perakende satış fişi"
            }

    def _extract_text(self, image_path: str) -> OCRResult:
        """Extract text and structured data from image."""

        image_data, media_type = self._image_to_base64(image_path)

        message = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": self.USER_PROMPT
                        }
                    ]
                }
            ]
        )

        response_text = message.content[0].text.strip()

        # Parse JSON
        try:
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```") and not in_json:
                        in_json = True
                        continue
                    elif line.startswith("```") and in_json:
                        break
                    elif in_json:
                        json_lines.append(line)
                response_text = "\n".join(json_lines)

            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            return OCRResult(
                raw_text=response_text,
                confidence=0.0,
                fields={"error": f"JSON parse error: {str(e)}"},
                metadata={"raw_response": response_text}
            )

        # Extract components
        raw_text = data.get("raw_text", "")
        header = data.get("header", {})
        financials = data.get("financials", {})
        items = data.get("items", [])
        detected_language = data.get("detected_language", "TR")
        tesk_from_claude = data.get("tesk_detected", False)
        keywords_from_claude = data.get("keywords_found", [])

        # === POST-PROCESSING WITH BUSINESS RULES ===
        # Workflow:
        # 1. Dil Kontrolü: Türkçe mi?
        #    - Hayır: YURT DIŞI MASRAFI, KDV: 0, TESK kuralı pas
        #    - Evet: Adım 2
        # 2. Kategori Kontrolü: Taksi Fişi mi?
        #    - Evet + TESK var: KDV %0
        #    - Evet + TESK yok: KDV %20
        #    - Hayır: Kategori bazlı KDV tablosu

        # 1. Account Code Matching (önce kategoriyi belirle)
        account_match = self._match_account_code(raw_text, items)
        is_taxi = any(kw in ['taksi', 'taxi', 'tesk'] for kw in account_match.get("matched_keywords", []))

        # 2. TESK Detection
        tesk_detected = tesk_from_claude or self._detect_tesk(raw_text)

        # 3. Dil bazlı KDV mantığı
        is_turkish = detected_language == "TR"
        is_foreign = not is_turkish  # EN, DE, FR, IT, ES vb.

        # Yurt dışı masraf flag'i
        yurt_disi_masrafi = False
        tesk_uygulandı = False

        if is_foreign:
            # YURT DIŞI MASRAFI - TESK kuralını pas geç
            yurt_disi_masrafi = True
            financials["yurt_disi_masrafi"] = True
            # Yurt dışı belgelerde KDV genellikle belgede yazılı olan değer korunur
            # veya 0 olarak işlenir (yurt dışı harcama)
            if financials.get("kdv_orani") is None:
                financials["kdv_orani"] = 0
            financials["kdv_notu"] = "Yurt dışı masrafı - yerel vergi oranı uygulandı"

        elif is_turkish:
            # TÜRKİYE MASRAFI - Kategori bazlı KDV mantığı
            if is_taxi:
                # TAKSİ FİŞİ
                if tesk_detected:
                    # TESK var → KDV %0
                    financials["kdv_orani"] = 0
                    financials["kdv_tutari"] = 0
                    tesk_uygulandı = True
                    financials["kdv_notu"] = "TESK tespit edildi - KDV %0"
                else:
                    # TESK yok → KDV %20
                    if financials.get("kdv_orani") is None or financials.get("kdv_orani") == 0:
                        financials["kdv_orani"] = 20
                    financials["kdv_notu"] = "Taksi fişi TESK yok - KDV %20"
            else:
                # DİĞER KATEGORİLER - Varsayılan KDV oranları
                self._apply_category_kdv(financials, account_match)

        # 4. Tevkifat Detection
        tevkifat_result = self._detect_tevkifat(raw_text, items)
        if tevkifat_result["tevkifatli"]:
            header["tevkifatli"] = True
            financials["tevkifat_orani"] = tevkifat_result.get("tevkifat_orani")

        # 5. KKEG Calculation
        kkeg_calc = None
        if account_match.get("requires_kkeg") and financials.get("brut_tutar"):
            kdv_orani = financials.get("kdv_orani", 20)
            if isinstance(kdv_orani, (int, float)):
                kkeg_calc = self._calculate_kkeg(financials["brut_tutar"], int(kdv_orani))

        # 6. Buyer Validation
        alici = header.get("alici", {})
        buyer_validation = self._validate_buyer(alici)

        # 7. Seller VKN Validation
        satici = header.get("satici", {})
        satici_vkn = satici.get("vkn", "")
        clean_satici_vkn = re.sub(r'\D', '', str(satici_vkn))
        satici_vkn_valid = False
        if len(clean_satici_vkn) == 10:
            satici_vkn_valid = self._validate_vkn_checksum(clean_satici_vkn)

        # Build validation result
        validation = {
            **buyer_validation,
            "satici_vkn_gecerli": satici_vkn_valid,
            "tesk_tespit": tesk_detected,
            "tesk_uygulandı": tesk_uygulandı,
            "yurt_disi_masrafi": yurt_disi_masrafi,
            "taksi_fisi": is_taxi,
            "tespit_edilen_dil": detected_language,
            "esleslen_anahtar_kelimeler": account_match.get("matched_keywords", []),
            "hesap_bilgisi": account_match.get("account_info"),
            "tevkifat": tevkifat_result,
            "kkeg_hesaplama": kkeg_calc.to_dict() if kkeg_calc else None,
            "kdv_notu": financials.get("kdv_notu")
        }

        # Build fields
        fields = {
            "header": header,
            "financials": financials,
            "items": items,
            "validation": validation
        }

        # Confidence calculation
        confidence = 0.80
        if financials.get("brut_tutar"):
            confidence += 0.05
        if header.get("belge_no"):
            confidence += 0.05
        if satici_vkn_valid:
            confidence += 0.05
        if buyer_validation.get("alici_gecerli"):
            confidence += 0.05

        return OCRResult(
            raw_text=raw_text,
            lines=[{"text": line} for line in raw_text.split("\n") if line.strip()],
            confidence=min(confidence, 1.0),
            fields=fields,
            metadata={
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
                "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "detected_language": detected_language,
                "is_foreign": is_foreign,
                "yurt_disi_masrafi": yurt_disi_masrafi,
                "tesk_detected": tesk_detected,
                "tesk_applied": tesk_uygulandı,
                "is_taxi": is_taxi,
                "requires_kkeg": account_match.get("requires_kkeg", False)
            }
        )

    def extract_to_sap_json(self, image_path: str) -> Dict[str, Any]:
        """Extract SAP-compatible JSON directly from image."""
        result = self.extract(image_path)

        return {
            "header": result.fields.get("header", {}),
            "financials": result.fields.get("financials", {}),
            "items": result.fields.get("items", []),
            "validation": result.fields.get("validation", {}),
            "metadata": {
                **result.metadata,
                "processing_time": result.processing_time,
                "engine": self.name,
                "confidence": result.confidence
            }
        }
