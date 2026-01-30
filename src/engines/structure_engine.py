"""
Structure Engine - Claude API Integration for FLO Masraf Modülü
Converts raw OCR text to SAP-compatible structured JSON using Claude's intelligence.

Features:
- FLO VKN validation (FLO Mağazacılık, Turuncu, FLO Teknoloji, FLO İç Dış)
- Buyer (Alıcı) validation - must be FLO company, not individual
- Slashed Zero & OCR error correction
- Mathematical validation (Brüt = Net + KDV)
- SAP-compatible JSON output (Header, Financials, Items)
- Noise filtering (OMU/UMU ghost text removal)
"""

import json
import os
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


# FLO Group Companies VKN Registry
FLO_COMPANIES = {
    "3880239429": {"name": "FLO Mağazacılık", "code": "FLO_MAG"},
    "8721503797": {"name": "Turuncu Ayakkabı", "code": "TURUNCU"},
    "3881618492": {"name": "FLO Teknoloji", "code": "FLO_TEK"},
    "3881765897": {"name": "FLO İç Dış Ticaret", "code": "FLO_IHR"},
}

# Valid KDV rates in Turkey
VALID_KDV_RATES = [0, 1, 8, 10, 18, 20]

# Noise patterns to filter (ghost text from paper back)
NOISE_PATTERNS = [
    r'\b[oO0][mM][uU]\b',      # OMU
    r'\b[uU][mM][uU]\b',       # UMU
    r'\bolu-umu\b',            # Common OCR ghost
    r'\b[xX]{3,}\b',           # XXX patterns
    r'^\d{1,2}$',              # Single/double digits alone
]

# Document type keywords
DOC_TYPE_KEYWORDS = {
    "FATURA": ["e-arşiv fatura", "e-fatura", "fatura no", "fatura"],
    "FIS": ["fiş no", "yazar kasa", "perakende satış"],
    "GIDER_PUSULASI": ["gider pusulası"],
    "BILGI_FISI": ["bilgi fişi", "e-arşiv bilgi"],
}

# Warning codes
WARNING_CODES = {
    "W001": "Fatura şahıs adına kesilmiş",
    "W002": "Alıcı FLO Grup şirketlerinden biri değil",
    "W003": "VKN formatı geçersiz",
    "W004": "Matematik tutarsızlığı",
    "W005": "KDV oranı geçersiz",
    "W006": "Zorunlu alan eksik",
    "W007": "Mükerrer belge şüphesi",
}


@dataclass
class Warning:
    """Structured warning for validation issues."""
    kod: str
    seviye: str  # ERROR, WARNING, INFO
    mesaj: str
    alan: Optional[str] = None
    deger: Optional[Any] = None
    beklenen: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "kod": self.kod,
            "seviye": self.seviye,
            "mesaj": self.mesaj
        }
        if self.alan:
            result["alan"] = self.alan
        if self.deger is not None:
            result["deger"] = self.deger
        if self.beklenen is not None:
            result["beklenen"] = self.beklenen
        return result


@dataclass
class SAPDocument:
    """SAP-compatible structured document."""

    # Header Information
    header: Dict[str, Any] = field(default_factory=dict)

    # Financial Details
    financials: Dict[str, Any] = field(default_factory=dict)

    # Line Items
    items: List[Dict[str, Any]] = field(default_factory=list)

    # Validation Results
    validation: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header,
            "financials": self.financials,
            "items": self.items,
            "validation": self.validation,
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def is_valid(self) -> bool:
        """Check if document passed all validations."""
        return self.validation.get("islem_durumu") != "REDDEDILDI"


class StructureEngine:
    """
    Uses Claude API to extract SAP-compatible structured data from OCR text.

    Implements FLO Masraf Modülü business rules:
    - Buyer (Alıcı) must be a FLO company (not individual)
    - VKN validation against FLO companies
    - OCR error correction (Slashed Zero: 0/O, 8/B, 5/S)
    - Mathematical validation (Brüt = Net + KDV)
    - Noise filtering
    """

    SYSTEM_PROMPT = """Sen FLO Grup şirketleri için masraf belgesi analiz uzmanısın.

## GÖREV
OCR'dan gelen fiş/fatura metnini analiz edip SAP'ye aktarılabilecek yapılandırılmış JSON üret.

## FLO GRUP ŞİRKETLERİ (Geçerli Alıcı VKN'leri)
- FLO Mağazacılık: 3880239429
- Turuncu Ayakkabı: 8721503797
- FLO Teknoloji: 3881618492
- FLO İç Dış Ticaret: 3881765897

## KRİTİK KURAL: ALICI KONTROLÜ
Fatura/fiş üzerinde iki taraf vardır:
1. SATICI: Mal/hizmet satan firma (tedarikçi)
2. ALICI: Faturanın kesildiği taraf (MÜŞTERİ bölümü)

ALICI mutlaka yukarıdaki FLO şirketlerinden biri olmalıdır.
- Alıcı kısmında şahıs ismi varsa → ŞAHIS ADINA KESİLMİŞ (kabul edilmez)
- Alıcı VKN 10 hane ise → Tüzel kişi (şirket)
- Alıcı TCKN 11 hane ise → Gerçek kişi (şahıs) → RED

## OCR HATA DÜZELTMELERİ (Slashed Zero)
- 0 ↔ O (sıfır/harf)
- 8 ↔ B, 5 ↔ S, 1 ↔ I/l/7, 6 ↔ G

## MATEMATİKSEL DOĞRULAMA
Brüt Tutar = KDV Hariç Tutar + KDV Tutarı

## GÜRÜLTÜ FİLTRELEME
Kağıdın arkasından sızan metinleri (OMU, UMU) yok say.

## GEÇERLİ KDV ORANLARI
%0, %1, %8, %10, %18, %20

SADECE JSON FORMATINDA YANIT VER."""

    USER_PROMPT_TEMPLATE = """Aşağıdaki OCR metnini analiz et ve SAP uyumlu JSON üret:

---OCR METNİ---
{ocr_text}
---OCR SONU---

ÖNEMLİ: Belgede SATICI ve ALICI (MÜŞTERİ) bilgilerini ayrı ayrı tespit et!

Yanıtını SADECE şu JSON formatında ver:
{{
    "header": {{
        "belge_turu": "FATURA|FIS|GIDER_PUSULASI|BILGI_FISI",
        "seri_no": "Seri numarası veya null",
        "belge_no": "Belge/Fiş numarası",
        "belge_tarihi": "DD.MM.YYYY formatında",
        "satici": {{
            "vkn": "Satıcı VKN (10 hane)",
            "firma_adi": "Satıcı firma adı",
            "adres": "Satıcı adresi veya null"
        }},
        "alici": {{
            "vkn_veya_tckn": "Alıcı VKN (10 hane) veya TCKN (11 hane)",
            "tip": "TUZEL_KISI|GERCEK_KISI",
            "unvan_veya_ad": "Şirket ünvanı veya şahıs adı",
            "adres": "Alıcı adresi veya null"
        }}
    }},
    "financials": {{
        "brut_tutar": Brüt toplam (sayı),
        "kdv_haric_tutar": KDV hariç tutar (sayı),
        "kdv_orani": KDV yüzdesi (0, 1, 8, 10, 18, 20),
        "kdv_tutari": KDV miktarı (sayı),
        "para_birimi": "TRY",
        "odeme_sekli": "NAKIT|KREDI_KARTI|HAVALE|null"
    }},
    "items": [
        {{
            "aciklama": "Ürün/hizmet açıklaması",
            "miktar": Adet,
            "birim_fiyat": Birim fiyat,
            "tutar": Satır tutarı
        }}
    ],
    "validation": {{
        "duzeltmeler": ["Yapılan OCR düzeltmeleri"]
    }}
}}"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize Structure Engine.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Claude model to use
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if anthropic package is installed."""
        try:
            import anthropic
            return True
        except ImportError:
            return False

    def _ensure_client(self):
        """Initialize Anthropic client if needed."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key not found. "
                    "Set ANTHROPIC_API_KEY environment variable "
                    "or provide api_key parameter."
                )
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)

    def _preprocess_text(self, text: str) -> str:
        """Remove noise patterns from OCR text."""
        cleaned = text
        for pattern in NOISE_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        return cleaned.strip()

    def _validate_buyer(self, alici: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate buyer (alıcı) information.

        Rules:
        - Buyer must be a FLO company
        - If TCKN (11 digits) → Individual (rejected)
        - If VKN (10 digits) not in FLO list → Rejected
        """
        warnings = []

        if not alici:
            return {
                "alici_gecerli": False,
                "alici_flo_sirketi": None,
                "alici_tipi": None,
                "islem_durumu": "REDDEDILDI",
                "warnings": [Warning(
                    kod="W006",
                    seviye="ERROR",
                    mesaj="Alıcı bilgisi bulunamadı",
                    alan="header.alici"
                ).to_dict()]
            }

        vkn_or_tckn = alici.get("vkn_veya_tckn") or alici.get("vkn") or alici.get("tckn")
        alici_tip = (alici.get("tip") or "").upper()
        alici_ad = alici.get("unvan_veya_ad") or alici.get("firma_adi") or alici.get("ad_soyad")

        # Clean the ID number
        clean_id = re.sub(r'\D', '', str(vkn_or_tckn)) if vkn_or_tckn else ""

        # Determine if individual or company based on ID length
        if len(clean_id) == 11:
            # TCKN - Individual person
            return {
                "alici_gecerli": False,
                "alici_flo_sirketi": None,
                "alici_tipi": "GERCEK_KISI",
                "islem_durumu": "REDDEDILDI",
                "red_sebebi": "Şahıs adına kesilmiş fatura - FLO masraf politikasına aykırı",
                "warnings": [Warning(
                    kod="W001",
                    seviye="ERROR",
                    mesaj="Fatura şahıs adına kesilmiş. Personel veya şahıs adına kesilen faturalar kabul edilmez.",
                    alan="header.alici.vkn_veya_tckn",
                    deger=f"TCKN: {clean_id[:3]}***{clean_id[-2:]}"
                ).to_dict()]
            }

        elif len(clean_id) == 10:
            # VKN - Company
            if clean_id in FLO_COMPANIES:
                return {
                    "alici_gecerli": True,
                    "alici_flo_sirketi": FLO_COMPANIES[clean_id]["name"],
                    "alici_flo_kodu": FLO_COMPANIES[clean_id]["code"],
                    "alici_tipi": "TUZEL_KISI",
                    "islem_durumu": "ONAYLANDI",
                    "warnings": []
                }
            else:
                return {
                    "alici_gecerli": False,
                    "alici_flo_sirketi": None,
                    "alici_tipi": "TUZEL_KISI",
                    "islem_durumu": "REDDEDILDI",
                    "red_sebebi": "Alıcı FLO Grup şirketlerinden biri değil",
                    "warnings": [Warning(
                        kod="W002",
                        seviye="ERROR",
                        mesaj="Alıcı FLO Grup şirketlerinden biri değil. Fatura FLO Mağazacılık, Turuncu Ayakkabı, FLO Teknoloji veya FLO İç Dış Ticaret adına kesilmiş olmalıdır.",
                        alan="header.alici.vkn_veya_tckn",
                        deger=clean_id,
                        beklenen=list(FLO_COMPANIES.keys())
                    ).to_dict()]
                }
        else:
            # Invalid ID format
            return {
                "alici_gecerli": False,
                "alici_flo_sirketi": None,
                "alici_tipi": None,
                "islem_durumu": "REDDEDILDI",
                "warnings": [Warning(
                    kod="W003",
                    seviye="ERROR",
                    mesaj=f"Geçersiz VKN/TCKN formatı. 10 hane (VKN) veya 11 hane (TCKN) olmalıdır.",
                    alan="header.alici.vkn_veya_tckn",
                    deger=vkn_or_tckn,
                    beklenen="10 veya 11 haneli numara"
                ).to_dict()]
            }

    def _validate_seller_vkn(self, satici: Dict[str, Any]) -> Dict[str, Any]:
        """Validate seller VKN format."""
        if not satici:
            return {"satici_vkn_gecerli": False, "error": "Satıcı bilgisi bulunamadı"}

        vkn = satici.get("vkn")
        if not vkn:
            return {"satici_vkn_gecerli": False, "error": "Satıcı VKN bulunamadı"}

        clean_vkn = re.sub(r'\D', '', str(vkn))

        if len(clean_vkn) == 10:
            return {"satici_vkn_gecerli": True, "satici_vkn": clean_vkn}
        else:
            return {"satici_vkn_gecerli": False, "error": f"Geçersiz satıcı VKN: {clean_vkn}"}

    def _validate_math(self, financials: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mathematical consistency."""
        brut = financials.get("brut_tutar")
        net = financials.get("kdv_haric_tutar")
        kdv = financials.get("kdv_tutari")

        if brut is None:
            return {
                "matematik_tutarli": False,
                "warning": Warning(
                    kod="W006",
                    seviye="WARNING",
                    mesaj="Brüt tutar bulunamadı",
                    alan="financials.brut_tutar"
                ).to_dict()
            }

        if net is not None and kdv is not None:
            expected_brut = net + kdv
            tolerance = 0.02 * brut if brut > 0 else 1  # 2% tolerance

            if abs(brut - expected_brut) <= tolerance:
                return {"matematik_tutarli": True}
            else:
                return {
                    "matematik_tutarli": False,
                    "warning": Warning(
                        kod="W004",
                        seviye="WARNING",
                        mesaj=f"Matematik tutarsızlığı: {net} + {kdv} = {net+kdv} ≠ {brut}",
                        alan="financials",
                        deger={"brut": brut, "net": net, "kdv": kdv},
                        beklenen=expected_brut
                    ).to_dict()
                }

        return {"matematik_tutarli": True}

    def extract_structured_data(self, ocr_text: str) -> SAPDocument:
        """
        Extract SAP-compatible structured data from OCR text using Claude.

        Args:
            ocr_text: Raw text from OCR engine

        Returns:
            SAPDocument with extracted and validated fields
        """
        self._ensure_client()

        # Preprocess text
        cleaned_text = self._preprocess_text(ocr_text)

        user_prompt = self.USER_PROMPT_TEMPLATE.format(ocr_text=cleaned_text)

        message = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self.SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        # Extract response text
        response_text = message.content[0].text.strip()

        # Parse JSON response
        try:
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            return SAPDocument(
                validation={"error": f"JSON parse hatası: {str(e)}"},
                metadata={"raw_response": response_text}
            )

        # Extract parts
        header = data.get("header", {})
        financials = data.get("financials", {})
        items = data.get("items", [])
        validation = data.get("validation", {})

        # Initialize warnings list
        all_warnings = []

        # 1. Validate BUYER (Alıcı) - CRITICAL
        alici = header.get("alici", {})
        buyer_validation = self._validate_buyer(alici)

        validation["alici_gecerli"] = buyer_validation.get("alici_gecerli", False)
        validation["alici_flo_sirketi"] = buyer_validation.get("alici_flo_sirketi")
        validation["alici_tipi"] = buyer_validation.get("alici_tipi")

        if buyer_validation.get("warnings"):
            all_warnings.extend(buyer_validation["warnings"])

        # 2. Validate SELLER VKN
        satici = header.get("satici", {})
        seller_validation = self._validate_seller_vkn(satici)
        validation["satici_vkn_gecerli"] = seller_validation.get("satici_vkn_gecerli", False)

        # 3. Math validation
        math_result = self._validate_math(financials)
        validation["matematik_tutarli"] = math_result.get("matematik_tutarli", True)
        if math_result.get("warning"):
            all_warnings.append(math_result["warning"])

        # 4. Set final status
        if buyer_validation.get("islem_durumu") == "REDDEDILDI":
            validation["islem_durumu"] = "REDDEDILDI"
            validation["red_sebebi"] = buyer_validation.get("red_sebebi", "Alıcı doğrulaması başarısız")
        else:
            validation["islem_durumu"] = "ONAYLANDI"

        # Add all warnings
        validation["uyarilar"] = all_warnings

        return SAPDocument(
            header=header,
            financials=financials,
            items=items,
            validation=validation,
            metadata={
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
                "ocr_length": len(ocr_text)
            }
        )

    def process_vision_result(self, vision_result) -> SAPDocument:
        """
        Process GoogleVisionEngine result directly.

        Args:
            vision_result: OCRResult from GoogleVisionEngine

        Returns:
            SAPDocument with extracted fields
        """
        return self.extract_structured_data(vision_result.raw_text)


def extract_receipt_data(ocr_text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to extract structured data from OCR text.

    Args:
        ocr_text: Raw text from OCR
        api_key: Optional Anthropic API key

    Returns:
        Dictionary with SAP-compatible structured data
    """
    engine = StructureEngine(api_key=api_key)
    result = engine.extract_structured_data(ocr_text)
    return result.to_dict()
