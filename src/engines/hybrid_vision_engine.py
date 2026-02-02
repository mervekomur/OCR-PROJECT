"""
Hybrid Vision OCR Engine - Google Vision + Claude
FLO Masraf Modülü v4

Workflow:
1. Google Vision API → OCR (metin çıkarma)
2. Claude API → Analiz ve yapılandırma (iş kuralları)

Bu hibrit yaklaşım:
- Google Vision'ın yüksek OCR doğruluğunu kullanır
- Claude'un akıllı analiz ve iş kuralı uygulamasını kullanır
"""

import os
import json
import re
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime

from .base import BaseOCREngine, OCRResult
from .google_vision_engine import GoogleVisionEngine

# Import tax calculator
try:
    from ..logic.tax_calculator import TaxCalculator, SemanticCompleter
    TAX_CALCULATOR_AVAILABLE = True
except ImportError:
    TAX_CALCULATOR_AVAILABLE = False


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
# LANGUAGE DETECTION PATTERNS
# =============================================================================
LANGUAGE_PATTERNS = {
    "TR": ["toplam", "kdv", "tarih", "fatura", "fiş", "tutar", "ödeme", "vergi"],
    "EN": ["total", "vat", "date", "invoice", "receipt", "amount", "payment", "tax"],
    "DE": ["gesamt", "mwst", "datum", "rechnung", "quittung", "betrag", "zahlung"],
    "FR": ["total", "tva", "date", "facture", "reçu", "montant", "paiement"],
    "IT": ["totale", "iva", "data", "fattura", "ricevuta", "importo", "pagamento"],
    "ES": ["total", "iva", "fecha", "factura", "recibo", "importe", "pago"],
}


class HybridVisionEngine(BaseOCREngine):
    """
    Hybrid OCR Engine: Google Vision + Claude

    Step 1: Google Vision API extracts raw text from image
    Step 2: Claude API analyzes and structures the text with business rules
    """

    name = "hybrid_vision"
    description = "Google Vision OCR + Claude Analysis - FLO Masraf Modülü"

    CLAUDE_SYSTEM_PROMPT = """Sen bir JSON formatlayıcısısın. Görevin OCR metnini yapılandırılmış JSON'a dönüştürmek.

## KRİTİK KURAL: SADECE METİNDEKİ VERİYİ KULLAN!
- Metinde OLMAYAN veriyi UYDURMA
- Bir değer metinde yoksa null yaz
- Tahmin yapma, sadece gördüğünü yaz

## SATICI vs ALICI AYRIMI
- SATICI: Faturanın üstündeki firma (mal/hizmet satan)
- ALICI: "Sayın", "Müşteri", "Alıcı" başlıkları altındaki kişi/firma

## VKN/TCKN KURALLARI
- VKN: 10 haneli sayı
- TCKN: 11 haneli sayı (0 ile başlamaz)
- Metinde görmezsen null yaz

## BELGE TÜRÜ
- "e-arşiv", "e-fatura", "fatura no" → FATURA
- "fiş no", "yazar kasa" → FIS
- "gider pusulası" → GIDER_PUSULASI

## KDV KURALLARI
- KDV oranını metinde gördüğün gibi yaz
- Metinde KDV oranı yoksa null yaz (sistem hesaplayacak)
- UYDURMA!

SADECE JSON DÖNDÜR, AÇIKLAMA YAPMA."""

    CLAUDE_USER_PROMPT = """Bu OCR metnini JSON'a dönüştür. SADECE metinde gördüğün değerleri yaz, UYDURMA!

OCR METNİ:
{ocr_text}

JSON:
{{
    "detected_language": "TR|EN|DE|FR|IT|ES",
    "header": {{
        "belge_turu": "FATURA|FIS|GIDER_PUSULASI|null",
        "belge_no": "metindeki belge/fiş numarası veya null",
        "belge_tarihi": "metindeki tarih veya null",
        "saat": "metindeki saat veya null",
        "satici": {{
            "vkn": "metindeki 10 haneli satıcı VKN veya null",
            "firma_adi": "metindeki satıcı adı veya null"
        }},
        "alici": {{
            "vkn_veya_tckn": "Sayın/Alıcı/Müşteri bölümündeki VKN/TCKN veya null",
            "unvan_veya_ad": "Alıcı adı veya null"
        }}
    }},
    "financials": {{
        "brut_tutar": metindeki toplam tutar (sayı) veya null,
        "kdv_haric_tutar": metindeki KDV hariç tutar veya null,
        "kdv_orani": metindeki KDV oranı (sayı) veya null,
        "kdv_tutari": metindeki KDV tutarı veya null,
        "para_birimi": "TRY|EUR|USD|CHF|GBP|DOP|AED"
    }},
    "tesk_detected": metinde TESK/Esnaf kelimesi var mı (true/false)
}}

HATIRLA: Metinde olmayan değer için null yaz, UYDURMA!"""

    def __init__(
        self,
        google_credentials_path: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        claude_model: str = "claude-sonnet-4-20250514"
    ):
        super().__init__()
        self.google_credentials_path = google_credentials_path
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.claude_model = claude_model
        self._google_engine = None
        self._claude_client = None

    @classmethod
    def _check_availability(cls) -> bool:
        try:
            from google.cloud import vision
            import anthropic
            return True
        except ImportError:
            return False

    def _initialize(self) -> None:
        # Initialize Google Vision
        self._google_engine = GoogleVisionEngine(
            credentials_path=self.google_credentials_path,
            language_hints=['tr', 'en', 'de', 'fr']
        )
        self._google_engine._ensure_initialized()

        # Initialize Claude
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        import anthropic
        self._claude_client = anthropic.Anthropic(api_key=self.anthropic_api_key)

    def _detect_language(self, text: str) -> str:
        """Detect document language from text."""
        text_lower = text.lower()
        scores = {}

        for lang, keywords in LANGUAGE_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[lang] = score

        if max(scores.values()) == 0:
            return "TR"  # Default to Turkish

        return max(scores, key=scores.get)

    def _detect_tesk(self, text: str) -> bool:
        """Detect TESK in text."""
        tesk_patterns = [
            r'\bTESK\b',
            r'T\.E\.S\.K\.',
            r'Türkiye\s+Esnaf',
            r'Esnaf\s+ve\s+Sanatkar',
        ]
        for pattern in tesk_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _validate_claude_output(self, data: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        """
        Validate that Claude's output values actually exist in raw_text.
        If a value doesn't exist in raw_text, set it to None.
        """
        raw_text_lower = raw_text.lower()
        raw_text_digits = re.findall(r'\d+[.,]?\d*', raw_text)

        financials = data.get("financials", {})

        # Validate brut_tutar exists in raw_text
        brut = financials.get("brut_tutar")
        if brut is not None:
            brut_str = str(brut).replace('.', ',')
            brut_str2 = str(brut).replace(',', '.')
            if brut_str not in raw_text and brut_str2 not in raw_text:
                # Try to find a close match
                found = False
                for num_str in raw_text_digits:
                    try:
                        num = float(num_str.replace(',', '.'))
                        if abs(num - brut) < 0.01:
                            found = True
                            break
                    except:
                        pass
                if not found:
                    financials["brut_tutar_validated"] = False

        # Validate KDV values
        kdv_oran = financials.get("kdv_orani")
        if kdv_oran is not None:
            # Check if KDV rate pattern exists
            kdv_patterns = [f"%{kdv_oran}", f"kdv {kdv_oran}", f"kdv%{kdv_oran}", f"kdv:{kdv_oran}"]
            if not any(p in raw_text_lower for p in kdv_patterns):
                financials["kdv_orani_validated"] = False

        data["financials"] = financials
        return data

    def _validate_vkn(self, vkn: str) -> bool:
        """Validate Turkish VKN checksum."""
        if not vkn or len(vkn) != 10 or not vkn.isdigit():
            return False
        try:
            digits = [int(d) for d in vkn]
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

    def _validate_buyer(self, alici: Dict[str, Any]) -> Dict[str, Any]:
        """Validate buyer against FLO companies."""
        if not alici:
            return {
                "alici_gecerli": True,
                "alici_tipi": "PERAKENDE",
                "islem_durumu": "ONAYLANDI",
                "not": "Perakende satış - alıcı bilgisi yok"
            }

        vkn_or_tckn = alici.get("vkn_veya_tckn") or alici.get("vkn")
        alici_adi = alici.get("unvan_veya_ad") or alici.get("firma_adi")

        if vkn_or_tckn:
            clean_id = re.sub(r'\D', '', str(vkn_or_tckn))

            if len(clean_id) == 11:
                return {
                    "alici_gecerli": False,
                    "alici_tipi": "GERCEK_KISI",
                    "islem_durumu": "REDDEDILDI",
                    "red_sebebi": "Şahıs adına kesilmiş fatura"
                }
            elif len(clean_id) == 10:
                if clean_id in FLO_COMPANIES:
                    return {
                        "alici_gecerli": True,
                        "alici_flo_sirketi": FLO_COMPANIES[clean_id]["name"],
                        "alici_tipi": "TUZEL_KISI",
                        "islem_durumu": "ONAYLANDI"
                    }
                else:
                    return {
                        "alici_gecerli": False,
                        "alici_tipi": "TUZEL_KISI",
                        "islem_durumu": "REDDEDILDI",
                        "red_sebebi": "Alıcı FLO Grup şirketlerinden biri değil"
                    }

        if alici_adi:
            # Check if it's a personal name
            company_indicators = ["A.Ş.", "LTD.", "ŞTİ.", "SAN.", "TİC.", "INC.", "LLC", "GMBH"]
            is_company = any(ind in alici_adi.upper() for ind in company_indicators)
            if not is_company and len(alici_adi.split()) <= 4:
                return {
                    "alici_gecerli": False,
                    "alici_tipi": "GERCEK_KISI",
                    "islem_durumu": "REDDEDILDI",
                    "red_sebebi": "Şahıs adına kesilmiş fatura"
                }

        return {
            "alici_gecerli": True,
            "alici_tipi": "PERAKENDE",
            "islem_durumu": "ONAYLANDI"
        }

    def _extract_text(self, image_path: str) -> OCRResult:
        """
        Extract and structure receipt data.

        Architecture:
        1. Google Vision API = Source of Truth (OCR metin çıkarma)
        2. Claude API = Formatter only (JSON yapılandırma, veri üretmez)
        3. KDV Heuristics = Works on Google's raw_text (matematiksel hesaplama)
        """
        # Step 1: Google Vision OCR (SOURCE OF TRUTH)
        google_result = self._google_engine.extract(image_path)
        ocr_text = google_result.raw_text  # This is our source of truth
        google_confidence = google_result.confidence

        if not ocr_text or len(ocr_text.strip()) < 10:
            return OCRResult(
                raw_text=ocr_text,
                confidence=0.0,
                fields={"error": "OCR text too short or empty"},
                metadata={"google_confidence": google_confidence}
            )

        # Step 2: Detect language
        detected_language = self._detect_language(ocr_text)
        is_foreign = detected_language != "TR"

        # Step 3: Detect TESK
        tesk_detected = self._detect_tesk(ocr_text)

        # Step 4: Claude Analysis
        prompt = self.CLAUDE_USER_PROMPT.format(ocr_text=ocr_text)

        message = self._claude_client.messages.create(
            model=self.claude_model,
            max_tokens=4096,
            system=self.CLAUDE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        # Parse Claude's JSON response
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

            # Validate Claude's output against raw_text
            data = self._validate_claude_output(data, ocr_text)

        except json.JSONDecodeError as e:
            return OCRResult(
                raw_text=ocr_text,
                confidence=google_confidence,
                fields={"error": f"Claude JSON parse error: {str(e)}"},
                metadata={
                    "google_confidence": google_confidence,
                    "claude_raw": response_text
                }
            )

        # Extract components
        header = data.get("header", {})
        financials = data.get("financials", {})
        items = data.get("items", [])
        claude_language = data.get("detected_language", detected_language)
        tesk_from_claude = data.get("tesk_detected", False)

        # Update language detection
        detected_language = claude_language if claude_language else detected_language
        is_foreign = detected_language != "TR"

        # Apply TESK rule
        tesk_detected = tesk_detected or tesk_from_claude
        is_taxi = any(kw in ocr_text.lower() for kw in ['taksi', 'taxi', 'tesk'])

        if is_foreign:
            financials["yurt_disi_masrafi"] = True
            if financials.get("kdv_orani") is None:
                financials["kdv_orani"] = 0
            financials["kdv_notu"] = "Yurt dışı masrafı"
        elif is_taxi:
            if tesk_detected:
                financials["kdv_orani"] = 0
                financials["kdv_tutari"] = 0
                financials["kdv_notu"] = "TESK tespit edildi - KDV %0"
            else:
                if financials.get("kdv_orani") is None:
                    financials["kdv_orani"] = 20
                financials["kdv_notu"] = "Taksi fişi TESK yok - KDV %20"

        # Step 5: Mathematical Heuristics for KDV
        # IMPORTANT: Heuristics work on Google Vision's raw_text (ocr_text), NOT Claude's output
        kdv_heuristics_result = None
        if TAX_CALCULATOR_AVAILABLE:
            brut_tutar = financials.get("brut_tutar")
            ocr_kdv_orani = financials.get("kdv_orani")
            ocr_kdv_tutari = financials.get("kdv_tutari")

            if brut_tutar and (ocr_kdv_orani is None or ocr_kdv_tutari is None):
                tax_calc = TaxCalculator()
                semantic = SemanticCompleter()
                # Semantic completion on Google Vision's raw_text
                semantic_result = semantic.complete_text(ocr_text)

                kdv_result = tax_calc.calculate_kdv(
                    brut_tutar=brut_tutar,
                    ocr_kdv_orani=ocr_kdv_orani,
                    ocr_kdv_tutari=ocr_kdv_tutari,
                    raw_text=ocr_text,
                    is_foreign=is_foreign
                )

                kdv_heuristics_result = kdv_result.to_dict()

                if kdv_result.kdv_orani is not None:
                    financials["kdv_orani"] = kdv_result.kdv_orani
                if kdv_result.kdv_tutari is not None:
                    financials["kdv_tutari"] = kdv_result.kdv_tutari
                if kdv_result.matrah is not None:
                    financials["kdv_matrah"] = kdv_result.matrah

                financials["kdv_hesaplama_yontemi"] = kdv_result.method
                financials["kdv_tahmini_atandi"] = kdv_result.tahmini_atandi

                if semantic_result["completions"]:
                    header["semantic_completions"] = semantic_result["completions"]

        # Step 6: Buyer Validation
        alici = header.get("alici", {})
        buyer_validation = self._validate_buyer(alici)

        # Build validation result
        validation = {
            **buyer_validation,
            "tesk_tespit": tesk_detected,
            "yurt_disi_masrafi": is_foreign,
            "taksi_fisi": is_taxi,
            "tespit_edilen_dil": detected_language,
            "kdv_heuristics": kdv_heuristics_result,
            "kdv_notu": financials.get("kdv_notu")
        }

        fields = {
            "header": header,
            "financials": financials,
            "items": items,
            "validation": validation
        }

        # Combined confidence
        confidence = (google_confidence * 0.4) + 0.5  # Base from Google + Claude boost
        if financials.get("brut_tutar"):
            confidence += 0.05
        if header.get("belge_no"):
            confidence += 0.05

        return OCRResult(
            raw_text=ocr_text,
            lines=google_result.lines,
            confidence=min(confidence, 1.0),
            fields=fields,
            metadata={
                "engine": "hybrid_vision",
                "google_confidence": google_confidence,
                "claude_model": self.claude_model,
                "detected_language": detected_language,
                "is_foreign": is_foreign,
                "tesk_detected": tesk_detected,
                "is_taxi": is_taxi,
                "timestamp": datetime.now().isoformat()
            }
        )

    def extract_to_sap_json(self, image_path: str) -> Dict[str, Any]:
        """Extract SAP-compatible JSON from image."""
        result = self.extract(image_path)

        return {
            "header": result.fields.get("header", {}),
            "financials": result.fields.get("financials", {}),
            "items": result.fields.get("items", []),
            "validation": result.fields.get("validation", {}),
            "metadata": {
                **result.metadata,
                "processing_time": result.processing_time,
                "confidence": result.confidence
            }
        }
