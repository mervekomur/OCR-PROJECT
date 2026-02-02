"""
Tax Calculator with Mathematical Heuristics - FLO Masraf Modulu

Implements:
1. Reverse Calculation: KDV from Total using %20 or %10 assumptions
2. Cross-Check: Match calculated KDV with receipt numbers (95% tolerance)
3. Smart Defaulting: Auto-assign KDV based on account code
4. Semantic Completion: Complete partial/corrupted words from keyword list

Reference: FLO Document Section 7.1 and 7.2
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


# =============================================================================
# ACCOUNT CODE KEYWORDS (66 entries from FLO document)
# Used for both account matching and semantic completion
# =============================================================================
ACCOUNT_KEYWORDS = {
    # Yemek Giderleri - KDV %10
    "yemek": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001", "default_kdv": 10},
    "restoran": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001", "default_kdv": 10},
    "lokanta": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001", "default_kdv": 10},
    "cafe": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001", "default_kdv": 10},
    "kahvalti": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001", "default_kdv": 10},
    "kahvaltı": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001", "default_kdv": 10},
    "menu": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001", "default_kdv": 10},
    "menü": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001", "default_kdv": 10},
    "yiyecek": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001", "default_kdv": 10},
    "icecek": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001", "default_kdv": 10},
    "içecek": {"hesap_adi": "Yemek Gideri", "hesap_kodu": "760.01.001", "default_kdv": 10},

    # Taksi Giderleri - KDV %0 (TESK) or %20
    "taksi": {"hesap_adi": "Taksi Gideri", "hesap_kodu": "760.02.001", "default_kdv": 20},
    "taxi": {"hesap_adi": "Taksi Gideri", "hesap_kodu": "760.02.001", "default_kdv": 20},
    "tesk": {"hesap_adi": "Taksi Gideri (TESK)", "hesap_kodu": "760.02.001", "default_kdv": 0},

    # Diger Ulasim - KDV %20
    "otobus": {"hesap_adi": "Ulasim Gideri", "hesap_kodu": "760.02.002", "default_kdv": 20},
    "otobüs": {"hesap_adi": "Ulasim Gideri", "hesap_kodu": "760.02.002", "default_kdv": 20},
    "havas": {"hesap_adi": "Ulasim Gideri", "hesap_kodu": "760.02.002", "default_kdv": 20},
    "havaş": {"hesap_adi": "Ulasim Gideri", "hesap_kodu": "760.02.002", "default_kdv": 20},
    "havaist": {"hesap_adi": "Ulasim Gideri", "hesap_kodu": "760.02.002", "default_kdv": 20},
    "dolmus": {"hesap_adi": "Ulasim Gideri", "hesap_kodu": "760.02.002", "default_kdv": 20},
    "dolmuş": {"hesap_adi": "Ulasim Gideri", "hesap_kodu": "760.02.002", "default_kdv": 20},
    "vapur": {"hesap_adi": "Ulasim Gideri", "hesap_kodu": "760.02.002", "default_kdv": 20},
    "iett": {"hesap_adi": "Ulasim Gideri", "hesap_kodu": "760.02.002", "default_kdv": 20},
    "metro": {"hesap_adi": "Ulasim Gideri", "hesap_kodu": "760.02.002", "default_kdv": 20},
    "metrobus": {"hesap_adi": "Ulasim Gideri", "hesap_kodu": "760.02.002", "default_kdv": 20},
    "metrobüs": {"hesap_adi": "Ulasim Gideri", "hesap_kodu": "760.02.002", "default_kdv": 20},
    "istanbulkart": {"hesap_adi": "Ulasim Gideri", "hesap_kodu": "760.02.002", "default_kdv": 20},
    "marmaray": {"hesap_adi": "Ulasim Gideri", "hesap_kodu": "760.02.002", "default_kdv": 20},

    # Arac Kiralama / Binek Arac - KDV %20 + KKEG
    "arac kiralama": {"hesap_adi": "Binek Arac Gideri", "hesap_kodu": "760.02.003", "default_kdv": 20, "kkeg": True},
    "araç kiralama": {"hesap_adi": "Binek Arac Gideri", "hesap_kodu": "760.02.003", "default_kdv": 20, "kkeg": True},
    "rent a car": {"hesap_adi": "Binek Arac Gideri", "hesap_kodu": "760.02.003", "default_kdv": 20, "kkeg": True},
    "europcar": {"hesap_adi": "Binek Arac Gideri", "hesap_kodu": "760.02.003", "default_kdv": 20, "kkeg": True},
    "avis": {"hesap_adi": "Binek Arac Gideri", "hesap_kodu": "760.02.003", "default_kdv": 20, "kkeg": True},
    "hertz": {"hesap_adi": "Binek Arac Gideri", "hesap_kodu": "760.02.003", "default_kdv": 20, "kkeg": True},
    "budget": {"hesap_adi": "Binek Arac Gideri", "hesap_kodu": "760.02.003", "default_kdv": 20, "kkeg": True},
    "enterprise": {"hesap_adi": "Binek Arac Gideri", "hesap_kodu": "760.02.003", "default_kdv": 20, "kkeg": True},

    # Akaryakit - KDV %20 + KKEG
    "benzin": {"hesap_adi": "Akaryakit Gideri", "hesap_kodu": "760.02.004", "default_kdv": 20, "kkeg": True},
    "motorin": {"hesap_adi": "Akaryakit Gideri", "hesap_kodu": "760.02.004", "default_kdv": 20, "kkeg": True},
    "lpg": {"hesap_adi": "Akaryakit Gideri", "hesap_kodu": "760.02.004", "default_kdv": 20, "kkeg": True},
    "akaryakit": {"hesap_adi": "Akaryakit Gideri", "hesap_kodu": "760.02.004", "default_kdv": 20, "kkeg": True},
    "akaryakıt": {"hesap_adi": "Akaryakit Gideri", "hesap_kodu": "760.02.004", "default_kdv": 20, "kkeg": True},
    "petrol": {"hesap_adi": "Akaryakit Gideri", "hesap_kodu": "760.02.004", "default_kdv": 20, "kkeg": True},
    "shell": {"hesap_adi": "Akaryakit Gideri", "hesap_kodu": "760.02.004", "default_kdv": 20, "kkeg": True},
    "opet": {"hesap_adi": "Akaryakit Gideri", "hesap_kodu": "760.02.004", "default_kdv": 20, "kkeg": True},
    "bp": {"hesap_adi": "Akaryakit Gideri", "hesap_kodu": "760.02.004", "default_kdv": 20, "kkeg": True},
    "total": {"hesap_adi": "Akaryakit Gideri", "hesap_kodu": "760.02.004", "default_kdv": 20, "kkeg": True},
    "petrol ofisi": {"hesap_adi": "Akaryakit Gideri", "hesap_kodu": "760.02.004", "default_kdv": 20, "kkeg": True},

    # Kirtasiye - KDV %20
    "kirtasiye": {"hesap_adi": "Kirtasiye Gideri", "hesap_kodu": "760.03.001", "default_kdv": 20},
    "kırtasiye": {"hesap_adi": "Kirtasiye Gideri", "hesap_kodu": "760.03.001", "default_kdv": 20},
    "kalem": {"hesap_adi": "Kirtasiye Gideri", "hesap_kodu": "760.03.001", "default_kdv": 20},
    "defter": {"hesap_adi": "Kirtasiye Gideri", "hesap_kodu": "760.03.001", "default_kdv": 20},
    "kagit": {"hesap_adi": "Kirtasiye Gideri", "hesap_kodu": "760.03.001", "default_kdv": 20},
    "kağıt": {"hesap_adi": "Kirtasiye Gideri", "hesap_kodu": "760.03.001", "default_kdv": 20},
    "toner": {"hesap_adi": "Kirtasiye Gideri", "hesap_kodu": "760.03.001", "default_kdv": 20},
    "kartus": {"hesap_adi": "Kirtasiye Gideri", "hesap_kodu": "760.03.001", "default_kdv": 20},
    "kartuş": {"hesap_adi": "Kirtasiye Gideri", "hesap_kodu": "760.03.001", "default_kdv": 20},

    # Konaklama - KDV %10
    "otel": {"hesap_adi": "Konaklama Gideri", "hesap_kodu": "760.04.001", "default_kdv": 10},
    "hotel": {"hesap_adi": "Konaklama Gideri", "hesap_kodu": "760.04.001", "default_kdv": 10},
    "konaklama": {"hesap_adi": "Konaklama Gideri", "hesap_kodu": "760.04.001", "default_kdv": 10},
    "pansiyon": {"hesap_adi": "Konaklama Gideri", "hesap_kodu": "760.04.001", "default_kdv": 10},
    "motel": {"hesap_adi": "Konaklama Gideri", "hesap_kodu": "760.04.001", "default_kdv": 10},
    "inn": {"hesap_adi": "Konaklama Gideri", "hesap_kodu": "760.04.001", "default_kdv": 10},
    "folio": {"hesap_adi": "Konaklama Gideri", "hesap_kodu": "760.04.001", "default_kdv": 10},

    # Temizlik - KDV %20
    "temizlik": {"hesap_adi": "Temizlik Gideri", "hesap_kodu": "760.05.001", "default_kdv": 20},
    "hijyen": {"hesap_adi": "Temizlik Gideri", "hesap_kodu": "760.05.001", "default_kdv": 20},
    "deterjan": {"hesap_adi": "Temizlik Gideri", "hesap_kodu": "760.05.001", "default_kdv": 20},

    # Icme Suyu - KDV %10
    "damacana": {"hesap_adi": "Icme Suyu Gideri", "hesap_kodu": "760.06.001", "default_kdv": 10},
    "su": {"hesap_adi": "Icme Suyu Gideri", "hesap_kodu": "760.06.001", "default_kdv": 10},

    # Bakim Onarim - KDV %20
    "bakim": {"hesap_adi": "Bakim Onarim Gideri", "hesap_kodu": "760.07.001", "default_kdv": 20},
    "bakım": {"hesap_adi": "Bakim Onarim Gideri", "hesap_kodu": "760.07.001", "default_kdv": 20},
    "onarim": {"hesap_adi": "Bakim Onarim Gideri", "hesap_kodu": "760.07.001", "default_kdv": 20},
    "onarım": {"hesap_adi": "Bakim Onarim Gideri", "hesap_kodu": "760.07.001", "default_kdv": 20},
    "tamir": {"hesap_adi": "Bakim Onarim Gideri", "hesap_kodu": "760.07.001", "default_kdv": 20},
    "servis": {"hesap_adi": "Bakim Onarim Gideri", "hesap_kodu": "760.07.001", "default_kdv": 20},

    # Kargo / Posta - KDV %20
    "kargo": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001", "default_kdv": 20},
    "ptt": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001", "default_kdv": 20},
    "mng": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001", "default_kdv": 20},
    "yurtici": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001", "default_kdv": 20},
    "yurtiçi": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001", "default_kdv": 20},
    "aras": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001", "default_kdv": 20},
    "ups": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001", "default_kdv": 20},
    "dhl": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001", "default_kdv": 20},
    "fedex": {"hesap_adi": "Kargo Gideri", "hesap_kodu": "760.08.001", "default_kdv": 20},

    # Tekstil - KDV %10
    "tekstil": {"hesap_adi": "Tekstil Gideri", "hesap_kodu": "760.09.001", "default_kdv": 10},
    "konfeksiyon": {"hesap_adi": "Tekstil Gideri", "hesap_kodu": "760.09.001", "default_kdv": 10},
    "giyim": {"hesap_adi": "Tekstil Gideri", "hesap_kodu": "760.09.001", "default_kdv": 10},
    "kıyafet": {"hesap_adi": "Tekstil Gideri", "hesap_kodu": "760.09.001", "default_kdv": 10},
    "kiyafet": {"hesap_adi": "Tekstil Gideri", "hesap_kodu": "760.09.001", "default_kdv": 10},

    # Elektrik/Elektronik - KDV %20
    "elektrik": {"hesap_adi": "Elektrik Gideri", "hesap_kodu": "760.10.001", "default_kdv": 20},
    "elektronik": {"hesap_adi": "Elektronik Gideri", "hesap_kodu": "760.10.002", "default_kdv": 20},
    "elektromekanik": {"hesap_adi": "Elektromekanik Gideri", "hesap_kodu": "760.10.003", "default_kdv": 20},
    "trafo": {"hesap_adi": "Trafo Gideri", "hesap_kodu": "760.10.004", "default_kdv": 20},
}

# Semantic completion: common partial words -> full words
PARTIAL_WORD_MAP = {
    "ismoni": "tekstil",
    "işmoni": "tekstil",
    "elektrome": "elektromekanik",
    "elektrom": "elektromekanik",
    "tekstil üretim pazarlam": "tekstil",
    "tekstil uretim pazarlam": "tekstil",
    "quality inn": "otel",
    "folio": "otel",
    "guest folio": "otel",
}


@dataclass
class KDVCalculationResult:
    """Result of KDV calculation with heuristics."""
    kdv_orani: Optional[int]
    kdv_tutari: Optional[float]
    matrah: Optional[float]
    method: str  # "ocr", "reverse_20", "reverse_10", "cross_check", "smart_default"
    confidence: float  # 0.0 - 1.0
    tahmini_atandi: bool  # Estimated flag
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kdv_orani": self.kdv_orani,
            "kdv_tutari": round(self.kdv_tutari, 2) if self.kdv_tutari else None,
            "matrah": round(self.matrah, 2) if self.matrah else None,
            "hesaplama_yontemi": self.method,
            "guven_orani": round(self.confidence, 2),
            "tahmini_atandi": self.tahmini_atandi,
            "notlar": self.notes
        }


class SemanticCompleter:
    """
    Completes partial/corrupted words using keyword dictionary.

    Example: "İşmoni" -> "tekstil"
    """

    def __init__(self, keywords: Dict[str, Any] = None, partial_map: Dict[str, str] = None):
        self.keywords = keywords or ACCOUNT_KEYWORDS
        self.partial_map = partial_map or PARTIAL_WORD_MAP

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio (0.0 - 1.0)."""
        if not s1 or not s2:
            return 0.0
        s1_norm = s1.lower().strip()
        s2_norm = s2.lower().strip()
        if s1_norm == s2_norm:
            return 1.0
        distance = self.levenshtein_distance(s1_norm, s2_norm)
        max_len = max(len(s1_norm), len(s2_norm))
        return 1.0 - (distance / max_len) if max_len > 0 else 1.0

    def complete_word(self, partial: str, threshold: float = 0.6) -> Tuple[Optional[str], float]:
        """
        Complete a partial/corrupted word to the closest keyword.

        Args:
            partial: The partial or corrupted word
            threshold: Minimum similarity threshold (default 0.6 = 60%)

        Returns:
            Tuple of (completed_word, similarity_score) or (None, 0.0)
        """
        if not partial:
            return None, 0.0

        partial_lower = partial.lower().strip()

        # First check exact matches in partial map
        for partial_key, full_word in self.partial_map.items():
            if partial_key in partial_lower:
                return full_word, 1.0

        # Then check similarity with keywords
        best_match = None
        best_score = 0.0

        for keyword in self.keywords.keys():
            score = self.similarity(partial_lower, keyword)
            if score > best_score:
                best_score = score
                best_match = keyword

        if best_score >= threshold:
            return best_match, best_score

        return None, best_score

    def complete_text(self, text: str, threshold: float = 0.6) -> Dict[str, Any]:
        """
        Analyze text and complete any partial words found.

        Returns dict with completions and matched account info.
        """
        if not text:
            return {"completions": [], "account_info": None}

        completions = []
        account_info = None

        # Split into words and check each
        words = text.lower().split()

        # First, check partial map for multi-word patterns
        text_lower = text.lower()
        for partial_key, full_word in self.partial_map.items():
            if partial_key in text_lower:
                completions.append({
                    "original": partial_key,
                    "completed": full_word,
                    "similarity": 1.0
                })
                if full_word in self.keywords and account_info is None:
                    account_info = self.keywords[full_word]

        # Then check individual words
        for word in words:
            if len(word) >= 4:  # Only check words with 4+ chars
                completed, score = self.complete_word(word, threshold)
                if completed and completed != word:
                    completions.append({
                        "original": word,
                        "completed": completed,
                        "similarity": round(score, 2)
                    })
                    if completed in self.keywords and account_info is None:
                        account_info = self.keywords[completed]

        return {
            "completions": completions,
            "account_info": account_info
        }


class TaxCalculator:
    """
    KDV Calculator with Mathematical Heuristics.

    Implements:
    1. Reverse Calculation from Total
    2. Cross-Check with receipt numbers
    3. Smart Defaulting based on account code
    """

    CROSS_CHECK_TOLERANCE = 0.05  # 5% tolerance (95% match)

    def __init__(self, keywords: Dict[str, Any] = None):
        self.keywords = keywords or ACCOUNT_KEYWORDS
        self.semantic_completer = SemanticCompleter(keywords)

    def reverse_calculate_kdv(self, brut_tutar: float, kdv_orani: int) -> Tuple[float, float]:
        """
        Reverse calculate KDV from brut tutar (total).

        Formula: KDV = Total - (Total / (1 + rate))

        Args:
            brut_tutar: Gross total amount
            kdv_orani: KDV rate (e.g., 20, 10, 8)

        Returns:
            Tuple of (matrah, kdv_tutari)
        """
        if not brut_tutar or brut_tutar <= 0:
            return 0.0, 0.0

        divisor = 1 + (kdv_orani / 100)
        matrah = brut_tutar / divisor
        kdv_tutari = brut_tutar - matrah

        return round(matrah, 2), round(kdv_tutari, 2)

    def cross_check_kdv(self, calculated_kdv: float, receipt_numbers: List[float]) -> Tuple[bool, Optional[float]]:
        """
        Check if calculated KDV matches any number on receipt within tolerance.

        Args:
            calculated_kdv: The calculated KDV amount
            receipt_numbers: List of numbers found on receipt

        Returns:
            Tuple of (matched, matched_number)
        """
        if not calculated_kdv or not receipt_numbers:
            return False, None

        for num in receipt_numbers:
            if num <= 0:
                continue
            # Check if within 5% tolerance (95% match)
            diff = abs(calculated_kdv - num)
            tolerance = calculated_kdv * self.CROSS_CHECK_TOLERANCE
            if diff <= tolerance:
                return True, num

        return False, None

    def extract_numbers_from_text(self, text: str) -> List[float]:
        """Extract all numbers (including decimals) from text."""
        if not text:
            return []

        # Pattern for numbers with optional decimals (handles both . and ,)
        pattern = r'\b\d+[.,]?\d*\b'
        matches = re.findall(pattern, text)

        numbers = []
        for match in matches:
            try:
                # Replace comma with dot for decimal
                num_str = match.replace(',', '.')
                num = float(num_str)
                if num > 0:
                    numbers.append(num)
            except ValueError:
                continue

        return numbers

    def match_account_code(self, text: str) -> Dict[str, Any]:
        """Match account code from text using keywords."""
        if not text:
            return {"matched": False, "account_info": None, "keywords_found": []}

        text_lower = text.lower()
        keywords_found = []
        account_info = None

        for keyword, info in self.keywords.items():
            if keyword in text_lower:
                keywords_found.append(keyword)
                if account_info is None:
                    account_info = info.copy()

        return {
            "matched": len(keywords_found) > 0,
            "account_info": account_info,
            "keywords_found": keywords_found
        }

    def calculate_kdv(
        self,
        brut_tutar: Optional[float],
        ocr_kdv_orani: Optional[int] = None,
        ocr_kdv_tutari: Optional[float] = None,
        raw_text: str = "",
        account_code: str = "",
        is_foreign: bool = False
    ) -> KDVCalculationResult:
        """
        Calculate KDV using mathematical heuristics.

        Priority:
        1. Use OCR-extracted KDV if valid
        2. Cross-check with reverse calculation
        3. Smart default based on account code
        4. Fallback to %20 assumption

        Args:
            brut_tutar: Gross total amount
            ocr_kdv_orani: KDV rate from OCR (if found)
            ocr_kdv_tutari: KDV amount from OCR (if found)
            raw_text: Raw text from receipt for number extraction
            account_code: Account code/category for smart defaulting
            is_foreign: Whether this is a foreign receipt

        Returns:
            KDVCalculationResult with calculated values and method
        """
        notes = []

        # Foreign receipts - use 0% or whatever is on document
        if is_foreign:
            return KDVCalculationResult(
                kdv_orani=ocr_kdv_orani or 0,
                kdv_tutari=ocr_kdv_tutari or 0,
                matrah=brut_tutar,
                method="yurt_disi",
                confidence=0.90,
                tahmini_atandi=False,
                notes=["Yurt disi masrafi - yerel vergi kurallari uygulanmaz"]
            )

        # No brut_tutar - can't calculate
        if not brut_tutar or brut_tutar <= 0:
            return KDVCalculationResult(
                kdv_orani=None,
                kdv_tutari=None,
                matrah=None,
                method="eksik_veri",
                confidence=0.0,
                tahmini_atandi=True,
                notes=["Brut tutar eksik - KDV hesaplanamadi"]
            )

        # Step 1: If OCR found both rate and amount, validate them
        if ocr_kdv_orani is not None and ocr_kdv_tutari is not None:
            expected_matrah, expected_kdv = self.reverse_calculate_kdv(brut_tutar, ocr_kdv_orani)

            # Check if OCR KDV matches calculation within tolerance
            if expected_kdv > 0:
                diff = abs(expected_kdv - ocr_kdv_tutari)
                tolerance = expected_kdv * self.CROSS_CHECK_TOLERANCE

                if diff <= tolerance:
                    notes.append(f"OCR KDV dogrulandi: %{ocr_kdv_orani}")
                    return KDVCalculationResult(
                        kdv_orani=ocr_kdv_orani,
                        kdv_tutari=ocr_kdv_tutari,
                        matrah=expected_matrah,
                        method="ocr_dogrulandi",
                        confidence=0.95,
                        tahmini_atandi=False,
                        notes=notes
                    )

        # Step 2: If only OCR rate found, calculate amount
        if ocr_kdv_orani is not None and ocr_kdv_orani in [0, 1, 8, 10, 18, 20]:
            matrah, kdv_tutari = self.reverse_calculate_kdv(brut_tutar, ocr_kdv_orani)
            notes.append(f"OCR'dan KDV orani: %{ocr_kdv_orani}, tutar hesaplandi")
            return KDVCalculationResult(
                kdv_orani=ocr_kdv_orani,
                kdv_tutari=kdv_tutari,
                matrah=matrah,
                method="ocr_oran",
                confidence=0.85,
                tahmini_atandi=False,
                notes=notes
            )

        # Step 3: Reverse calculation with cross-check
        receipt_numbers = self.extract_numbers_from_text(raw_text)

        # Try %20 first (most common)
        matrah_20, kdv_20 = self.reverse_calculate_kdv(brut_tutar, 20)
        matched_20, matched_num_20 = self.cross_check_kdv(kdv_20, receipt_numbers)

        if matched_20:
            notes.append(f"Reverse calculation %20: KDV {kdv_20} fis uzerindeki {matched_num_20} ile eslesti")
            return KDVCalculationResult(
                kdv_orani=20,
                kdv_tutari=kdv_20,
                matrah=matrah_20,
                method="cross_check_20",
                confidence=0.90,
                tahmini_atandi=False,
                notes=notes
            )

        # Try %10 (food, accommodation, textile)
        matrah_10, kdv_10 = self.reverse_calculate_kdv(brut_tutar, 10)
        matched_10, matched_num_10 = self.cross_check_kdv(kdv_10, receipt_numbers)

        if matched_10:
            notes.append(f"Reverse calculation %10: KDV {kdv_10} fis uzerindeki {matched_num_10} ile eslesti")
            return KDVCalculationResult(
                kdv_orani=10,
                kdv_tutari=kdv_10,
                matrah=matrah_10,
                method="cross_check_10",
                confidence=0.90,
                tahmini_atandi=False,
                notes=notes
            )

        # Try %8 (some services)
        matrah_8, kdv_8 = self.reverse_calculate_kdv(brut_tutar, 8)
        matched_8, matched_num_8 = self.cross_check_kdv(kdv_8, receipt_numbers)

        if matched_8:
            notes.append(f"Reverse calculation %8: KDV {kdv_8} fis uzerindeki {matched_num_8} ile eslesti")
            return KDVCalculationResult(
                kdv_orani=8,
                kdv_tutari=kdv_8,
                matrah=matrah_8,
                method="cross_check_8",
                confidence=0.90,
                tahmini_atandi=False,
                notes=notes
            )

        # Step 4: Smart defaulting based on account code/keywords
        account_match = self.match_account_code(raw_text)

        if account_match["matched"] and account_match["account_info"]:
            default_kdv = account_match["account_info"].get("default_kdv", 20)
            hesap_adi = account_match["account_info"].get("hesap_adi", "")

            matrah, kdv_tutari = self.reverse_calculate_kdv(brut_tutar, default_kdv)
            notes.append(f"Smart default: {hesap_adi} kategorisi -> KDV %{default_kdv}")
            notes.append(f"Eslesen anahtar kelimeler: {account_match['keywords_found']}")

            return KDVCalculationResult(
                kdv_orani=default_kdv,
                kdv_tutari=kdv_tutari,
                matrah=matrah,
                method="smart_default",
                confidence=0.75,
                tahmini_atandi=True,
                notes=notes
            )

        # Step 5: Semantic completion - try to complete partial words
        semantic_result = self.semantic_completer.complete_text(raw_text)

        if semantic_result["completions"] and semantic_result["account_info"]:
            default_kdv = semantic_result["account_info"].get("default_kdv", 20)
            hesap_adi = semantic_result["account_info"].get("hesap_adi", "")

            matrah, kdv_tutari = self.reverse_calculate_kdv(brut_tutar, default_kdv)

            for completion in semantic_result["completions"]:
                notes.append(f"Semantic completion: '{completion['original']}' -> '{completion['completed']}' ({completion['similarity']:.0%})")

            notes.append(f"Tamamlanan kategori: {hesap_adi} -> KDV %{default_kdv}")

            return KDVCalculationResult(
                kdv_orani=default_kdv,
                kdv_tutari=kdv_tutari,
                matrah=matrah,
                method="semantic_completion",
                confidence=0.70,
                tahmini_atandi=True,
                notes=notes
            )

        # Step 6: Fallback to %20 (most common in Turkey)
        matrah, kdv_tutari = self.reverse_calculate_kdv(brut_tutar, 20)
        notes.append("Fallback: KDV orani tespit edilemedi, varsayilan %20 uyguland")

        return KDVCalculationResult(
            kdv_orani=20,
            kdv_tutari=kdv_tutari,
            matrah=matrah,
            method="fallback_20",
            confidence=0.50,
            tahmini_atandi=True,
            notes=notes
        )


# Convenience function
def calculate_kdv_with_heuristics(
    brut_tutar: Optional[float],
    ocr_kdv_orani: Optional[int] = None,
    ocr_kdv_tutari: Optional[float] = None,
    raw_text: str = "",
    is_foreign: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to calculate KDV with all heuristics.

    Returns dict with all KDV information.
    """
    calculator = TaxCalculator()
    result = calculator.calculate_kdv(
        brut_tutar=brut_tutar,
        ocr_kdv_orani=ocr_kdv_orani,
        ocr_kdv_tutari=ocr_kdv_tutari,
        raw_text=raw_text,
        is_foreign=is_foreign
    )
    return result.to_dict()
