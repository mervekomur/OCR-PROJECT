"""
[DEPRECATED] Test script for Employee Name Matching functionality.

Bu dosya şu anda devre dışı - çalışan eşleştirme özelliği TODO'ya alındı.
İleride SAP/HR entegrasyonu yapıldığında tekrar aktif edilecek.

Tests:
- Levenshtein Distance calculation
- Name similarity matching
- Personal invoice rejection rule
"""

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent / "src"))

from engines.claude_vision_engine import (
    levenshtein_distance,
    calculate_similarity,
    find_matching_employee,
    is_personal_name,
    EMPLOYEE_LIST
)


def test_levenshtein():
    """Test Levenshtein distance calculation."""
    print("=" * 60)
    print("LEVENSHTEIN DISTANCE TEST")
    print("=" * 60)

    test_cases = [
        ("hello", "hello", 0),
        ("hello", "hallo", 1),
        ("kitten", "sitting", 3),
        ("Ahmet", "Ahmat", 1),
        ("Mehmet Kaya", "Mehmat Kaya", 1),
        ("Ayşe Demir", "Ayse Demir", 1),
    ]

    for s1, s2, expected in test_cases:
        result = levenshtein_distance(s1, s2)
        status = "OK" if result == expected else "FAIL"
        print(f"  '{s1}' vs '{s2}': distance={result}, expected={expected} [{status}]")


def test_similarity():
    """Test similarity calculation."""
    print("\n" + "=" * 60)
    print("SIMILARITY TEST")
    print("=" * 60)

    test_cases = [
        ("Ahmet Yılmaz", "Ahmet Yılmaz", 1.0),
        ("Ahmet Yılmaz", "Ahmat Yilmaz", 0.8),  # ~83%
        ("Mehmet Kaya", "Mehmat Kaya", 0.9),   # ~91%
        ("Ayşe Demir", "Ayse Demır", 0.8),     # ~80%
        ("Ali Öztürk", "Ali Ozturk", 0.8),     # ~80%
        ("Fatma Çelik", "Fatna Celik", 0.8),   # ~80%
    ]

    for s1, s2, min_expected in test_cases:
        result = calculate_similarity(s1, s2)
        status = "OK" if result >= min_expected else "FAIL"
        print(f"  '{s1}' vs '{s2}': similarity={result:.1%}, min={min_expected:.0%} [{status}]")


def test_employee_matching():
    """Test employee matching with fuzzy search."""
    print("\n" + "=" * 60)
    print("EMPLOYEE MATCHING TEST")
    print("=" * 60)

    # OCR might misread names
    ocr_read_names = [
        "Ahmet Yilmaz",      # Missing Turkish char
        "Mehmat Kaya",       # Typo
        "Ayse Demır",        # Missing Turkish char
        "Fatna Çelik",       # Typo
        "Ali Ozturk",        # Missing Turkish char
        "Zeyneb Arslan",     # Typo
        "Mustafa Sahin",     # Missing Turkish char
        "John Smith",        # Not in list
        "Serkan Günes",      # Missing Turkish char
        "Büsra Yılmazer",    # Missing Turkish char
    ]

    print(f"\nÇalışan listesinde {len(EMPLOYEE_LIST)} kişi var.\n")

    for name in ocr_read_names:
        matched, score = find_matching_employee(name)
        if matched:
            print(f"  '{name}' -> '{matched}' ({score:.1%})")
        else:
            print(f"  '{name}' -> EŞLEŞMEDİ (en iyi: {score:.1%})")


def test_personal_name_detection():
    """Test personal name vs company name detection."""
    print("\n" + "=" * 60)
    print("PERSONAL NAME DETECTION TEST")
    print("=" * 60)

    test_cases = [
        ("Ahmet Yılmaz", True),
        ("Mehmet Kaya", True),
        ("Ali Veli", True),
        ("FLO Mağazacılık A.Ş.", False),
        ("Turuncu Ayakkabı SAN. TİC. LTD. ŞTİ.", False),
        ("AVIS RENT A CAR LLC", False),
        ("Europcar Deutschland GmbH", False),
        ("Halil NALÇAKAN", True),
        ("THE TEAM YEMİNLİ MALİ MÜŞAVİRLİK A.Ş.", False),
        ("KUZEYLER OTOMOTİV TURİZM TAŞIMACILIK", False),
        ("Fatih Çelik", True),
        ("ERHAN İLHAN ERTEK ELEKTRİK VE MÜHENDİSLİK SANAYİ", False),
    ]

    for name, expected in test_cases:
        result = is_personal_name(name)
        status = "OK" if result == expected else "FAIL"
        type_str = "Kişi" if result else "Şirket"
        expected_str = "Kişi" if expected else "Şirket"
        print(f"  '{name[:40]:<40}' -> {type_str} (beklenen: {expected_str}) [{status}]")


def test_real_scenario():
    """Test real-world OCR scenario."""
    print("\n" + "=" * 60)
    print("GERÇEK SENARYO TESTİ")
    print("=" * 60)

    # Simulate OCR reading a receipt with employee name
    print("\nSenaryo 1: Fatura çalışan adına kesilmiş (OCR hatalı okumuş)")
    ocr_name = "Mehmat Kaya"  # OCR typo
    matched, score = find_matching_employee(ocr_name)
    print(f"  OCR okuması: '{ocr_name}'")
    print(f"  Eşleşme: '{matched}' ({score:.1%})")
    print(f"  Sonuç: ŞAHIS FATURASI - REDDEDİLDİ")

    print("\nSenaryo 2: Fatura yabancı isme kesilmiş (listede yok)")
    ocr_name = "John Smith"
    matched, score = find_matching_employee(ocr_name)
    print(f"  OCR okuması: '{ocr_name}'")
    print(f"  Eşleşme: {matched} (en iyi skor: {score:.1%})")
    is_personal = is_personal_name(ocr_name)
    if is_personal:
        print(f"  Sonuç: ŞAHIS FATURASI (kişisel isim) - REDDEDİLDİ")
    else:
        print(f"  Sonuç: ŞİRKET FATURASI - VKN KONTROL")

    print("\nSenaryo 3: Fatura şirket adına kesilmiş")
    ocr_name = "FLO Mağazacılık A.Ş."
    matched, score = find_matching_employee(ocr_name)
    is_personal = is_personal_name(ocr_name)
    print(f"  OCR okuması: '{ocr_name}'")
    print(f"  Kişisel isim mi: {is_personal}")
    print(f"  Sonuç: ŞİRKET FATURASI - VKN KONTROL GEREKLİ")


if __name__ == "__main__":
    test_levenshtein()
    test_similarity()
    test_employee_matching()
    test_personal_name_detection()
    test_real_scenario()

    print("\n" + "=" * 60)
    print("TÜM TESTLER TAMAMLANDI")
    print("=" * 60)
