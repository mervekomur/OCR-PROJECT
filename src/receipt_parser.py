"""
Fis/Makbuz Parser Modulu
OCR ciktisindan yapilandirilmis veri cikarir.
Turkce ve uluslararasi fisler icin destek.
"""

import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime


# ============================================================
# OCR HATA DUZELTME (CLEANING)
# ============================================================

def clean_ocr_text(text: str) -> str:
    """
    OCR ciktisindaki yaygin hatalari duzeltir.

    Args:
        text: Ham OCR metni

    Returns:
        str: Temizlenmis metin
    """
    cleaned = text

    # Euro sembolu varyasyonlari
    cleaned = re.sub(r'[Є∈]', '€', cleaned)

    # Fazla bosluklari temizle
    cleaned = re.sub(r' +', ' ', cleaned)

    # Z vs % heuristik duzeltme
    cleaned = fix_z_percent_ocr_error(cleaned)

    return cleaned


def fix_z_percent_ocr_error(text: str) -> str:
    """
    OCR'in Z ve % karakterlerini karistirma hatasini duzeltir.

    Heuristik Kurallar:
    1. Vergi/oran baglami: Z + rakam -> % + rakam (Z20 -> %20)
    2. Urun kodu baglami: % + rakam -> korunur (model numarasi olabilir)
    3. Baglam tespiti: KDV, TVA, TAX, HT, TTC gibi anahtar kelimeler

    Args:
        text: Ham OCR metni

    Returns:
        str: Duzeltilmis metin
    """
    if not text:
        return text

    corrected = text

    # Vergi/oran baglam kelimeleri
    tax_context_keywords = [
        'KDV', 'TVA', 'TAX', 'VAT', 'HT', 'TTC',
        'TAUX', 'RATE', 'ORAN', 'VERGI', 'TOPKDV'
    ]

    # Baglam kontrolu: Eger satirda vergi kelimesi varsa Z->% duzelt
    text_upper = text.upper()
    is_tax_context = any(kw in text_upper for kw in tax_context_keywords)

    if is_tax_context:
        # Z + rakam -> % + rakam (ornek: Z20 -> %20, Z5.5 -> %5.5)
        corrected = re.sub(r'Z\s*(\d+[.,]?\d*)', r'%\1', corrected, flags=re.IGNORECASE)
        if corrected != text:
            print(f"[Heuristik] Z->% duzeltme (vergi baglami): {text} -> {corrected}")

    # Ozel durum: Satir basinda veya bosluktan sonra Z + rakam (bagimsiz oran)
    # Ornek: "Z20 KDV" veya "TVA Z5.5"
    corrected = re.sub(r'(?<![A-Za-z])Z(\d+[.,]?\d*)\s*(%|KDV|TVA|TAX)',
                       r'%\1 \2', corrected, flags=re.IGNORECASE)

    # Tek basina Z + rakam + % sembolu olmayan durumlar (oran gibi gorunen)
    # Ornek: "Z20" satirda tek basina ve rakamla bitiyorsa
    if re.match(r'^Z\d+[.,]?\d*$', corrected.strip(), re.IGNORECASE):
        # Bu buyuk ihtimalle bir oran (%20 gibi)
        corrected = re.sub(r'^Z(\d+[.,]?\d*)$', r'%\1', corrected.strip(), flags=re.IGNORECASE)
        if corrected != text.strip():
            print(f"[Heuristik] Z->% duzeltme (tek basina oran): {text} -> {corrected}")

    return corrected


def is_product_code_context(text: str, full_line: str = None) -> bool:
    """
    Metnin urun kodu baglaminda olup olmadigini kontrol eder.

    Args:
        text: Kontrol edilecek metin
        full_line: Tam satir (baglam icin)

    Returns:
        bool: Urun kodu baglaminda mi?
    """
    # Urun kodu gostergeleri
    product_indicators = [
        r'[A-Z]{2,}\d+[A-Z]*',  # ABC123, AB12C gibi
        r'PLUS', r'PRO', r'MAX', r'LITE',
        r'MODEL', r'REF', r'SKU', r'EAN'
    ]

    for pattern in product_indicators:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def clean_business_name(name: str) -> str:
    """
    Isletme adindaki OCR hatalarini duzeltir.

    Args:
        name: Ham isletme adi

    Returns:
        str: Temizlenmis isletme adi
    """
    if not name:
        return name

    cleaned = name

    # Bilinen OCR hatalari - kelime ici yanlis bosluklar
    ocr_fixes = {
        'Utr echt': 'Utrecht',
        'Amst erdam': 'Amsterdam',
        'Istan bul': 'Istanbul',
        'Ank ara': 'Ankara',
        'Izm ir': 'Izmir',
        'Hilt on': 'Hilton',
        'Star bucks': 'Starbucks',
        'McDon alds': 'McDonalds',
        'Burg er': 'Burger',
    }

    for wrong, correct in ocr_fixes.items():
        cleaned = cleaned.replace(wrong, correct)

    # Tek harfli parcalar arasindaki bosluklari kaldir (orn: "H i l t o n" -> "Hilton")
    # Ard arda gelen tek harfleri birlestir
    cleaned = re.sub(r'\b(\w) (\w) (\w)\b', r'\1\2\3', cleaned)
    cleaned = re.sub(r'\b(\w) (\w)\b', r'\1\2', cleaned)

    # Fazla bosluklari temizle
    cleaned = re.sub(r' +', ' ', cleaned).strip()

    if cleaned != name:
        print(f"[Heuristik] Isletme adi duzeltildi: {name} -> {cleaned}")

    return cleaned


def clean_product_name(name: str) -> str:
    """
    Urun adindaki OCR hatalarini duzeltir.

    Args:
        name: Ham urun adi

    Returns:
        str: Temizlenmis urun adi
    """
    if not name:
        return name

    cleaned = name

    # Bilinen OCR hatalari - urun adlari
    ocr_fixes = {
        # Unlu Mamuller varyasyonlari
        'uNlu Hamuller': 'Unlu Mamuller',
        'uNlu Mamuller': 'Unlu Mamuller',
        'Unlu Hamuller': 'Unlu Mamuller',
        'UNLU HAMULLER': 'UNLU MAMULLER',
        'uNLU HAMULLER': 'UNLU MAMULLER',
        # Diger yaygin urunler
        'Ekmek': 'Ekmek',
        'EKHEK': 'EKMEK',
        'SlMlT': 'SIMIT',
        'SIHIT': 'SIMIT',
        'POGACA': 'POGACA',
        'P0GACA': 'POGACA',
    }

    for wrong, correct in ocr_fixes.items():
        if cleaned == wrong:
            cleaned = correct
            break

    # Genel duzeltmeler
    # Bas harfi buyut, gerisi kucuk (title case)
    if cleaned.isupper() or cleaned.islower():
        cleaned = cleaned.title()

    # Fazla bosluklari temizle
    cleaned = re.sub(r' +', ' ', cleaned).strip()

    if cleaned != name:
        print(f"[Heuristik] Urun adi duzeltildi: {name} -> {cleaned}")

    return cleaned


def clean_date_string(date_str: str) -> Dict[str, Any]:
    """
    Tarih stringindeki hatalari duzeltir.

    Returns:
        Dict: {
            'cleaned': Duzeltilmis tarih,
            'original': Orijinal OCR degeri,
            'correction_applied': Duzeltme yapildi mi?
        }
    """
    result = {
        'cleaned': date_str,
        'original': date_str,
        'correction_applied': False
    }

    if not date_str:
        return result

    cleaned = date_str
    original = date_str
    corrected = False

    # Tek haneli ay/gun duzeltme
    parts = re.split(r'[/.-]', cleaned)
    if len(parts) == 3:
        day, month, year = parts
        orig_day, orig_month, orig_year = day, month, year

        # Gun duzeltme
        if len(day) == 1:
            day = '0' + day
            print(f"[Heuristik] Gun tek haneli: {orig_day} -> {day}")
            corrected = True
        # Gun 31'den buyukse duzelt
        try:
            if int(day) > 31:
                old_day = day
                day = day[-2:]
                print(f"[Heuristik] Gun degeri gecersiz: {old_day} -> {day}")
                corrected = True
        except:
            pass

        # Ay duzeltme
        if len(month) == 1:
            month = '0' + month
            print(f"[Heuristik] Ay tek haneli: {orig_month} -> {month}")
            corrected = True
        if month == '0' or month == '00':
            print(f"[Heuristik] Ay degeri gecersiz: {month} -> 07")
            month = '07'  # Varsayilan (Temmuz)
            corrected = True
        # Ay 12'den buyukse duzelt
        try:
            if int(month) > 12:
                old_month = month
                month = month[-2:] if len(month) > 2 else month[-1].zfill(2)
                print(f"[Heuristik] Ay degeri gecersiz: {old_month} -> {month}")
                corrected = True
        except:
            pass

        # Yil duzeltme
        try:
            year_int = int(year)
            if len(year) == 4:
                # 1207, 0207 gibi hatali yillar veya 2020'den eski yillar
                if year_int < 2020 or year_int > 2100:
                    old_year = year
                    year = '2025'  # Guncel yil olarak varsay
                    print(f"[Heuristik] Yil degeri gecersiz veya eski (<2020): {old_year} -> {year}")
                    corrected = True
            elif len(year) == 3:
                old_year = year
                year = '2025'
                print(f"[Heuristik] Yil 3 haneli, gecersiz: {old_year} -> {year}")
                corrected = True
            elif len(year) == 2:
                old_year = year
                year = '20' + year
                # 2020'den kucukse yine 2025 yap
                if int(year) < 2020:
                    year = '2025'
                print(f"[Heuristik] Yil 2 haneli: {old_year} -> {year}")
                corrected = True
        except:
            print(f"[Heuristik] Yil okunamadi, varsayilan: 2025")
            year = '2025'  # Varsayilan
            corrected = True

        cleaned = f"{day}/{month}/{year}"

        if corrected:
            print(f"[Heuristik] Tarih duzeltildi: {original} -> {cleaned}")

    result = {
        'cleaned': cleaned,
        'original': original,
        'correction_applied': corrected
    }

    return result


def clean_time_string(time_str: str) -> Dict[str, Any]:
    """
    Saat stringindeki hatalari duzeltir.

    Returns:
        Dict: {
            'cleaned': Duzeltilmis saat,
            'original': Orijinal OCR degeri,
            'correction_applied': Duzeltme yapildi mi?
        }
    """
    result = {
        'cleaned': time_str,
        'original': time_str,
        'correction_applied': False
    }

    if not time_str:
        return result

    # Virgul ve noktali virgulleri iki noktaya cevir
    cleaned = re.sub(r'[,;.]', ':', time_str)

    # Fazla iki noktalari temizle
    cleaned = re.sub(r':+', ':', cleaned)

    # Orijinal degeri kaydet (ayiricilar normalize edildikten sonra)
    original_normalized = cleaned

    # Heuristik OCR hata duzeltme:
    # Saat 23'u, dakika ve saniye 59'u gecemez. Gecen degerler OCR hatasidir.
    # Ornek: 14:46:71 -> 14:46:00 (71 gecersiz saniye)
    # Ornek: 25:30:00 -> 00:30:00 (25 gecersiz saat)
    # Ozel kural: Saniye hatasi varsa, saat de silik okunmus olabilir (21 -> 16 gibi)
    parts = cleaned.split(':')
    if len(parts) >= 2:
        try:
            corrected = False
            saniye_hatasi = False

            # Once saniye kontrolu yap (silik karakter tespiti icin)
            if len(parts) >= 3 and parts[2].isdigit():
                second = int(parts[2])
                if second >= 60:
                    saniye_hatasi = True
                    print(f"[Heuristik] Saniye degeri gecersiz: {second} -> 00")
                    parts[2] = '00'
                    corrected = True

            # Saat kontrolu (parts[0])
            # Sadece 24'ten buyuk saatleri duzelt, fişteki orijinal saati koru
            if parts[0].isdigit():
                hour = int(parts[0])
                if hour >= 24:
                    print(f"[Heuristik] Saat degeri gecersiz: {hour} -> 00")
                    parts[0] = '00'
                    corrected = True
                # NOT: Saniye hatasi olsa bile saat degerini degistirmiyoruz
                # Fişteki orijinal saat korunuyor

            # Dakika kontrolu (parts[1])
            if parts[1].isdigit():
                minute = int(parts[1])
                if minute >= 60:
                    print(f"[Heuristik] Dakika degeri gecersiz: {minute} -> 00")
                    parts[1] = '00'
                    corrected = True

            cleaned = ':'.join(parts)

            if corrected:
                print(f"[Heuristik] Saat duzeltildi: {original_normalized} -> {cleaned}")

            result = {
                'cleaned': cleaned,
                'original': original_normalized,
                'correction_applied': corrected
            }

        except (ValueError, IndexError):
            result['cleaned'] = cleaned

    return result


def parse_receipt(ocr_lines: List[tuple]) -> Dict[str, Any]:
    """
    OCR ciktisindan fis bilgilerini cikarir.

    Args:
        ocr_lines: EasyOCR ciktisi [(bbox, text, confidence), ...]

    Returns:
        Dict: Yapilandirilmis fis verisi (yeni format)
    """
    # Tum metinleri birlestirilmis liste olarak al
    texts = [item[1] for item in ocr_lines]
    full_text = '\n'.join(texts)

    # OCR hatalarini temizle
    cleaned_text = clean_ocr_text(full_text)

    # Tarih ve saat bilgilerini cikar
    raw_date = extract_date(cleaned_text)
    raw_time = extract_time(cleaned_text)

    # Tarih duzeltme (metadata ile birlikte)
    date_result = clean_date_string(raw_date)

    # Saat duzeltme (metadata ile birlikte)
    time_result = clean_time_string(raw_time)

    # Toplam tutari cikar
    total = extract_amount(cleaned_text,
        ["TOPLAM", "TUTAR", "TOTAL", "AMOUNT", "SUM", "GESAMT"])

    # Urun kalemlerini cikar
    items = extract_items(texts)

    # Islem turunu cikar
    transaction_info = extract_transaction_type(cleaned_text, texts)

    # Yeni JSON formati
    raw_merchant = extract_business_name(texts)
    receipt = {
        "merchant": clean_business_name(raw_merchant),
        "date": date_result['cleaned'],
        "time": time_result['cleaned'],
        "total": total,
        "transaction_type": transaction_info['type'],
        "items": items,
        "metadata": {
            "original_ocr_date": date_result['original'],
            "date_correction_applied": date_result['correction_applied'],
            "original_ocr_time": time_result['original'],
            "time_correction_applied": time_result['correction_applied'],
            "transaction_subtype": transaction_info['subtype'],
            "document_number": transaction_info['document_number'],
            "is_invoice": transaction_info['is_invoice']
        }
    }

    return receipt


def extract_business_name(texts: List[str]) -> Optional[str]:
    """Isletme adini cikarir (genellikle ilk satirlarda)."""
    # Ilk 3 satira bak
    for text in texts[:3]:
        # Buyuk harfli ve anlamli uzunlukta ise
        if len(text) > 3 and text.isupper():
            return text
        # FIRIN, MARKET, RESTORAN gibi kelimeler varsa
        keywords = ["FIRIN", "MARKET", "RESTAURANT", "CAFE", "KAFE", "GIDA"]
        for kw in keywords:
            if kw in text.upper():
                return text
    return texts[0] if texts else None


def extract_address(texts: List[str]) -> Optional[str]:
    """Adres bilgisini cikarir."""
    address_keywords = ["NO:", "CD.", "SK.", "MAH", "GEBZE", "ISTANBUL", "ANKARA", "IZMIR", "KOCAELI"]
    for text in texts:
        for kw in address_keywords:
            if kw in text.upper():
                return text
    return None


def extract_tax_number(text: str) -> Optional[str]:
    """Vergi numarasini cikarir (10-11 haneli sayi)."""
    # V.D., VD, VN gibi etiketlerden sonra
    patterns = [
        r'V\.?D\.?\s*[:.]?\s*(\d{10,11})',
        r'VN\s*[:.]?\s*(\d{10,11})',
        r'(\d{10,11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def extract_tax_office(text: str) -> Optional[str]:
    """Vergi dairesini cikarir."""
    patterns = [
        r'([A-Z]+)\s*V\.?D\.?',
        r'V\.?D\.?\s*[:.]?\s*([A-Z]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.upper())
        if match:
            office = match.group(1)
            if len(office) > 2 and not office.isdigit():
                return office
    return None


def extract_date(text: str) -> Optional[str]:
    """Tarihi cikarir (coklu format destegi)."""
    patterns = [
        # Etiketli formatlar (TR ve EN)
        r'(?:TARIH|DATE|DATUM)\s*[:.]?\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})',
        # DD/MM/YYYY veya DD-MM-YYYY veya DD.MM.YYYY
        r'(\d{2}[/.-]\d{2}[/.-]\d{4})',
        # D/M/YYYY (tek haneli gun/ay)
        r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})',
        # OCR hatali formatlar: 31/0/1207 gibi
        r'(\d{2}[/.-]\d{1}[/.-]\d{4})',
        # YYYY-MM-DD (ISO format)
        r'(\d{4}[/.-]\d{2}[/.-]\d{2})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_time(text: str) -> Optional[str]:
    """Saati cikarir (coklu format destegi)."""
    patterns = [
        # Etiketli formatlar (TR ve EN)
        r'(?:SAAT|TIME|ZEIT)\s*[:.]?\s*(\d{1,2}[:.;,]\d{2}[:.;,]?\d{0,2})',
        # HH:MM:SS veya HH,MM,SS veya HH;MM;SS
        r'(\d{2}[:.;,]\d{2}[:.;,]\d{2})',
        # HH:MM
        r'(\d{2}[:.;,]\d{2})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Tum ayiricilari : yap
            result = match.group(1)
            result = re.sub(r'[;,.]', ':', result)
            return result
    return None


def extract_receipt_number(text: str) -> Optional[str]:
    """Fis numarasini cikarir."""
    patterns = [
        r'FIS\s*NO\s*[:.]?\s*(\d+)',
        r'FISNO\s*[:.]?\s*(\d+)',
        r'Z\s*NO\s*[:.]?\s*(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_amount(text: str, keywords: List[str]) -> Optional[float]:
    """Tutar bilgisini cikarir (coklu para birimi destegi)."""
    for keyword in keywords:
        patterns = [
            # TOTAL € 24,90 veya TOTAL: €24.90
            rf'{keyword}\s*[:.]?\s*[€$£₺]?\s*(\d+[.,]\d{{2}})',
            # € 24,90 TOTAL
            rf'[€$£₺]\s*(\d+[.,]\d{{2}})\s*{keyword}',
            # TOTAL *242,00 (Turk fisleri)
            rf'{keyword}\s*[:.]?\s*\*?(\d+[.,]\d{{2}})',
            # TOTAL 242
            rf'{keyword}\s*[:.]?\s*\*?(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '.')
                try:
                    return float(amount_str)
                except ValueError:
                    continue

    # Eger keyword bulunamazsa, € ile baslayan tutarlari ara
    # € 24,90 veya € 24:90 (OCR hatasi) veya € 24.90
    euro_pattern = r'[€]\s*(\d+[.,;:]\d{2})'
    match = re.search(euro_pattern, text)
    if match:
        amount_str = match.group(1)
        # Tum ayiricilari noktaya cevir
        amount_str = re.sub(r'[,:;]', '.', amount_str)
        try:
            return float(amount_str)
        except ValueError:
            pass

    return None


def extract_items(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Fis metninden urun kalemlerini cikarir.

    Desteklenen formatlar:
    Tek satir:
    - 2 x EKMEK 10,00
    - 2*EKMEK 10,00
    - EKMEK 2 10,00
    - EKMEK 10,00 (miktar 1 varsayilir)

    Ardisik satir:
    - Satir 1: "Unlu Mamuller"
    - Satir 2: "B10" (atlanir)
    - Satir 3: "*242,00"

    Args:
        texts: OCR satirlari listesi

    Returns:
        List[Dict]: [{"quantity": 2, "name": "EKMEK", "price": 10.00}, ...]
    """
    items = []

    # Atlanacak satirlar (baslik, toplam, tarih vb.)
    skip_keywords = [
        'TOPLAM', 'TOTAL', 'TUTAR', 'AMOUNT', 'SUM', 'GESAMT',
        'TOPKDV', 'TOP KDV', 'KDV',  # Toplam KDV satirlari
        'TARIH', 'DATE', 'SAAT', 'TIME', 'FIRIN', 'MARKET',
        'RESTAURANT', 'CAFE', 'ADRES', 'TEL', 'VD', 'VERGI',
        'FIS NO', 'FISNO', 'Z NO', 'NAKIT', 'KREDI',
        'ODEME', 'PARA USTU', 'KART', 'ONAY', 'BANKA',
        'TESEKKUR', 'THANKS', 'HOSGELDINIZ', 'WELCOME',
        'MAH', 'CAD', 'SOK', 'NO:', 'GEBZE', 'ISTANBUL',
        'VAKIFBANK', 'GARANTI', 'ISBANK', 'AKBANK', 'ZIRAAT',
        'VISA', 'MASTERCARD', 'TROY', 'APPROVED', 'ONAYLANDI',
        'TSICIL', 'ISYERI', 'POS', 'BATCH', 'SATIS'
    ]

    # Kullanilmis satirlarin indekslerini takip et
    used_indices = set()

    def should_skip_text(txt: str) -> bool:
        """Satirin atlanip atlanmayacagini kontrol et."""
        txt_upper = txt.upper()
        for kw in skip_keywords:
            if kw in txt_upper:
                return True
        return False

    def is_product_name(txt: str) -> bool:
        """Satirin urun adi olup olmadigini kontrol et."""
        txt = txt.strip()
        if len(txt) < 3:
            return False
        if should_skip_text(txt):
            return False
        # Cogunlukla harflerden olusmali (Turkce karakterler dahil)
        letter_count = sum(1 for c in txt if c.isalpha())
        return letter_count >= len(txt) * 0.6  # En az %60 harf

    def extract_price_from_text(txt: str) -> Optional[float]:
        """Satirdan fiyat cikar."""
        txt = txt.strip()
        # *242,00 veya 8242,00 (OCR hatasi * -> 8) veya 242,00
        # OCR'da * karakteri 8 olarak okunabiliyor

        # Ozel durum: 8 ile baslayan 4+ haneli sayi (8242,00 -> 242,00)
        # Bu durumda 8 aslinda * karakteri
        match_8_prefix = re.match(r'^8(\d{2,}[.,]\d{2})$', txt)
        if match_8_prefix:
            price_str = match_8_prefix.group(1).replace(',', '.')
            try:
                return float(price_str)
            except ValueError:
                pass

        patterns = [
            r'^\*(\d+[.,]\d{2})$',        # *242,00
            r'^(\d+[.,]\d{2})$',          # 242,00
        ]
        for pattern in patterns:
            match = re.match(pattern, txt)
            if match:
                price_str = match.group(1).replace(',', '.')
                try:
                    return float(price_str)
                except ValueError:
                    continue
        return None

    def try_single_line_match(text: str) -> Optional[Dict[str, Any]]:
        """Tek satirda urun+fiyat eslestirmeyi dene."""
        item = None

        # Pattern 1: MIKTAR x URUN FIYAT veya MIKTAR*URUN FIYAT
        pattern1 = r'^(\d+)\s*[xX\*]\s*(.+?)\s+(\d+[.,]\d{2})$'
        match = re.match(pattern1, text)
        if match:
            qty = int(match.group(1))
            name = match.group(2).strip()
            price_str = match.group(3).replace(',', '.')
            item = {"quantity": qty, "name": name, "price": float(price_str)}

        # Pattern 2: URUN MIKTAR FIYAT
        if not item:
            pattern2 = r'^([A-Za-zÀ-ÿĞğÜüŞşİıÖöÇç\s]+)\s+(\d{1,2})\s+(\d+[.,]\d{2})$'
            match = re.match(pattern2, text)
            if match:
                name = match.group(1).strip()
                qty = int(match.group(2))
                price_str = match.group(3).replace(',', '.')
                if 1 <= qty <= 99:
                    item = {"quantity": qty, "name": name, "price": float(price_str)}

        # Pattern 3: URUN FIYAT
        if not item:
            pattern3 = r'^([A-Za-zÀ-ÿĞğÜüŞşİıÖöÇç\s]+)\s+(\d+[.,]\d{2})$'
            match = re.match(pattern3, text)
            if match:
                name = match.group(1).strip()
                price_str = match.group(2).replace(',', '.')
                if len(name) >= 2:
                    item = {"quantity": 1, "name": name, "price": float(price_str)}

        # Pattern 4: *FIYAT URUN
        if not item:
            pattern4 = r'^\*(\d+[.,]\d{2})\s+(.+)$'
            match = re.match(pattern4, text)
            if match:
                price_str = match.group(1).replace(',', '.')
                name = match.group(2).strip()
                if len(name) >= 2:
                    item = {"quantity": 1, "name": name, "price": float(price_str)}

        # Pattern 5: URUN *FIYAT
        if not item:
            pattern5 = r'^(.+?)\s+\*(\d+[.,]\d{2})$'
            match = re.match(pattern5, text)
            if match:
                name = match.group(1).strip()
                price_str = match.group(2).replace(',', '.')
                if len(name) >= 2:
                    item = {"quantity": 1, "name": name, "price": float(price_str)}

        return item

    # ============================================================
    # ANA ISLEM: Tek satir + Ardisik satir eslestirme
    # ============================================================

    i = 0
    while i < len(texts):
        text = texts[i].strip()

        if not text or len(text) < 2 or i in used_indices:
            i += 1
            continue

        if should_skip_text(text):
            i += 1
            continue

        # 1. Tek satir eslestirme dene
        item = try_single_line_match(text)
        if item:
            item['name'] = re.sub(r'\s+', ' ', item['name']).strip()
            item['name'] = clean_product_name(item['name'])
            items.append(item)
            used_indices.add(i)
            print(f"[Items] Tek satir: {item['quantity']}x {item['name']} = {item['price']}")
            i += 1
            continue

        # 2. Ardisik satir eslestirme dene
        # Eger bu satir urun adi gibi gorunuyorsa, sonraki satirlarda fiyat ara
        if is_product_name(text):
            product_name = text
            price = None
            price_index = None

            # Sonraki 1-3 satira bak
            for j in range(i + 1, min(i + 4, len(texts))):
                if j in used_indices:
                    continue
                next_text = texts[j].strip()

                # Fiyat pattern'i ara
                price = extract_price_from_text(next_text)
                if price:
                    price_index = j
                    break

            if price and price_index:
                cleaned_name = re.sub(r'\s+', ' ', product_name).strip()
                cleaned_name = clean_product_name(cleaned_name)
                item = {
                    "quantity": 1,
                    "name": cleaned_name,
                    "price": price
                }
                items.append(item)
                used_indices.add(i)
                used_indices.add(price_index)
                print(f"[Items] Ardisik satir: {item['quantity']}x {item['name']} = {item['price']}")

        i += 1

    return items


def extract_currency(text: str) -> Optional[str]:
    """Para birimini cikarir."""
    currencies = {
        '€': 'EUR',
        'EUR': 'EUR',
        '$': 'USD',
        'USD': 'USD',
        '£': 'GBP',
        'GBP': 'GBP',
        '₺': 'TRY',
        'TL': 'TRY',
        'TRY': 'TRY',
    }

    for symbol, code in currencies.items():
        if symbol in text:
            return code

    return None


def extract_payment_method(text: str) -> Optional[str]:
    """Odeme yontemini cikarir."""
    methods = {
        "KREDI": "Kredi Karti",
        "NAKIT": "Nakit",
        "TEMASSIZ": "Temassiz",
        "PAYWAVE": "Temassiz",
        "CONTACTLESS": "Temassiz",
    }
    text_upper = text.upper()
    for key, value in methods.items():
        if key in text_upper:
            return value
    return None


def extract_bank(text: str) -> Optional[str]:
    """Banka bilgisini cikarir."""
    banks = ["VAKIFBANK", "GARANTI", "ISBANK", "YAPI KREDI", "AKBANK",
             "ZIRAAT", "HALKBANK", "DENIZBANK", "QNB", "TEB", "ING"]
    text_upper = text.upper()
    for bank in banks:
        if bank in text_upper:
            return bank.title()
    return None


def extract_card_number(text: str) -> Optional[str]:
    """Maskeli kart numarasini cikarir."""
    patterns = [
        r'\*{4}\s*\*{4}\s*\*{4}\s*(\d{4})',
        r'\*+(\d{4})',
        r'[^\d](\d{4})\s*\n.*SATIS',  # Kart son 4 hane SATIS'tan once
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"****{match.group(1)}"
    return None


def extract_approval_code(text: str) -> Optional[str]:
    """Onay kodunu cikarir."""
    patterns = [
        r'ONAY\s*KODU?\s*[:.]?\s*(\d{5,6})',
        r'KODU\s*[:.]?\s*\d?\s*(\d{5,6})',
        r'PROV\s*[:.]?\s*(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_transaction_type(text: str, texts: List[str] = None) -> Dict[str, Any]:
    """
    Fis/islem turunu cikarir.

    Turk fislerinde yaygin islem turleri:
    - SATIS: Normal satis fisi
    - IADE: Iade/geri odeme fisi
    - IPTAL: Iptal edilmis islem
    - ARSIV FATURA / E-ARSIV: Arsiv fatura
    - BILGI FISI: Bilgi amacli fis (genellikle fatura ile birlikte)
    - IRSALIYE: Irsaliye belgesi

    Args:
        text: Birlestirilmis OCR metni
        texts: OCR satirlari listesi

    Returns:
        Dict: {
            'type': str (islem turu),
            'subtype': str (alt tur, varsa),
            'document_number': str (belge numarasi, varsa),
            'is_invoice': bool (fatura mi?)
        }
    """
    result = {
        'type': 'SATIS',  # Varsayilan
        'subtype': None,
        'document_number': None,
        'is_invoice': False
    }

    text_upper = text.upper()

    # Islem turu tespiti - oncelik sirasina gore
    transaction_patterns = [
        # Iade/Geri Odeme
        (r'IADE|REFUND|RETURN|GERI\s*ODEME', 'IADE'),
        # Iptal
        (r'IPTAL|VOID|CANCEL', 'IPTAL'),
        # E-Arsiv Fatura
        (r'E[-\s]*ARSIV|E[-\s]*FATURA', 'E-ARSIV'),
        # Arsiv Fatura
        (r'ARSIV\s*FATURA|ARSIV\s*BELGE', 'ARSIV_FATURA'),
        # Bilgi Fisi
        (r'BILGI\s*FISI|INFO\s*RECEIPT', 'BILGI_FISI'),
        # Irsaliye
        (r'IRSALIYE|DELIVERY\s*NOTE', 'IRSALIYE'),
        # Fatura (genel)
        (r'FATURA|INVOICE', 'FATURA'),
    ]

    for pattern, trans_type in transaction_patterns:
        if re.search(pattern, text_upper):
            result['type'] = trans_type
            break

    # Alt tur tespiti (TUR: ARSIV FATURA gibi)
    subtype_match = re.search(r'TUR\s*[:.]?\s*([A-Z\s]+?)(?:\n|$|FATURA)', text_upper)
    if subtype_match:
        subtype = subtype_match.group(1).strip()
        if subtype and len(subtype) > 2:
            result['subtype'] = subtype

    # Belge numarasi tespiti
    doc_patterns = [
        # Fatura/Irsaliye Seri/Sira
        r'(?:FATURA|IRSALIYE).*?(?:SERI|SIRA)\s*[:.]?\s*(\d+)',
        # Belge No
        r'BELGE\s*NO\s*[:.]?\s*(\d+)',
        # Fatura No
        r'FATURA\s*NO\s*[:.]?\s*(\d+)',
        # E-Arsiv No
        r'E[-\s]*ARSIV\s*NO\s*[:.]?\s*(\d+)',
    ]

    for pattern in doc_patterns:
        match = re.search(pattern, text_upper)
        if match:
            result['document_number'] = match.group(1)
            break

    # Fatura mi kontrolu
    if any(t in result['type'] for t in ['FATURA', 'ARSIV', 'E-ARSIV']):
        result['is_invoice'] = True

    # "IRSALIYE YERINE GECER" kontrolu
    if re.search(r'IRSALIYE\s*YERINE\s*GECER', text_upper):
        result['subtype'] = result.get('subtype') or 'IRSALIYE_YERINE_GECER'

    return result


def clean_none_values(d: Dict) -> Dict:
    """None degerleri temizler."""
    cleaned = {}
    for key, value in d.items():
        if isinstance(value, dict):
            nested = clean_none_values(value)
            if nested:  # Bos dict degilse ekle
                cleaned[key] = nested
        elif value is not None:
            cleaned[key] = value
    return cleaned


def to_json(receipt: Dict[str, Any], indent: int = 2) -> str:
    """Dict'i JSON string'e cevirir."""
    return json.dumps(receipt, ensure_ascii=False, indent=indent)


# Ornek kullanim
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, str(__file__).rsplit('\\', 1)[0])

    from ocr_engine import OCREngine

    # Cikti dosyasi
    OUTPUT_FILE = "parsed_receipts.json"

    print("=" * 50)
    print("Fis Parser - OCR'dan JSON'a")
    print("=" * 50)

    if len(sys.argv) > 1:
        image_file = sys.argv[1]

        try:
            # OCR yap
            print(f"\nDosya: {image_file}")
            print("OCR yapiliyor...")

            engine = OCREngine(languages=['tr', 'en'], gpu=False)
            ocr_result = engine.extract_text_from_file(image_file, detail=True)

            # Parse et
            print("Veri ayiklaniyor...")
            receipt = parse_receipt(ocr_result)

            # Mevcut JSON dosyasini oku (varsa)
            receipts_list = []
            if os.path.exists(OUTPUT_FILE):
                try:
                    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                        receipts_list = json.load(f)
                        if not isinstance(receipts_list, list):
                            receipts_list = [receipts_list]
                except (json.JSONDecodeError, IOError):
                    receipts_list = []

            # Yeni fisyi ekle
            receipts_list.append(receipt)

            # JSON dosyasina yaz
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(receipts_list, f, ensure_ascii=False, indent=2)

            print(f"\nSonuc '{OUTPUT_FILE}' dosyasina kaydedildi.")

            # Ozet goster
            print("\n" + "=" * 50)
            print("OZET")
            print("=" * 50)
            print(f"Isletme        : {receipt.get('merchant', '-')}")
            print(f"Tarih          : {receipt.get('date', '-')}")
            print(f"Saat           : {receipt.get('time', '-')}")
            print(f"Toplam         : {receipt.get('total', '-')}")
            print(f"Islem Turu     : {receipt.get('transaction_type', '-')}")

            # Transaction type detaylari
            if receipt['metadata'].get('transaction_subtype'):
                print(f"  Alt Tur      : {receipt['metadata']['transaction_subtype']}")
            if receipt['metadata'].get('document_number'):
                print(f"  Belge No     : {receipt['metadata']['document_number']}")
            if receipt['metadata'].get('is_invoice'):
                print(f"  Fatura       : Evet")

            # Tarih duzeltme bilgisi
            date_corrected = receipt['metadata']['date_correction_applied']
            print(f"Tarih Duzeltme : {'Evet' if date_corrected else 'Hayir'}")
            if date_corrected:
                print(f"  Orijinal     : {receipt['metadata']['original_ocr_date']}")
                print(f"  Duzeltilmis  : {receipt['date']}")

            # Saat duzeltme bilgisi
            time_corrected = receipt['metadata']['time_correction_applied']
            print(f"Saat Duzeltme  : {'Evet' if time_corrected else 'Hayir'}")
            if time_corrected:
                print(f"  Orijinal     : {receipt['metadata']['original_ocr_time']}")
                print(f"  Duzeltilmis  : {receipt['time']}")

        except Exception as e:
            print(f"Hata: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\nKullanim:")
        print("  python receipt_parser.py <goruntu_dosyasi>")
        print("\nOrnek:")
        print("  python receipt_parser.py fis.jpg")
        print(f"\nCikti dosyasi: {OUTPUT_FILE}")
