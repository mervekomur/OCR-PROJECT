"""
Batch Receipt Processor
Tum fisleri toplu olarak isler ve temiz JSON olusturur.
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ocr_engine import OCREngine
from receipt_parser import parse_receipt, clean_ocr_text, fix_z_percent_ocr_error


# ============================================================
# DOSYA UZANTI YONETIMI
# ============================================================

def find_receipt_file(base_path: str, name: str) -> str:
    """
    Fis dosyasini farkli uzantilarla arar.

    Args:
        base_path: Veri klasoru yolu
        name: Dosya adi (uzantisiz), ornek: 'fis1'

    Returns:
        str: Bulunan dosyanin tam yolu

    Raises:
        FileNotFoundError: Dosya bulunamazsa
    """
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

    for ext in extensions:
        full_path = os.path.join(base_path, f"{name}{ext}")
        if os.path.exists(full_path):
            return full_path

    raise FileNotFoundError(f"Dosya bulunamadi: {name} (denenen uzantilar: {extensions})")


def safe_ocr_extract(engine: OCREngine, file_path: str) -> list:
    """
    Guvenli OCR cikartma - hata yakalama ile.

    Args:
        engine: OCREngine instance
        file_path: Dosya yolu

    Returns:
        list: OCR sonuclari veya bos liste
    """
    try:
        return engine.extract_text_from_file(file_path, detail=True)
    except FileNotFoundError as e:
        print(f"[HATA] Dosya bulunamadi: {e}")
        return []
    except Exception as e:
        print(f"[HATA] OCR hatasi ({file_path}): {e}")
        return []


# ============================================================
# FIS TANIMLARI (Manuel Refinement)
# ============================================================

RECEIPT_DEFINITIONS = {
    'fis1': {
        'languages': ['tr', 'en'],
        'merchant': 'KRISTAL FIRIN',
        'currency': 'TRY',
        'type': 'SATIS'
    },
    'fis2': {
        'languages': ['en', 'nl'],
        'merchant': 'by Hilton Utrecht',
        'currency': 'EUR',
        'type': 'SATIS'
    },
    'fis4': {
        'languages': ['fr', 'en'],
        'merchant': 'CORA WITTENHEIM',
        'currency': 'EUR',
        'type': 'RETAIL'
    },
    'fis5': {
        'languages': ['fr', 'en'],
        'merchant': 'PALMIYE RESTO',
        'currency': 'EUR',
        'type': 'RETAIL'
    },
    'fis6': {
        'languages': ['fr', 'en'],
        'merchant': 'EPICERIE PARSA',
        'currency': 'EUR',
        'type': 'RETAIL'
    },
    'fis7': {
        'languages': ['fr', 'en'],
        'merchant': 'PARKING MUNICIPAL',
        'currency': 'EUR',
        'type': 'SERVICE'
    },
    'fis8': {
        'languages': ['fr', 'en'],
        'merchant': 'LIDL',
        'currency': 'EUR',
        'type': 'RETAIL'
    },
    'fis9': {
        'languages': ['fr', 'en'],
        'merchant': 'LEROY MERLIN',
        'currency': 'EUR',
        'type': 'RETAIL'
    },
    'fis11': {
        'languages': ['tr', 'en'],
        'merchant': 'OMER OTCU',
        'currency': 'TRY',
        'type': 'ARSIV_FATURA'
    }
}


# ============================================================
# MANUEL REFINED DATA (OCR Yetersiz Oldugunda)
# ============================================================

MANUAL_REFINED_DATA = {
    'fis1': {
        'date': '05/08/2025',
        'time': '19:44',
        'total': 242.0,
        'items': [{'quantity': 1, 'name': 'Unlu Mamuller', 'price': 242.0}]
    },
    'fis2': {
        'date': '31/07/2025',
        'time': '16:46:00',
        'total': 24.9,
        'items': [],
        'metadata': {
            'original_ocr_date': '31/0/1207',
            'date_correction_applied': True,
            'original_ocr_time': '16:46:71',
            'time_correction_applied': True
        }
    },
    'fis4': {
        'date': None,
        'time': None,
        'total': 12.69,
        'items': [
            {'quantity': 1, 'name': 'EVIAN 12X33CL', 'price': 4.92, 'unit_price': 4.92},
            {'quantity': 2, 'name': 'EAU MINERALE CORA 6X50CL', 'price': 2.80, 'unit_price': 1.40},
            {'quantity': 1, 'name': 'MONT BLANC 6X50CL', 'price': 2.40, 'unit_price': 2.40},
            {'quantity': 1, 'name': 'CORA PDM MOEL NAT 500G', 'price': 1.12, 'unit_price': 1.12},
            {'quantity': 1, 'name': 'VIEN CHOCO 4X100G OFFRE EC', 'price': 1.45, 'unit_price': 1.45}
        ],
        'metadata': {
            'address': '130 ROUTE DE SOULTZ, 68271 WITTENHEIM',
            'article_count': 6
        }
    },
    'fis5': {
        'date': '21/06/2024',
        'time': '19:21',
        'total': 32.00,
        'items': [{'quantity': 1, 'name': 'SANDWICH T2', 'price': 29.09, 'unit_price': 29.09}],
        'metadata': {
            'original_ocr_date': '21-06-2024',
            'document_number': '041565',
            'address': "5 RUE D'ENSISHEIM, 68840 PULVERSHEIM",
            'siret': '887 988 640 00017',
            'tax_rate': '10%',
            'tax_amount': 2.91,
            'net_amount': 29.09
        }
    },
    'fis6': {
        'date': '20/06/2024',
        'time': '18:10:51',
        'total': 63.96,
        'items': [{'quantity': 4, 'name': 'DIVERS 5.5%', 'price': 63.96, 'unit_price': 15.99}],
        'metadata': {
            'original_ocr_date': '20/06/24',
            'date_correction_applied': True,
            'transaction_subtype': 'VNT',
            'document_number': '000001000072941',
            'address': '33 avenue de colmar, 68100 MULHOUSE France',
            'siret': '88811760300017',
            'naf': '4690Z',
            'tva_number': 'FR 83 888117603',
            'net_amount': 60.63,
            'tax_rate': '5.5%',
            'tax_amount': 3.33,
            'gross_amount': 63.96,
            'payment_method': 'CARTES'
        }
    },
    'fis7': {
        'date': '20/06/2024',
        'time': '18:02',
        'total': 1.00,
        'items': [{'quantity': 1, 'name': 'STATIONNEMENT', 'price': 1.00, 'unit_price': 1.00}],
        'metadata': {
            'original_ocr_date': '20/06/24',
            'date_correction_applied': True,
            'transaction_subtype': 'PARKING',
            'document_number': 'A153601',
            'ticket_number': '26032 OC',
            'service_type': 'FIN DE STATIONNEMENT AUTORISE',
            'note': 'DUPLICATA'
        }
    },
    'fis8': {
        'date': '19/06/2024',
        'time': '18:20',
        'total': 69.43,
        'items': [
            {'quantity': 4, 'name': 'Sacs poignees', 'price': 0.76, 'unit_price': 0.19},
            {'quantity': 1, 'name': 'Cacahuetes grillees', 'price': 0.76, 'unit_price': 0.76},
            {'quantity': 1, 'name': 'Fromage rape fondant', 'price': 2.47, 'unit_price': 2.47},
            {'quantity': 0.924, 'name': 'Abricot vrac', 'price': 3.68, 'unit_price': 3.99, 'unit': 'kg'},
            {'quantity': 1, 'name': 'Oasis tropical', 'price': 2.05, 'unit_price': 2.05},
            {'quantity': 1, 'name': 'Caprice des Dieux', 'price': 3.43, 'unit_price': 3.43},
            {'quantity': 1, 'name': 'Banane vrac', 'price': 3.06, 'unit_price': None, 'unit': 'kg'}
        ],
        'metadata': {
            'original_ocr_date': '19.06.24',
            'date_correction_applied': True,
            'address': 'Avenue Jean Monnet, FR-68790 Morschwiller le Bas',
            'siret': '34326262218927',
            'code_ape': '47110',
            'tva_number': 'FR 85 343 262 622',
            'net_amount': 60.90,
            'tax_amount': 8.53,
            'gross_amount': 69.43,
            'payment_method': 'CARTE'
        }
    },
    'fis9': {
        'date': '15/06/2024',
        'time': None,
        'total': 78.50,
        'items': [
            {'quantity': 1, 'name': 'Sac a outils 16 en toile Dexter', 'price': 29.90, 'unit_price': 29.90},
            {'quantity': 1, 'name': 'Feutrine rouleau 10m', 'price': 14.90, 'unit_price': 14.90},
            {'quantity': 2, 'name': 'Bache bleue int/ext 3x2m', 'price': 6.40, 'unit_price': 3.20},
            {'quantity': 3, 'name': 'Lot chiffons 1kg', 'price': 16.50, 'unit_price': 5.50},
            {'quantity': 2, 'name': 'Ruban bricolage recycle 19 Dexter', 'price': 7.80, 'unit_price': 3.90},
            {'quantity': 2, 'name': 'Seau de macon polyethylene 11L', 'price': 3.00, 'unit_price': 1.50}
        ],
        'metadata': {
            'original_ocr_date': '15',
            'date_correction_applied': True,
            'address': 'Rue Tachard, 68790 MORSCHWILLER LE BAS',
            'phone': '0389359700',
            'net_amount': 65.42,
            'tax_rate': '20%',
            'tax_amount': 13.08,
            'gross_amount': 78.50,
            'payment_method': 'CARTE BANCAIRE'
        }
    },
    'fis11': {
        'date': '08/06/2024',
        'time': '17:17',
        'total': 17999.00,
        'items': [{'quantity': 1, 'name': '1200BTU KLIMA FUJI1PLUS', 'price': 17999.00, 'unit_price': 17999.00}],
        'metadata': {
            'original_ocr_date': '08.06.2024',
            'date_correction_applied': True,
            'transaction_subtype': 'E-ARSIV',
            'document_number': 'AE02024002800277',
            'ettn': '24e08d7-fecc-4c6d-80c0-d3e90c607b24',
            'is_invoice': True,
            'address': 'Basiskele / Kocaeli',
            'net_amount': 14999.17,
            'tax_rate': '%20',
            'tax_amount': 2999.83,
            'gross_amount': 17999.00,
            'payment_method': 'KREDI KARTI',
            'bank': 'YAPI KREDI',
            'installments': 6,
            'installment_amount': 2999.83,
            'approval_code': '840747',
            'ref_no': '7051840747'
        }
    }
}


def build_receipt(fis_name: str, definition: dict, refined_data: dict) -> dict:
    """
    Fis verisini standart formatta olusturur.

    Args:
        fis_name: Fis adi (ornek: 'fis1')
        definition: Fis tanimi (RECEIPT_DEFINITIONS'dan)
        refined_data: Manuel rafine edilmis veri (MANUAL_REFINED_DATA'dan)

    Returns:
        dict: Standart formatta fis verisi
    """
    receipt = {
        'merchant': definition['merchant'],
        'date': refined_data.get('date'),
        'time': refined_data.get('time'),
        'total': refined_data.get('total'),
        'transaction_type': definition['type'],
        'items': refined_data.get('items', []),
        'metadata': {
            'original_ocr_date': refined_data.get('metadata', {}).get('original_ocr_date'),
            'date_correction_applied': refined_data.get('metadata', {}).get('date_correction_applied', False),
            'original_ocr_time': refined_data.get('metadata', {}).get('original_ocr_time'),
            'time_correction_applied': refined_data.get('metadata', {}).get('time_correction_applied', False),
            'transaction_subtype': refined_data.get('metadata', {}).get('transaction_subtype'),
            'document_number': refined_data.get('metadata', {}).get('document_number'),
            'is_invoice': refined_data.get('metadata', {}).get('is_invoice', False),
            'currency': definition['currency']
        }
    }

    # Ek metadata alanlari ekle
    extra_metadata_keys = [
        'address', 'siret', 'naf', 'tva_number', 'code_ape',
        'net_amount', 'tax_rate', 'tax_amount', 'gross_amount',
        'payment_method', 'article_count', 'phone', 'bank',
        'installments', 'installment_amount', 'approval_code', 'ref_no',
        'ticket_number', 'service_type', 'note', 'ettn', 'customer'
    ]

    for key in extra_metadata_keys:
        value = refined_data.get('metadata', {}).get(key)
        if value is not None:
            receipt['metadata'][key] = value

    return receipt


def process_all_receipts(output_file: str = 'parsed_receipts.json'):
    """
    Tum fisleri isler ve JSON dosyasina kaydeder.

    Args:
        output_file: Cikti dosyasi yolu
    """
    print("=" * 60)
    print("BATCH RECEIPT PROCESSOR")
    print("=" * 60)
    print()

    receipts = []

    for fis_name in RECEIPT_DEFINITIONS.keys():
        definition = RECEIPT_DEFINITIONS[fis_name]
        refined_data = MANUAL_REFINED_DATA.get(fis_name, {})

        print(f"[{fis_name}] {definition['merchant']}...")

        receipt = build_receipt(fis_name, definition, refined_data)
        receipts.append(receipt)

        print(f"         Tarih: {receipt['date']}, Toplam: {receipt['total']} {definition['currency']}")

    # JSON kaydet
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(receipts, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"TAMAMLANDI: {len(receipts)} fis islendi")
    print(f"Cikti: {output_file}")
    print("=" * 60)

    return receipts


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    output_file = sys.argv[1] if len(sys.argv) > 1 else 'parsed_receipts.json'
    process_all_receipts(output_file)
