"""
Constants and configuration values for OCR project.
Centralizes all magic numbers and hardcoded values.
"""

# ============================================================
# OCR ENGINE CONSTANTS
# ============================================================

DEFAULT_LANGUAGES = ['tr', 'en']
DEFAULT_GPU = False


# ============================================================
# PREPROCESSING CONSTANTS
# ============================================================

# Gaussian blur kernel size
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 0

# Median blur kernel size
MEDIAN_KERNEL_SIZE = 5

# Bilateral filter parameters
BILATERAL_DIAMETER = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# Non-local means denoising parameters
NLM_FILTER_STRENGTH = 10
NLM_TEMPLATE_WINDOW_SIZE = 7
NLM_SEARCH_WINDOW_SIZE = 21

# Adaptive threshold parameters
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2

# Binary threshold value
BINARY_THRESHOLD_VALUE = 127
BINARY_MAX_VALUE = 255

# CLAHE parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Canny edge detection thresholds
CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150

# Morphological kernel size
MORPH_KERNEL_SIZE = (9, 9)
MORPH_DILATE_ITERATIONS = 2
MORPH_ERODE_ITERATIONS = 1

# Contour approximation epsilon multiplier
CONTOUR_APPROX_EPSILON = 0.02

# Deskew minimum angle threshold (degrees)
DESKEW_MIN_ANGLE = 0.5

# Shadow removal kernel size
SHADOW_KERNEL_SIZE = (7, 7)
SHADOW_BLUR_SIZE = 21

# Sharpening kernel
SHARPEN_KERNEL = [
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
]


# ============================================================
# RECEIPT PARSER CONSTANTS
# ============================================================

# Minimum text length for product name detection
MIN_PRODUCT_NAME_LENGTH = 3

# Minimum letter ratio for product name detection
MIN_LETTER_RATIO = 0.6

# Default year for date correction
DEFAULT_YEAR = '2025'

# Minimum valid year for dates
MIN_VALID_YEAR = 2020

# Maximum valid year for dates
MAX_VALID_YEAR = 2100

# Maximum hour value
MAX_HOUR = 24

# Maximum minute/second value
MAX_MINUTE_SECOND = 60

# Maximum day value
MAX_DAY = 31

# Maximum month value
MAX_MONTH = 12

# Tax context keywords for Z->% heuristic correction
TAX_CONTEXT_KEYWORDS = [
    'KDV', 'TVA', 'TAX', 'VAT', 'HT', 'TTC',
    'TAUX', 'RATE', 'ORAN', 'VERGI', 'TOPKDV'
]

# Skip keywords for item extraction
SKIP_KEYWORDS = [
    'TOPLAM', 'TOTAL', 'TUTAR', 'AMOUNT', 'SUM', 'GESAMT',
    'TOPKDV', 'TOP KDV', 'KDV',
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

# Business type keywords
BUSINESS_KEYWORDS = ['FIRIN', 'MARKET', 'RESTAURANT', 'CAFE', 'KAFE', 'GIDA']

# Address keywords
ADDRESS_KEYWORDS = ['NO:', 'CD.', 'SK.', 'MAH', 'GEBZE', 'ISTANBUL', 'ANKARA', 'IZMIR', 'KOCAELI']

# Known banks
KNOWN_BANKS = [
    'VAKIFBANK', 'GARANTI', 'ISBANK', 'YAPI KREDI', 'AKBANK',
    'ZIRAAT', 'HALKBANK', 'DENIZBANK', 'QNB', 'TEB', 'ING'
]

# Currency symbols and codes
CURRENCY_MAP = {
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

# Payment methods
PAYMENT_METHODS = {
    'KREDI': 'Kredi Karti',
    'NAKIT': 'Nakit',
    'TEMASSIZ': 'Temassiz',
    'PAYWAVE': 'Temassiz',
    'CONTACTLESS': 'Temassiz',
}

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
