"""
Goruntu Isleme (Preprocessing) Modulu
OCR oncesi goruntu hazirlama islemleri icin kullanilir.
"""

import cv2
import numpy as np
from pathlib import Path


def load_image(image_path: str) -> np.ndarray:
    """
    Goruntu dosyasini yukler.

    Args:
        image_path: Goruntu dosyasinin yolu

    Returns:
        numpy.ndarray: Yuklenen goruntu

    Raises:
        FileNotFoundError: Dosya bulunamazsa
        ValueError: Goruntu okunamazsa
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadi: {image_path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Goruntu okunamadi: {image_path}")

    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Goruntuyu gri tona (grayscale) cevirir.

    Args:
        image: BGR formatinda goruntu

    Returns:
        numpy.ndarray: Gri tonlu goruntu
    """
    if len(image.shape) == 2:
        # Zaten gri tonlu
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(image: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """
    Goruntudeki gurultuyu temizler.

    Args:
        image: Gri tonlu goruntu
        method: Gurultu temizleme yontemi
                - "gaussian": Gaussian Blur (hizli, genel amacli)
                - "median": Median Blur (tuz-biber gurultusu icin)
                - "bilateral": Bilateral Filter (kenarlari korur)
                - "nlm": Non-Local Means (en iyi kalite, yavas)

    Returns:
        numpy.ndarray: Gurultusu temizlenmis goruntu
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)

    elif method == "median":
        return cv2.medianBlur(image, 5)

    elif method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)

    elif method == "nlm":
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    else:
        raise ValueError(f"Bilinmeyen yontem: {method}. "
                        f"Gecerli yontemler: gaussian, median, bilateral, nlm")


def apply_threshold(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Thresholding uygulayarak yazilari belirginlestirir.

    Args:
        image: Gri tonlu goruntu
        method: Thresholding yontemi
                - "binary": Basit binary threshold
                - "otsu": Otsu'nun otomatik threshold yontemi
                - "adaptive": Adaptive threshold (degisken aydinlatma icin)
                - "adaptive_gaussian": Gaussian adaptive threshold

    Returns:
        numpy.ndarray: Threshold uygulanmis goruntu
    """
    if method == "binary":
        _, result = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return result

    elif method == "otsu":
        _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    elif method == "adaptive":
        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

    elif method == "adaptive_gaussian":
        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

    else:
        raise ValueError(f"Bilinmeyen yontem: {method}. "
                        f"Gecerli yontemler: binary, otsu, adaptive, adaptive_gaussian")


def preprocess(
    image: np.ndarray,
    noise_method: str = "gaussian",
    threshold_method: str = "adaptive"
) -> np.ndarray:
    """
    Tum preprocessing adimlarini sirasiyla uygular.

    Args:
        image: BGR formatinda goruntu
        noise_method: Gurultu temizleme yontemi
        threshold_method: Thresholding yontemi

    Returns:
        numpy.ndarray: Islenmis goruntu
    """
    # 1. Gri tona cevir
    gray = to_grayscale(image)

    # 2. Gurultu temizle
    denoised = reduce_noise(gray, method=noise_method)

    # 3. Threshold uygula
    result = apply_threshold(denoised, method=threshold_method)

    return result


def enhance_contrast(image: np.ndarray, method: str = "clahe") -> np.ndarray:
    """
    Goruntu kontrastini arttirir.

    Args:
        image: Gri tonlu goruntu
        method: Kontrast artirma yontemi
                - "clahe": Adaptive histogram equalization (onerilen)
                - "hist_eq": Standart histogram equalization
                - "normalize": Min-max normalization

    Returns:
        numpy.ndarray: Kontrasti arttirilmis goruntu
    """
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    elif method == "hist_eq":
        return cv2.equalizeHist(image)

    elif method == "normalize":
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    else:
        raise ValueError(f"Bilinmeyen yontem: {method}")


def detect_receipt_contour(image: np.ndarray) -> np.ndarray:
    """
    Goruntuде fis/belge konturunu tespit eder.

    Args:
        image: BGR formatinda goruntu

    Returns:
        numpy.ndarray: 4 koseli kontur (varsa), yoksa None
    """
    gray = to_grayscale(image)

    # Kenar tespiti
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Morfolojik islemler - kenarlari birlestir
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Konturlari bul
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # En buyuk konturu bul
    largest_contour = max(contours, key=cv2.contourArea)

    # Kontur yaklasiklama
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    # 4 koseli mi kontrol et
    if len(approx) == 4:
        return approx

    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    4 noktayi saat yonunde siralar: sol-ust, sag-ust, sag-alt, sol-alt.

    Args:
        pts: 4 noktadan olusan array

    Returns:
        numpy.ndarray: Siralanmis noktalar
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sol-ust en kucuk toplam, sag-alt en buyuk toplam
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Sag-ust en kucuk fark, sol-alt en buyuk fark
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def perspective_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Perspektif duzeltme uygular.

    Args:
        image: BGR formatinda goruntu
        pts: 4 kose noktasi

    Returns:
        numpy.ndarray: Duzeltilmis goruntu
    """
    rect = order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect

    # Yeni goruntu boyutlarini hesapla
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Hedef noktalar
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    # Perspektif donusumu
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Egik goruntuyu duzeltir (skew correction).

    Args:
        image: Gri tonlu goruntu

    Returns:
        numpy.ndarray: Duzeltilmis goruntu
    """
    # Koordinatlari bul
    coords = np.column_stack(np.where(image > 0))

    if len(coords) < 5:
        return image

    # Minimum alan dikdortgeni bul
    angle = cv2.minAreaRect(coords)[-1]

    # Aciyi duzelt
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Cok kucuk acilar icin dondurme yapma
    if abs(angle) < 0.5:
        return image

    # Goruntuyu dondur
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def remove_shadows(image: np.ndarray) -> np.ndarray:
    """
    Goruntuден golgeleri temizler.

    Args:
        image: BGR formatinda goruntu

    Returns:
        numpy.ndarray: Golgeleri temizlenmis gri tonlu goruntu
    """
    # RGB kanallarini ayir
    rgb_planes = cv2.split(image)

    result_planes = []
    for plane in rgb_planes:
        # Dilate ile arka plani bul
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)

        # Arka plani cikar
        diff = 255 - cv2.absdiff(plane, bg)
        result_planes.append(diff)

    # Kanallari birlestir ve gri tona cevir
    result = cv2.merge(result_planes)
    return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)


def sharpen(image: np.ndarray) -> np.ndarray:
    """
    Goruntuyu keskinlestirir.

    Args:
        image: Gri tonlu goruntu

    Returns:
        numpy.ndarray: Keskinlestirilmis goruntu
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def preprocess_receipt(
    image: np.ndarray,
    detect_roi: bool = True,
    remove_shadow: bool = True,
    enhance: bool = True
) -> np.ndarray:
    """
    Fis goruntusu icin optimize edilmis preprocessing pipeline.

    Args:
        image: BGR formatinda goruntu
        detect_roi: Fis bolgesini tespit et ve kirp
        remove_shadow: Golgeleri temizle
        enhance: Kontrast artirma uygula

    Returns:
        numpy.ndarray: Islenmis goruntu
    """
    result = image.copy()

    # 1. ROI tespiti ve perspektif duzeltme
    if detect_roi:
        contour = detect_receipt_contour(result)
        if contour is not None:
            result = perspective_transform(result, contour)
            print("[Preprocessing] Fis bolgesi tespit edildi, perspektif duzeltildi")

    # 2. Golge temizleme
    if remove_shadow:
        result = remove_shadows(result)
        print("[Preprocessing] Golgeler temizlendi")
    else:
        result = to_grayscale(result)

    # 3. Kontrast artirma
    if enhance:
        result = enhance_contrast(result, method="clahe")
        print("[Preprocessing] Kontrast artirildi (CLAHE)")

    # 4. Gurultu temizleme
    result = reduce_noise(result, method="bilateral")
    print("[Preprocessing] Gurultu temizlendi (bilateral)")

    # 5. Keskinlestirme
    result = sharpen(result)
    print("[Preprocessing] Keskinlestirildi")

    # 6. Adaptive threshold
    result = apply_threshold(result, method="adaptive_gaussian")
    print("[Preprocessing] Threshold uygulandi")

    return result


def preprocess_file(
    input_path: str,
    output_path: str = None,
    noise_method: str = "gaussian",
    threshold_method: str = "adaptive"
) -> np.ndarray:
    """
    Dosyadan goruntu yukler, isler ve opsiyonel olarak kaydeder.

    Args:
        input_path: Giris goruntusu yolu
        output_path: Cikis goruntusu yolu (None ise kaydetmez)
        noise_method: Gurultu temizleme yontemi
        threshold_method: Thresholding yontemi

    Returns:
        numpy.ndarray: Islenmis goruntu
    """
    # Goruntuyu yukle
    image = load_image(input_path)

    # Islemleri uygula
    result = preprocess(image, noise_method, threshold_method)

    # Kaydet (istendiyse)
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Islenmis goruntu kaydedildi: {output_path}")

    return result


# Ornek kullanim
if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("Goruntu Onisleme (Preprocessing) Modulu")
    print("=" * 50)

    # Komut satiri argumanlarindan dosya yolu al
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output_processed.png"

        try:
            # Goruntuyu isle
            result = preprocess_file(
                input_path=input_file,
                output_path=output_file,
                noise_method="gaussian",
                threshold_method="adaptive"
            )
            print(f"Basarili! Goruntu boyutu: {result.shape}")

        except Exception as e:
            print(f"Hata: {e}")
            sys.exit(1)

    else:
        # Kullanim ornekleri goster
        print("\nKullanim:")
        print("-" * 50)
        print("python preprocessing.py <giris_resmi> [cikis_resmi]")
        print("\nOrnek:")
        print("  python preprocessing.py fatura.jpg islenmis.png")

        print("\n" + "=" * 50)
        print("Kod Icinden Kullanim Ornekleri:")
        print("=" * 50)

        print("""
# Temel kullanim
from preprocessing import load_image, preprocess

image = load_image("fatura.jpg")
result = preprocess(image)

# Ozel ayarlarla kullanim
result = preprocess(
    image,
    noise_method="bilateral",      # Kenarlari koruyarak gurultu temizle
    threshold_method="otsu"        # Otsu threshold kullan
)

# Dosya bazli kullanim
from preprocessing import preprocess_file

result = preprocess_file(
    input_path="fatura.jpg",
    output_path="islenmis.png"
)

# Adim adim kullanim
from preprocessing import to_grayscale, reduce_noise, apply_threshold

image = load_image("fatura.jpg")
gray = to_grayscale(image)
denoised = reduce_noise(gray, method="gaussian")
final = apply_threshold(denoised, method="adaptive")
""")

        print("=" * 50)
        print("Mevcut Yontemler:")
        print("=" * 50)
        print("""
Gurultu Temizleme (noise_method):
  - gaussian  : Hizli, genel amacli
  - median    : Tuz-biber gurultusu icin
  - bilateral : Kenarlari korur
  - nlm       : En iyi kalite (yavas)

Thresholding (threshold_method):
  - binary           : Sabit esik degeri (127)
  - otsu             : Otomatik esik hesaplama
  - adaptive         : Degisken aydinlatma icin
  - adaptive_gaussian: Gaussian adaptive (onerilen)
""")
