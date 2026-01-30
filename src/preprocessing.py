"""
Image Preprocessing Module
Provides preprocessing functions for OCR optimization.
"""

import cv2
import numpy as np


def document_scan_preprocess(image: np.ndarray, target_width: int = 800) -> np.ndarray:
    """
    Preprocess image for better OCR results.

    Applies:
    - Resize (if too large)
    - Grayscale conversion
    - Shadow removal (if needed)
    - Denoising

    Args:
        image: Input image (BGR numpy array)
        target_width: Target width for resizing (default: 800)

    Returns:
        Preprocessed grayscale image
    """
    if image is None:
        raise ValueError("Input image is None")

    # 1. Resize if too large
    h, w = image.shape[:2]
    if w > target_width:
        scale = target_width / w
        new_h = int(h * scale)
        image = cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)

    # 2. Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 3. Check if image needs shadow removal
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_clean = laplacian_var > 500

    if is_clean:
        # Clean image: just denoise
        result = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
    else:
        # Dirty image: remove shadows
        result = _remove_shadow(gray)

    return result


def _remove_shadow(gray: np.ndarray) -> np.ndarray:
    """
    Remove shadows from grayscale image.

    Args:
        gray: Grayscale image

    Returns:
        Shadow-removed binary image
    """
    # Background estimation
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    # Normalize (removes shadows)
    normalized = cv2.divide(gray, blur, scale=255)
    # Otsu threshold
    _, thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding for OCR.

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        Binary thresholded image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE.

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)
