"""
Image Preprocessing Module
Prepares images for OCR processing.
"""

import cv2
import numpy as np
from pathlib import Path

from utils.logger import get_logger
from constants import (
    GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA,
    MEDIAN_KERNEL_SIZE,
    BILATERAL_DIAMETER, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE,
    NLM_FILTER_STRENGTH, NLM_TEMPLATE_WINDOW_SIZE, NLM_SEARCH_WINDOW_SIZE,
    ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C,
    BINARY_THRESHOLD_VALUE, BINARY_MAX_VALUE
)

logger = get_logger(__name__)


def load_image(image_path: str) -> np.ndarray:
    """Load image from file."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(image: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """
    Remove noise from image.

    Args:
        image: Grayscale image
        method: gaussian, median, bilateral, nlm
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    elif method == "median":
        return cv2.medianBlur(image, MEDIAN_KERNEL_SIZE)
    elif method == "bilateral":
        return cv2.bilateralFilter(
            image, BILATERAL_DIAMETER, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
        )
    elif method == "nlm":
        return cv2.fastNlMeansDenoising(
            image, None, NLM_FILTER_STRENGTH,
            NLM_TEMPLATE_WINDOW_SIZE, NLM_SEARCH_WINDOW_SIZE
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def apply_threshold(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Apply thresholding to enhance text.

    Args:
        image: Grayscale image
        method: binary, otsu, adaptive, adaptive_gaussian
    """
    if method == "binary":
        _, result = cv2.threshold(
            image, BINARY_THRESHOLD_VALUE, BINARY_MAX_VALUE, cv2.THRESH_BINARY
        )
        return result
    elif method == "otsu":
        _, result = cv2.threshold(
            image, 0, BINARY_MAX_VALUE, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return result
    elif method == "adaptive":
        return cv2.adaptiveThreshold(
            image, BINARY_MAX_VALUE, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
        )
    elif method == "adaptive_gaussian":
        return cv2.adaptiveThreshold(
            image, BINARY_MAX_VALUE, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def preprocess(
    image: np.ndarray,
    noise_method: str = "gaussian",
    threshold_method: str = "adaptive"
) -> np.ndarray:
    """
    Basic preprocessing: grayscale -> denoise -> threshold.
    """
    gray = to_grayscale(image)
    denoised = reduce_noise(gray, method=noise_method)
    result = apply_threshold(denoised, method=threshold_method)
    return result


# =============================================================================
# Receipt Isolation Pipeline (Scan & Warp)
# =============================================================================

def detect_edges(image: np.ndarray) -> np.ndarray:
    """
    Canny edge detection + morphology to find receipt boundaries.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)

    return closed


def find_receipt_contour(edges: np.ndarray, min_area: int = 5000) -> np.ndarray:
    """
    Find the largest contour that could be a receipt.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    if not filtered:
        return None

    return max(filtered, key=cv2.contourArea)


def get_receipt_quad(contour: np.ndarray, padding: int = 40) -> np.ndarray:
    """
    Get 4 corners of receipt using minAreaRect for clean rectangle.
    """
    if contour is None:
        return None

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    if padding > 0:
        center = box.mean(axis=0)
        for i in range(len(box)):
            direction = box[i] - center
            norm = np.linalg.norm(direction)
            if norm > 0:
                box[i] = box[i] + (direction / norm) * padding

    return box.astype(np.float32)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def warp_perspective(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """
    Apply perspective transform to straighten receipt.
    """
    rect = order_points(quad)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def extract_receipt(image: np.ndarray, padding: int = 40) -> np.ndarray:
    """
    Full pipeline: detect receipt, crop and straighten.

    Args:
        image: Input BGR image
        padding: Extra pixels around detected receipt

    Returns:
        Cropped and straightened receipt image, or original if detection fails
    """
    edges = detect_edges(image)
    contour = find_receipt_contour(edges)

    if contour is None:
        logger.warning("Receipt contour not found, returning original")
        return image

    quad = get_receipt_quad(contour, padding)
    if quad is None:
        logger.warning("Receipt quad not found, returning original")
        return image

    warped = warp_perspective(image, quad)
    logger.info(f"Receipt extracted: {warped.shape[1]}x{warped.shape[0]}")

    return warped


def preprocess_file(
    input_path: str,
    output_path: str = None,
    noise_method: str = "gaussian",
    threshold_method: str = "adaptive"
) -> np.ndarray:
    """Load, process, and optionally save image."""
    image = load_image(input_path)
    result = preprocess(image, noise_method, threshold_method)

    if output_path:
        cv2.imwrite(output_path, result)
        logger.info(f"Saved preprocessed image: {output_path}")

    return result
