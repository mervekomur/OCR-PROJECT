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
    BINARY_THRESHOLD_VALUE, BINARY_MAX_VALUE,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE,
    CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH,
    MORPH_KERNEL_SIZE, MORPH_DILATE_ITERATIONS, MORPH_ERODE_ITERATIONS,
    CONTOUR_APPROX_EPSILON,
    DESKEW_MIN_ANGLE,
    SHADOW_KERNEL_SIZE, SHADOW_BLUR_SIZE,
    SHARPEN_KERNEL
)

logger = get_logger(__name__)


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file.

    Args:
        image_path: Path to image file

    Returns:
        Loaded image as numpy array

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If image can't be read
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale.

    Args:
        image: BGR format image

    Returns:
        Grayscale image
    """
    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(image: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """
    Remove noise from image.

    Args:
        image: Grayscale image
        method: Noise reduction method
            - "gaussian": Gaussian Blur (fast, general purpose)
            - "median": Median Blur (for salt-and-pepper noise)
            - "bilateral": Bilateral Filter (preserves edges)
            - "nlm": Non-Local Means (best quality, slow)

    Returns:
        Denoised image
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)

    elif method == "median":
        return cv2.medianBlur(image, MEDIAN_KERNEL_SIZE)

    elif method == "bilateral":
        return cv2.bilateralFilter(
            image,
            BILATERAL_DIAMETER,
            BILATERAL_SIGMA_COLOR,
            BILATERAL_SIGMA_SPACE
        )

    elif method == "nlm":
        return cv2.fastNlMeansDenoising(
            image, None,
            NLM_FILTER_STRENGTH,
            NLM_TEMPLATE_WINDOW_SIZE,
            NLM_SEARCH_WINDOW_SIZE
        )

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Valid methods: gaussian, median, bilateral, nlm"
        )


def apply_threshold(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Apply thresholding to enhance text.

    Args:
        image: Grayscale image
        method: Thresholding method
            - "binary": Simple binary threshold
            - "otsu": Otsu's automatic threshold
            - "adaptive": Adaptive threshold
            - "adaptive_gaussian": Gaussian adaptive threshold

    Returns:
        Thresholded image
    """
    if method == "binary":
        _, result = cv2.threshold(
            image,
            BINARY_THRESHOLD_VALUE,
            BINARY_MAX_VALUE,
            cv2.THRESH_BINARY
        )
        return result

    elif method == "otsu":
        _, result = cv2.threshold(
            image, 0, BINARY_MAX_VALUE,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return result

    elif method == "adaptive":
        return cv2.adaptiveThreshold(
            image, BINARY_MAX_VALUE,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
        )

    elif method == "adaptive_gaussian":
        return cv2.adaptiveThreshold(
            image, BINARY_MAX_VALUE,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
        )

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Valid methods: binary, otsu, adaptive, adaptive_gaussian"
        )


def preprocess(
    image: np.ndarray,
    noise_method: str = "gaussian",
    threshold_method: str = "adaptive"
) -> np.ndarray:
    """
    Apply all preprocessing steps in sequence.

    Args:
        image: BGR format image
        noise_method: Noise reduction method
        threshold_method: Thresholding method

    Returns:
        Preprocessed image
    """
    gray = to_grayscale(image)
    denoised = reduce_noise(gray, method=noise_method)
    result = apply_threshold(denoised, method=threshold_method)

    return result


def enhance_contrast(image: np.ndarray, method: str = "clahe") -> np.ndarray:
    """
    Enhance image contrast.

    Args:
        image: Grayscale image
        method: Contrast enhancement method
            - "clahe": Adaptive histogram equalization (recommended)
            - "hist_eq": Standard histogram equalization
            - "normalize": Min-max normalization

    Returns:
        Contrast-enhanced image
    """
    if method == "clahe":
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE
        )
        return clahe.apply(image)

    elif method == "hist_eq":
        return cv2.equalizeHist(image)

    elif method == "normalize":
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    else:
        raise ValueError(f"Unknown method: {method}")


def detect_receipt_contour(image: np.ndarray) -> np.ndarray:
    """
    Detect receipt/document contour in image.

    Args:
        image: BGR format image

    Returns:
        4-point contour if found, None otherwise
    """
    gray = to_grayscale(image)

    # Edge detection
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    edges = cv2.Canny(blurred, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    edges = cv2.dilate(edges, kernel, iterations=MORPH_DILATE_ITERATIONS)
    edges = cv2.erode(edges, kernel, iterations=MORPH_ERODE_ITERATIONS)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate contour
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, CONTOUR_APPROX_EPSILON * peri, True)

    # Check if 4-point contour
    if len(approx) == 4:
        return approx

    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points clockwise: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array of 4 points

    Returns:
        Ordered points array
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def perspective_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply perspective correction.

    Args:
        image: BGR format image
        pts: 4 corner points

    Returns:
        Corrected image
    """
    rect = order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect

    # Calculate new dimensions
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    # Transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct image skew.

    Args:
        image: Grayscale image

    Returns:
        Deskewed image
    """
    coords = np.column_stack(np.where(image > 0))

    if len(coords) < 5:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < DESKEW_MIN_ANGLE:
        return image

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
    Remove shadows from image.

    Args:
        image: BGR format image

    Returns:
        Shadow-free grayscale image
    """
    rgb_planes = cv2.split(image)

    result_planes = []
    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones(SHADOW_KERNEL_SIZE, np.uint8))
        bg = cv2.medianBlur(dilated, SHADOW_BLUR_SIZE)
        diff = 255 - cv2.absdiff(plane, bg)
        result_planes.append(diff)

    result = cv2.merge(result_planes)
    return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)


def sharpen(image: np.ndarray) -> np.ndarray:
    """
    Sharpen image.

    Args:
        image: Grayscale image

    Returns:
        Sharpened image
    """
    kernel = np.array(SHARPEN_KERNEL)
    return cv2.filter2D(image, -1, kernel)


def preprocess_receipt(
    image: np.ndarray,
    detect_roi: bool = True,
    remove_shadow: bool = True,
    enhance: bool = True
) -> np.ndarray:
    """
    Optimized preprocessing pipeline for receipts.

    Args:
        image: BGR format image
        detect_roi: Detect and crop receipt region
        remove_shadow: Remove shadows
        enhance: Apply contrast enhancement

    Returns:
        Preprocessed image
    """
    result = image.copy()

    # 1. ROI detection and perspective correction
    if detect_roi:
        contour = detect_receipt_contour(result)
        if contour is not None:
            result = perspective_transform(result, contour)
            logger.debug("Receipt region detected, perspective corrected")

    # 2. Shadow removal
    if remove_shadow:
        result = remove_shadows(result)
        logger.debug("Shadows removed")
    else:
        result = to_grayscale(result)

    # 3. Contrast enhancement
    if enhance:
        result = enhance_contrast(result, method="clahe")
        logger.debug("Contrast enhanced (CLAHE)")

    # 4. Noise reduction
    result = reduce_noise(result, method="bilateral")
    logger.debug("Noise reduced (bilateral)")

    # 5. Sharpening
    result = sharpen(result)
    logger.debug("Sharpened")

    # 6. Adaptive threshold
    result = apply_threshold(result, method="adaptive_gaussian")
    logger.debug("Threshold applied")

    return result


def document_scan_preprocess(image: np.ndarray) -> np.ndarray:
    """
    Standard document scanning preprocessing pipeline.
    Simulates CamScanner-like document enhancement.

    Pipeline:
        1. Grayscale conversion
        2. Gaussian blur (noise reduction)
        3. Adaptive thresholding (shadow removal, text enhancement)

    Args:
        image: BGR format image (or grayscale)

    Returns:
        Preprocessed binary image optimized for OCR
    """
    # Step 1: Convert to grayscale
    gray = to_grayscale(image)
    logger.debug("Document scan: Converted to grayscale")

    # Step 2: Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    logger.debug("Document scan: Applied Gaussian blur")

    # Step 3: Adaptive thresholding for shadow removal and text enhancement
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    logger.debug("Document scan: Applied adaptive thresholding")

    return binary


def document_scan_preprocess_file(input_path: str, output_path: str = None) -> np.ndarray:
    """
    Load image, apply document scanning preprocessing, optionally save.

    Args:
        input_path: Input image path
        output_path: Output image path (None to skip saving)

    Returns:
        Preprocessed image
    """
    image = load_image(input_path)
    result = document_scan_preprocess(image)

    if output_path:
        cv2.imwrite(output_path, result)
        logger.info(f"Saved document-scanned image: {output_path}")

    return result


def preprocess_file(
    input_path: str,
    output_path: str = None,
    noise_method: str = "gaussian",
    threshold_method: str = "adaptive"
) -> np.ndarray:
    """
    Load, process, and optionally save image.

    Args:
        input_path: Input image path
        output_path: Output image path (None to skip saving)
        noise_method: Noise reduction method
        threshold_method: Thresholding method

    Returns:
        Preprocessed image
    """
    image = load_image(input_path)
    result = preprocess(image, noise_method, threshold_method)

    if output_path:
        cv2.imwrite(output_path, result)
        logger.info(f"Saved preprocessed image: {output_path}")

    return result
