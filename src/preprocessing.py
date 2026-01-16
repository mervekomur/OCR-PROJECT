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
