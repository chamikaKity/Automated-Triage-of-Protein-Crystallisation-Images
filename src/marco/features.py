"""Hand-crafted HOG + LBP feature extraction for the classical baseline.

HOG captures edge/gradient structure (crystal facets, needle shapes); LBP captures local
texture (precipitate's grain vs. clear's smooth, uniform appearance) - see notebooks/01_eda.ipynb
for the visual evidence behind this choice. Images vary in size (632x504 to 2448x2048), so
everything is resized to one fixed size first so every feature vector has the same length.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern

IMAGE_SIZE = (128, 128)
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_GRID = (4, 4)  # non-overlapping blocks, for some spatial detail instead of one global histogram


def load_grayscale(image_path: Path) -> np.ndarray:
    """Load an image, resize to IMAGE_SIZE, return a grayscale uint8 array.

    Kept as uint8 (not normalized to float) because local_binary_pattern expects integer-dtype
    images - normalizing first triggers skimage's own floating-point-precision warning.
    """
    img = Image.open(image_path).convert("L").resize(IMAGE_SIZE)
    return np.asarray(img, dtype=np.uint8)


def hog_features(gray: np.ndarray) -> np.ndarray:
    """HOG descriptor over the whole image."""
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def lbp_features(gray: np.ndarray) -> np.ndarray:
    """Grid of uniform-LBP histograms, one per block, concatenated."""
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    grid_h, grid_w = LBP_GRID
    block_h, block_w = gray.shape[0] // grid_h, gray.shape[1] // grid_w

    histograms = []
    for i in range(grid_h):
        for j in range(grid_w):
            block = lbp[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]
            hist, _ = np.histogram(block, bins=n_bins, range=(0, n_bins), density=True)
            histograms.append(hist)
    return np.concatenate(histograms)


def extract_features(image_path: Path) -> np.ndarray:
    """HOG + LBP feature vector for one image - the input to the classical baseline model."""
    gray = load_grayscale(image_path)
    return np.concatenate([hog_features(gray), lbp_features(gray)])
