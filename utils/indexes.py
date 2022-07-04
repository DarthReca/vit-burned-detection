# Copyright 2022 Daniele Rege Cambrin
import numpy as np

EPSILON = 1e-5


def ndwi(b03: np.ndarray, b08: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Water Index. The index was proposed by McFeeters, 1996."""
    return (b03 - b08) / (b03 + b08 + EPSILON)


def ndgr(b03: np.ndarray, b04: np.ndarray) -> np.ndarray:
    return (b03 - b04) / (b03 + b04 + EPSILON)


def cloud_coverage(b03: np.ndarray, b04: np.ndarray) -> np.ndarray:
    """
    Cloud detection according to Braaten J, Cohen WB, Yang Z. 2015.
    Automated cloud and cloud shadow identification in
    Landsat MSS imagery for temperate ecosystems.
    Remote Sensing of Environment. 169:128-138.
    """
    return np.logical_and(b03 > 0.175, ndgr(b03, b04) > 0) | (b03 > 0.39)


def bais2(
    b06: np.ndarray,
    b07: np.ndarray,
    b08a: np.ndarray,
    b04: np.ndarray,
    b12: np.ndarray,
) -> np.ndarray:
    """
    Compute Burned Area Index for Sentinel 2 (BAIS 2).

    References
    ----------
    Filipponi, Federico. (2018). BAIS2: Burned Area Index for Sentinel-2. Proceedings. 2. 5177. 10.3390/ecrs-2-05177.
    """
    return (1 - np.sqrt(b06 * b07 * b08a / (b04 + EPSILON))) * (
        (b12 - b08a) / np.sqrt(b12 + b08a + EPSILON) + 1
    )


def ndvi(b08: np.ndarray, b04: np.ndarray) -> np.ndarray:
    """Calculate Normalized difference vegetation index."""
    return (b08 - b04) / (b08 + b04 + EPSILON)


def gndvi(b08: np.ndarray, b03: np.ndarray) -> np.ndarray:
    """Compute Green Normalized Difference Vegetation Index."""
    return (b08 - b03) / (b08 + b03 + EPSILON)


def wiw(b08a: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """Compute Water in Wetlands

    References
    ----------
    Lefebvre et al; Introducing WIW for Detecting the Presence of Water in Wetlands with Landsat and Sentinel Satellites.
    """
    return np.logical_and(b08a < 0.1804, b12 < 0.1132)


def bsi(
    b02: np.ndarray, b04: np.ndarray, b08: np.ndarray, b11: np.ndarray
) -> np.ndarray:
    """Compute Modified Barren Soil index.

    References
    ----------
    Nguyen et al. A Modified Bare Soil Index to Identify Bare Land Features during Agricultural Fallow-Period in Southeast Asia Using Landsat 8.
    """
    return (b11 + b04 - b08 - b02) / (b11 + b04 + b08 + b02 + EPSILON)
