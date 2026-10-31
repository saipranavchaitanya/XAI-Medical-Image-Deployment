
import numpy as np
from scipy.stats import skew, kurtosis, entropy, gmean, hmean


FEATURE_NAMES = [
    "mean",
    "median",
    "variance",
    "geometric_mean",
    "std",
    "minimum",
    "autocorrelation",
    "iqr",
    "kurtosis",
    "entropy",
    "energy",
    "contrast",
    "mad",
    "shannon_entropy",
    "range",
    "skewness",
    "maximum",
    "snr",
    "percentile_10",
    "percentile_25",
    "percentile_50",
    "percentile_75",
    "harmonic_mean",
    "coefficient_variation",
    "sem",
    "rms"
]


def safe_positive(values):
    values = np.asarray(values, dtype=np.float64)
    return np.maximum(
        values,
        1e-8
    )


def calculate_features(feature_map):
    """Calculate the project's 26 statistical features."""

    x = np.asarray(
        feature_map,
        dtype=np.float64
    ).flatten()

    x_positive = safe_positive(x)

    mean_value = np.mean(x)

    median_value = np.median(x)

    variance_value = np.var(x)

    std_value = np.std(x)

    minimum_value = np.min(x)

    maximum_value = np.max(x)

    # Lag-1 autocorrelation
    if len(x) > 1 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0:
        autocorrelation = np.corrcoef(
            x[:-1],
            x[1:]
        )[0, 1]
    else:
        autocorrelation = 0.0

    q25 = np.percentile(x, 25)
    q75 = np.percentile(x, 75)

    iqr_value = q75 - q25

    try:
        kurtosis_value = kurtosis(
            x,
            fisher=True,
            bias=False
        )
    except Exception:
        kurtosis_value = 0.0

    # Histogram entropy
    hist, _ = np.histogram(
        x,
        bins=32,
        density=True
    )

    hist = hist + 1e-12

    entropy_value = entropy(
        hist
    )

    energy_value = np.mean(
        x ** 2
    )

    contrast_value = np.std(
        x
    )

    mad_value = np.median(
        np.abs(
            x - median_value
        )
    )

    probabilities = np.abs(x) + 1e-12
    probabilities = probabilities / probabilities.sum()

    shannon_entropy_value = entropy(
        probabilities
    )

    range_value = (
        maximum_value -
        minimum_value
    )

    try:
        skewness_value = skew(
            x,
            bias=False
        )
    except Exception:
        skewness_value = 0.0

    snr_value = (
        mean_value /
        (std_value + 1e-8)
    )

    p10 = np.percentile(x, 10)
    p25 = np.percentile(x, 25)
    p50 = np.percentile(x, 50)

    try:
        geometric_mean_value = gmean(
            x_positive
        )
    except Exception:
        geometric_mean_value = 0.0

    try:
        harmonic_mean_value = hmean(
            x_positive
        )
    except Exception:
        harmonic_mean_value = 0.0

    coefficient_variation_value = (
        std_value /
        (abs(mean_value) + 1e-8)
    )

    sem_value = (
        std_value /
        np.sqrt(len(x))
    )

    rms_value = np.sqrt(
        np.mean(x ** 2)
    )

    values = [
        mean_value,
        median_value,
        variance_value,
        geometric_mean_value,
        std_value,
        minimum_value,
        autocorrelation,
        iqr_value,
        kurtosis_value,
        entropy_value,
        energy_value,
        contrast_value,
        mad_value,
        shannon_entropy_value,
        range_value,
        skewness_value,
        maximum_value,
        snr_value,
        p10,
        p25,
        p50,
        p75,
        harmonic_mean_value,
        coefficient_variation_value,
        sem_value,
        rms_value
    ]

    return np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


def calculate_feature_dict(feature_map):
    values = calculate_features(
        feature_map
    )

    return dict(
        zip(
            FEATURE_NAMES,
            values
        )
    )
