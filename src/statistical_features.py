import numpy as np
from scipy.stats import skew, kurtosis, gmean, hmean

FEATURE_NAMES = [
    "mean",
    "median",
    "variance",
    "geometric_mean",
    "standard_deviation",
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
    "rms",
]


def calculate_features(feature_map):

    x = np.asarray(feature_map, dtype=np.float64).ravel()

    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    x_positive = np.maximum(x, 1e-8)

    mean_value = np.mean(x)
    median_value = np.median(x)
    variance_value = np.var(x)
    std_value = np.std(x)

    minimum_value = np.min(x)
    maximum_value = np.max(x)

    range_value = maximum_value - minimum_value

    q25 = np.percentile(x, 25)
    q75 = np.percentile(x, 75)

    iqr_value = q75 - q25

    mad_value = np.mean(np.abs(x - mean_value))

    energy_value = np.sum(x ** 2)

    contrast_value = np.mean(
        (x - mean_value) ** 2
    )

    try:
        entropy_value = -np.sum(
            x_positive * np.log(x_positive)
        )
    except Exception:
        entropy_value = 0.0

    try:
        shannon_entropy_value = -np.sum(
            (x_positive / np.sum(x_positive))
            * np.log2(x_positive / np.sum(x_positive) + 1e-8)
        )
    except Exception:
        shannon_entropy_value = 0.0

    if len(x) > 1:
        autocorrelation = np.corrcoef(
            x[:-1], x[1:]
        )[0, 1]

        if not np.isfinite(autocorrelation):
            autocorrelation = 0.0
    else:
        autocorrelation = 0.0

    try:
        kurtosis_value = kurtosis(x)
        if not np.isfinite(kurtosis_value):
            kurtosis_value = 0.0
    except Exception:
        kurtosis_value = 0.0

    try:
        skewness_value = skew(x)
        if not np.isfinite(skewness_value):
            skewness_value = 0.0
    except Exception:
        skewness_value = 0.0

    snr_value = (
        mean_value / (std_value + 1e-8)
    )

    p10 = np.percentile(x, 10)
    p25 = np.percentile(x, 25)
    p50 = np.percentile(x, 50)
    p75 = np.percentile(x, 75)

    try:
        geometric_mean_value = gmean(x_positive)
    except Exception:
        geometric_mean_value = 0.0

    try:
        harmonic_mean_value = hmean(x_positive)
    except Exception:
        harmonic_mean_value = 0.0

    coefficient_variation_value = (
        std_value / (abs(mean_value) + 1e-8)
    )

    sem_value = (
        std_value / np.sqrt(len(x))
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
        rms_value,
    ]

    return np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def calculate_feature_dict(feature_map):

    values = calculate_features(feature_map)

    return {
        name: float(value)
        for name, value in zip(FEATURE_NAMES, values)
    }
