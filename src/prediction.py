
import os
import numpy as np
import tensorflow as tf

from .preprocessing import load_image


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DATASET_CONFIG = {
    "covid": {
        "model": "covid_baseline_mobilenetv2.keras",
        "classes": [
            "COVID",
            "Lung_Opacity",
            "Normal",
            "Viral Pneumonia"
        ]
    },

    "breast_ultrasound": {
        "model": "breast_ultrasound_mobilenetv2_best.keras",
        "classes": ["benign", "malignant"]
    },

    "bttypes": {
        "model": "bttypes_mobilenetv2_best.keras",
        "classes": ["Benign", "Malignant"]
    },

    "lung_colon": {
        "model": "lung_colon_mobilenetv2_best.keras",
        "classes": [
            "colon_aca",
            "colon_n",
            "lung_aca",
            "lung_scc",
            "lung_n"
        ]
    },

    "acrima": {
        "model": "acrima_mobilenetv2_best.keras",
        "classes": ["Glaucoma", "Normal"]
    }
}


_model_cache = {}


def get_model(dataset="covid"):
    """Load and cache the selected dataset model."""

    dataset = dataset.lower().strip()

    if dataset not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from: {list(DATASET_CONFIG.keys())}"
        )

    if dataset not in _model_cache:
        model_path = os.path.join(
            PROJECT_ROOT,
            "models",
            DATASET_CONFIG[dataset]["model"]
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        _model_cache[dataset] = tf.keras.models.load_model(
            model_path,
            compile=False
        )

    return _model_cache[dataset]


def predict_image(image_path, dataset="covid"):
    """Predict an image using the selected dataset model."""

    dataset = dataset.lower().strip()

    if dataset not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from: {list(DATASET_CONFIG.keys())}"
        )

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    class_names = DATASET_CONFIG[dataset]["classes"]

    model = get_model(dataset)
    image = load_image(image_path)

    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            f"Model output has {len(probabilities)} classes, "
            f"but {dataset} configuration has {len(class_names)}."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    probability_dict = {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }

    return {
        "dataset": dataset,
        "predicted_class": predicted_class,
        "class_index": predicted_index,
        "confidence": confidence,
        "probabilities": probability_dict
    }
