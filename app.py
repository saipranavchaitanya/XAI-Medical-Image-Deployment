
import os
import tempfile

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from src.prediction import predict_image, get_model
from src.preprocessing import load_image
from src.gradcam import make_gradcam_heatmap, resize_heatmap, build_gradcam_model
from src.sfmov import create_original_sfmov, upscale_map


st.set_page_config(
    page_title="Explainable Medical Image Diagnosis",
    page_icon="🩺",
    layout="wide"
)


st.title(
    "Explainable Deep Learning Framework "
    "for Transparent Medical Image Diagnosis"
)

st.write(
    "Upload a medical image, select its dataset, "
    "and view the model prediction with visual explanations."
)


DATASETS = {
    "COVID-19 Radiography": "covid",
    "Breast Ultrasound": "breast_ultrasound",
    "BTTypes": "bttypes",
    "Lung and Colon Histopathology": "lung_colon",
    "ACRIMA": "acrima"
}


selected_dataset = st.selectbox(
    "Select Medical Image Dataset",
    list(DATASETS.keys())
)

dataset_key = DATASETS[selected_dataset]


uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=["png", "jpg", "jpeg"]
)


def create_overlay(original_image, heatmap, alpha=0.4):
    """Create a heatmap overlay on the original image."""

    original = original_image.resize((224, 224))
    original_array = np.asarray(original).astype(np.float32) / 255.0

    colored_heatmap = plt.get_cmap("jet")(
        heatmap
    )[..., :3]

    overlay = (
        (1 - alpha) * original_array
        + alpha * colored_heatmap
    )

    overlay = np.clip(overlay, 0, 1)

    return (overlay * 255).astype(np.uint8)


if uploaded_file is not None:

    original_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Medical Image")
    st.image(
        original_image,
        caption="Original Image",
        use_container_width=True
    )

    temp_path = None

    try:
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        with st.spinner("Analyzing medical image..."):

            # Prediction
            result = predict_image(
                temp_path,
                dataset=dataset_key
            )

            # Load selected model
            model = get_model(dataset_key)

            # Preprocess image
            image_array = load_image(temp_path)

            # Grad-CAM
            gradcam_map = make_gradcam_heatmap(
                image_array,
                model,
                predicted_class_index=result["class_index"]
            )

            gradcam_map = resize_heatmap(gradcam_map)

            # Extract MobileNetV2 feature maps for SFMOV
            feature_extractor, _ = build_gradcam_model(model)

            feature_maps = feature_extractor(
                image_array,
                training=False
            ).numpy()

            # SFMOV
            sfmov_map = create_original_sfmov(
                feature_maps[0]
            )

            sfmov_map = upscale_map(sfmov_map)

        # Prediction results
        st.divider()
        st.header("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Class",
                result["predicted_class"]
            )

        with col2:
            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

        # Class probabilities
        st.subheader("Class Probabilities")

        for class_name, probability in result["probabilities"].items():

            st.write(
                f"**{class_name}** — {probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )

        # Explainability
        st.divider()
        st.header("Explainable AI Results")

        gradcam_overlay = create_overlay(
            original_image,
            gradcam_map
        )

        sfmov_overlay = create_overlay(
            original_image,
            sfmov_map
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Grad-CAM")
            st.image(
                gradcam_overlay,
                caption="Grad-CAM Heatmap Overlay",
                use_container_width=True
            )

        with col2:
            st.subheader("SFMOV")
            st.image(
                sfmov_overlay,
                caption="SFMOV Heatmap Overlay",
                use_container_width=True
            )

        # Explanation maps separately
        with st.expander("View Individual Explanation Maps"):

            col1, col2 = st.columns(2)

            with col1:
                st.write("Grad-CAM Heatmap")
                st.image(
                    gradcam_map,
                    clamp=True,
                    use_container_width=True
                )

            with col2:
                st.write("SFMOV Heatmap")
                st.image(
                    sfmov_map,
                    clamp=True,
                    use_container_width=True
                )

        st.info(
            "These visualizations show model-derived image features "
            "and attention-related patterns. They are experimental "
            "explanations, not clinical evidence or a medical diagnosis."
        )

    except Exception as error:
        st.error(f"Analysis failed: {error}")
        st.exception(error)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
