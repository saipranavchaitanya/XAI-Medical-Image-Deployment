import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from src.prediction import predict_image, get_model, DATASET_CONFIG
from src.preprocessing import load_image
from src.gradcam import make_gradcam_heatmap, resize_heatmap, build_gradcam_model
from src.sfmov import create_original_sfmov, upscale_map
from src.statistical_features import calculate_feature_dict, FEATURE_NAMES


# ---------------------------------------------------------------------
# Page setup and visual styling
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="XAI Medical Image Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f7f9fc; }
    [data-testid="stSidebar"] { background: #eef3f9; }
    .hero {
        padding: 1.35rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #102a43 0%, #176b87 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { color: white; margin: 0 0 .35rem 0; font-size: 2rem; }
    .hero p { color: #e5f2f8; margin: 0; font-size: 1rem; }
    .stage-note {
        background: white; border: 1px solid #e3eaf2;
        padding: 1rem 1.1rem; border-radius: 12px;
        margin: .35rem 0 1rem 0;
    }
    .muted { color: #526579; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Explainable Medical Image Diagnosis</h1>
      <p>Explore model predictions through visual explanations, statistical features,
      feature selection and rule-fidelity experiments.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Dataset keys must match src/prediction.py.
DATASET_LABELS = {
    "covid": "COVID-19 Radiography",
    "breast_ultrasound": "Breast Ultrasound",
    "bttypes": "BTTypes",
    "lung_colon": "Lung & Colon Histopathology",
    "acrima": "ACRIMA",
}

# These are dataset-level experimental results transcribed from the
# project's saved metadata. They are NOT predictions for the uploaded image.
EXPERIMENTS = {
    "covid": {
        "zfmis_status": "Top-9 ZFMIS feature subset recorded in the saved rule-fidelity metadata.",
        "selected_features": [
            "percentile_75", "iqr", "autocorrelation", "rms", "energy",
            "snr", "coefficient_variation", "maximum", "range"
        ],
        "rule_fidelity": 0.5108661417322835,
        "tree_depth": 7,
        "tree_leaves": None,
        "note": "Test fidelity from the saved rule-fidelity metadata.",
    },
    "bttypes": {
        "zfmis_status": "Mutual Information ranking recorded; the inspected metadata does not specify a final selected subset.",
        "selected_features": [],
        "mi_top_features": ["p75", "iqr", "mean", "shannon_entropy", "entropy"],
        "rule_fidelity": None,
        "tree_depth": None,
        "tree_leaves": None,
        "note": "A final selected subset and matching deployed rule model were not confirmed in the inspected metadata.",
    },
    "breast_ultrasound": {
        "zfmis_status": "18 features selected using validation weighted F1.",
        "selected_features": [
            "geometric_mean", "entropy", "shannon_entropy", "autocorrelation",
            "mad", "median", "p50", "snr", "coefficient_variation", "p75",
            "iqr", "sem", "std", "variance", "contrast", "maximum", "range", "mean"
        ],
        "rule_fidelity": 0.7107438016528925,
        "tree_depth": 10,
        "tree_leaves": 288,
        "note": "Saved test-set rule-fidelity accuracy; tree depth was selected using validation data.",
    },
    "lung_colon": {
        "zfmis_status": "12-feature subset recorded in the saved rule-fidelity metadata.",
        "selected_features": [
            "entropy", "shannon_entropy", "coefficient_variation", "snr",
            "geometric_mean", "skewness", "harmonic_mean", "kurtosis",
            "iqr", "percentile_75", "autocorrelation", "mean"
        ],
        "rule_fidelity": None,
        "tree_depth": 10,
        "tree_leaves": None,
        "note": "The inspected metadata records the selected depth and feature subset, but not a numeric test-fidelity score.",
    },
    "acrima": {
        "zfmis_status": "The inspected rule-fidelity metadata records three selected statistical features.",
        "selected_features": ["geometric_mean", "coefficient_variation", "iqr"],
        "rule_fidelity": 0.9444444444444444,
        "tree_depth": 4,
        "tree_leaves": 16,
        "note": "Saved test split fidelity accuracy. The inspected metadata records 900 test samples.",
    },
}


def as_numpy(value):
    """Convert TensorFlow tensors or arrays to NumPy."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def make_overlay(original_image, heatmap, alpha=0.38):
    """Create a Grad-CAM overlay without changing the source image."""
    base = np.asarray(original_image.convert("RGB").resize((224, 224))).astype(np.float32)
    heat = np.asarray(heatmap, dtype=np.float32)
    heat = np.clip(heat, 0, 1)
    cmap = plt.get_cmap("jet")
    colored = (cmap(heat)[..., :3] * 255).astype(np.float32)
    overlay = np.clip((1 - alpha) * base + alpha * colored, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def show_feature_table(feature_dict):
    rows = []
    for name in FEATURE_NAMES:
        if name in feature_dict:
            value = feature_dict[name]
            try:
                value = float(np.asarray(value).squeeze())
            except Exception:
                pass
            rows.append({"Feature": name, "Value": value})
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)


def render_zfmis(dataset_key):
    info = EXPERIMENTS.get(dataset_key, {})
    st.subheader("ZFMIS — Statistical Feature Selection")
    st.markdown(
        '<div class="stage-note">ZFMIS is presented here as a saved experimental '
        'feature-selection result. The ranking/subset below is dataset-level '
        'metadata, not a newly calculated selection for this individual image.</div>',
        unsafe_allow_html=True,
    )
    st.write(info.get("zfmis_status", "No ZFMIS metadata was confirmed for this dataset."))
    selected = info.get("selected_features", [])
    if selected:
        st.caption(f"Recorded selected feature count: {len(selected)}")
        st.write(", ".join(selected))
    elif info.get("mi_top_features"):
        st.caption("Top entries in the saved Mutual Information ranking:")
        st.write(", ".join(info["mi_top_features"]))
    else:
        st.info("A confirmed selected feature subset is not available in the inspected metadata.")


def render_rule_fidelity(dataset_key):
    info = EXPERIMENTS.get(dataset_key, {})
    st.subheader("Rule-Fidelity Model — Experimental Evaluation")
    st.markdown(
        '<div class="stage-note">Rule fidelity measures how closely a simpler '
        'decision-tree surrogate reproduces the MobileNetV2 model predictions. '
        'It is not the same as agreement with clinical ground truth.</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    score = info.get("rule_fidelity")
    if score is not None:
        c1.metric("Saved test fidelity", f"{score * 100:.2f}%")
    else:
        c1.metric("Saved test fidelity", "Not confirmed")
    depth = info.get("tree_depth")
    c2.metric("Selected tree depth", str(depth) if depth is not None else "Not confirmed")
    leaves = info.get("tree_leaves")
    c3.metric("Tree leaves", str(leaves) if leaves is not None else "Not confirmed")
    st.caption(info.get("note", "Values are shown only where supported by saved metadata."))
    st.warning(
        "This panel reports previously saved dataset-level experiment results. "
        "It does not run the surrogate tree on the uploaded image."
    )


# ---------------------------------------------------------------------
# Sidebar: dataset and image selection
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Analysis setup")
    dataset_key = st.selectbox(
        "Select medical image dataset",
        options=list(DATASET_LABELS.keys()),
        format_func=lambda key: DATASET_LABELS[key],
    )
    st.markdown("---")
    st.caption("Supported image formats: JPG, JPEG, PNG")
    st.caption("This research prototype is for educational and experimental use only.")

st.subheader("1. Upload a medical image")
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"],
    help="Choose an image belonging to the selected dataset.",
)

if uploaded_file is None:
    st.info("Select a dataset and upload an image to begin the explainability workflow.")
    st.markdown("### Five-stage explainability workflow")
    stages = [
        ("01", "Grad-CAM", "Highlights image regions influential to the selected prediction."),
        ("02", "Original SFMOV", "Displays the original statistical feature-map explanation."),
        ("03", "Statistical Features", "Extracts the 26 statistical descriptors."),
        ("04", "ZFMIS", "Presents saved feature-selection and ranking results."),
        ("05", "Rule-Fidelity", "Presents saved decision-tree surrogate evaluation results."),
    ]
    cols = st.columns(5)
    for col, (number, name, description) in zip(cols, stages):
        with col:
            st.markdown(f"**{number} · {name}**")
            st.caption(description)
    st.stop()


# ---------------------------------------------------------------------
# Prediction and per-image explainability
# ---------------------------------------------------------------------
original_image = Image.open(uploaded_file).convert("RGB")
left, right = st.columns([1, 1.35], gap="large")
with left:
    st.image(original_image, caption="Uploaded medical image", use_container_width=True)
with right:
    st.markdown("### 2. Model prediction")
    run_analysis = st.button("Run prediction & explanations", type="primary", use_container_width=True)

if not run_analysis:
    st.info("Click **Run prediction & explanations** to analyse the uploaded image.")
    st.stop()

suffix = Path(uploaded_file.name).suffix or ".png"
temp_path = None

try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    with st.spinner("Loading the selected model and analysing the image..."):
        result = predict_image(temp_path, dataset=dataset_key)
        model = get_model(dataset_key)
        image_batch = load_image(temp_path)

    prediction_name = result["predicted_class"]
    confidence = float(result["confidence"])

    with right:
        st.success(f"Prediction: {prediction_name}")
        metric_cols = st.columns(2)
        metric_cols[0].metric("Model confidence", f"{confidence * 100:.2f}%")
        metric_cols[1].metric("Dataset", DATASET_LABELS[dataset_key])

        st.markdown("**Class probabilities**")
        probs = result.get("probabilities", {})
        for class_name, probability in probs.items():
            st.write(f"{class_name} — {float(probability) * 100:.2f}%")
            st.progress(float(np.clip(probability, 0, 1)))

    # Generate Grad-CAM and intermediate feature maps once.
    gradcam_image = None
    sfmov_image = None
    feature_dict = None
    explainability_error = None

    try:
        with st.spinner("Generating Grad-CAM, SFMOV and statistical features..."):
            class_index = int(result["class_index"])
            heatmap = make_gradcam_heatmap(
                image_batch,
                model,
                predicted_class_index=class_index,
            )
            heatmap = resize_heatmap(heatmap, target_size=(224, 224))
            gradcam_image = make_overlay(original_image, heatmap)

            feature_extractor, _ = build_gradcam_model(model)
            feature_maps = as_numpy(feature_extractor(image_batch, training=False))
            feature_maps = np.squeeze(feature_maps, axis=0) if feature_maps.ndim == 4 else feature_maps

            sfmov_map = create_original_sfmov(feature_maps)
            sfmov_map = as_numpy(sfmov_map)
            sfmov_map = np.squeeze(sfmov_map)
            sfmov_map = upscale_map(sfmov_map, target_size=(224, 224))
            sfmov_map = as_numpy(sfmov_map)
            sfmov_map = np.squeeze(sfmov_map)
            sfmov_image = sfmov_map

            # Statistical descriptors are calculated from the extracted
            # convolutional feature maps, consistent with the source module API.
            feature_dict = calculate_feature_dict(feature_maps)
    except Exception as exc:
        explainability_error = str(exc)

    st.markdown("---")
    st.subheader("3. Explainability Explorer")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "01 · Grad-CAM",
        "02 · Original SFMOV",
        "03 · Statistical Features",
        "04 · ZFMIS",
        "05 · Rule-Fidelity",
    ])

    with tab1:
        st.markdown("### Grad-CAM visual explanation")
        st.write("Class-discriminative heatmap for the class predicted by MobileNetV2.")
        if gradcam_image is not None:
            a, b = st.columns(2)
            a.image(original_image, caption="Original image", use_container_width=True)
            b.image(gradcam_image, caption="Grad-CAM overlay", use_container_width=True)
        else:
            st.error("Grad-CAM could not be generated.")
            if explainability_error:
                st.code(explainability_error)

    with tab2:
        st.markdown("### Original SFMOV explanation map")
        st.write("Original SFMOV map generated from the extracted convolutional feature maps.")
        if sfmov_image is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(sfmov_image, cmap="inferno")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.error("SFMOV could not be generated.")
            if explainability_error:
                st.code(explainability_error)

    with tab3:
        st.markdown("### Statistical feature extraction")
        st.write("The feature table below is calculated from the uploaded image's extracted feature maps.")
        if feature_dict is not None:
            show_feature_table(feature_dict)
        else:
            st.error("Statistical features could not be calculated.")
            if explainability_error:
                st.code(explainability_error)

    with tab4:
        render_zfmis(dataset_key)

    with tab5:
        render_rule_fidelity(dataset_key)

    with st.expander("Technical details"):
        st.write(f"**Dataset:** {DATASET_LABELS[dataset_key]}")
        st.write(f"**Prediction class index:** {result['class_index']}")
        st.write(f"**Model confidence:** {confidence:.6f}")
        st.write("**Model:** MobileNetV2 transfer-learning classifier")
        st.write("**Explanation method:** Grad-CAM and Original SFMOV")
        st.write("**Statistical feature count:** 26 (where extraction succeeds)")

    st.caption(
        "Research prototype only. Outputs are model predictions and experimental "
        "explanations, not a medical diagnosis. Do not use for clinical decisions."
    )

except Exception as exc:
    st.error("The analysis could not be completed.")
    st.exception(exc)

finally:
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)
