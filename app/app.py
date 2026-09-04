import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Config ────────────────────────────────────────────────────────────
# Confirmed against notebook-2-v2-three-custom-models-2.ipynb: IMG_SIZE=(224,224),
# rescale=1./255, class_mode='categorical' (2-unit softmax output), RGB.
IMG_SIZE = (224, 224)

A1_LABELS = ["Immature", "Mature"]   # Task A1: Maturity Detection (binary)
A2_LABELS = ["Fresh", "Rotten"]      # Task A2: Quality Grading (binary)

A1_MODEL_PATH = "models/TomatoNet_A1_best.keras"
A2_MODEL_PATH = "models/TomatoNet_A2_best.keras"

st.set_page_config(page_title="TomatoNet — Maturity & Quality Classifier", page_icon="🍅")

# ── Model loading (cached so it only loads once per session) ───────────
@st.cache_resource
def load_models():
    a1_model = tf.keras.models.load_model(A1_MODEL_PATH)
    a2_model = tf.keras.models.load_model(A2_MODEL_PATH)
    return a1_model, a2_model

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_binary(model, arr, labels):
    """Model heads are 2-unit softmax (class_mode='categorical' in training)."""
    raw = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(raw))
    confidence = float(raw[idx])
    return labels[idx], confidence

# ── UI ───────────────────────────────────────────────────────────────
st.title("🍅 TomatoNet")
st.caption("Dual-task classifier: maturity detection + quality grading")

uploaded_file = st.file_uploader("Upload a tomato image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Loading models and running inference..."):
        a1_model, a2_model = load_models()
        arr = preprocess(image)

        a1_label, a1_conf = predict_binary(a1_model, arr, A1_LABELS)
        a2_label, a2_conf = predict_binary(a2_model, arr, A2_LABELS)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Maturity", a1_label, f"{a1_conf:.1%} confidence")
    with col2:
        st.metric("Quality", a2_label, f"{a2_conf:.1%} confidence")

    st.divider()
    with st.expander("About this model"):
        st.write(
            "TomatoNet is a custom dual-stream CNN trained on the Sher-e-Bangla "
            "Tomato Maturity Detection and Quality Grading Dataset. "
            "Task A1 (maturity) reaches ~98.7% accuracy; Task A2 (quality) ~93.0% "
            "accuracy on held-out test data."
        )
else:
    st.info("Upload an image above to get started.")
