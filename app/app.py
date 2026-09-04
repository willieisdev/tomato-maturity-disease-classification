import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Config ────────────────────────────────────────────────────────────
# Confirmed against notebook-2-v2-three-custom-models-2.ipynb: IMG_SIZE=(224,224),
# rescale=1./255, class_mode='categorical' (softmax output), RGB.
IMG_SIZE = (224, 224)

A1_LABELS = ["Immature", "Mature"]   # Task A1: Maturity Detection (fruit image)
A2_LABELS = ["Fresh", "Rotten"]      # Task A2: Quality Grading (fruit image)
B_LABELS = [                          # Task B: Disease Classification (leaf image)
    "Bacterial_Spot", "Early_Blight", "Late_Blight", "Leaf_Mold",
    "Septoria_Leaf_Spot", "Spider_Mites", "Target_Spot",
    "Yellow_Leaf_Curl_Virus", "Mosaic_Virus", "Healthy",
]

TASKS = {
    "Maturity Detection": {
        "emoji": "🍅",
        "subject": "fruit",
        "model_path": "models/TomatoNet_A1_best.keras",
        "labels": A1_LABELS,
        "metric_name": "Maturity",
    },
    "Quality Grading": {
        "emoji": "🍅",
        "subject": "fruit",
        "model_path": "models/TomatoNet_A2_best.keras",
        "labels": A2_LABELS,
        "metric_name": "Quality",
    },
    "Disease Detection": {
        "emoji": "🍃",
        "subject": "leaf",
        "model_path": "models/TomatoNet_B_best.keras",
        "labels": B_LABELS,
        "metric_name": "Diagnosis",
    },
}

st.set_page_config(page_title="TomatoNet", page_icon="🍅")

# ── Model loading (cached per task, only loads the one you actually use) ─
@st.cache_resource
def load_model_for(task_name: str):
    return tf.keras.models.load_model(TASKS[task_name]["model_path"])

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, arr, labels):
    """Model heads are softmax (class_mode='categorical' in training)."""
    raw = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(raw))
    confidence = float(raw[idx])
    return labels[idx], confidence

# ── Landing: task selection ─────────────────────────────────────────────
st.title("🍅 TomatoNet")
st.caption("Select a task, then upload the matching image type below.")

task_name = st.radio(
    "Choose a task",
    options=list(TASKS.keys()),
    horizontal=True,
    index=None,
    label_visibility="collapsed",
)

if task_name is None:
    st.info(
        "**Maturity Detection** and **Quality Grading** need a photo of the "
        "**fruit**. **Disease Detection** needs a photo of a **leaf**. "
        "Pick a task above to continue."
    )
else:
    task = TASKS[task_name]
    st.write(f"**{task_name}** — upload a photo of the tomato **{task['subject']}**.")

    uploaded_file = st.file_uploader(
        f"Upload a {task['subject']} image",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{task_name}",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded {task['subject']} image", use_container_width=True)

        with st.spinner("Running inference..."):
            model = load_model_for(task_name)
            arr = preprocess(image)
            label, confidence = predict(model, arr, task["labels"])

        st.metric(task["metric_name"], label.replace("_", " "), f"{confidence:.1%} confidence")
    else:
        st.info(f"Upload a {task['subject']} image above to get started.")

st.divider()
with st.expander("About this model"):
    st.write(
        "TomatoNet is a custom CNN trained across three separate tasks: fruit "
        "maturity and quality grading (Sher-e-Bangla dataset, ~98.7% and ~93.0% "
        "accuracy) and leaf disease classification across 10 classes "
        "(PlantVillage tomato leaf dataset, ~88.4% accuracy). Each task uses "
        "its own dedicated model, so only the image type matching your "
        "selected task will produce a meaningful result."
    )
