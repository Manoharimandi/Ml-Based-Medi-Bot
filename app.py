from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

ROOT = Path(__file__).parent
DATA = pd.read_csv(ROOT / "Training.csv")
FEATURES = [column for column in DATA.columns if column != "prognosis"]
DISEASES = sorted(DATA["prognosis"].dropna().unique().tolist())

st.set_page_config(page_title="Medi-Bot | Symptom Explorer", page_icon="🩺", layout="centered")
st.title("🩺 Medi-Bot")
st.subheader("Explore possible conditions from symptoms")
st.warning(
    "For educational demonstration only. This tool cannot diagnose illness or replace a "
    "licensed clinician. If you have severe or urgent symptoms, seek immediate medical care."
)
st.write("Choose the symptoms you are experiencing, then run the model.")

@st.cache_resource
def get_model():
    return load(ROOT / "model.pkl")

symptoms = st.multiselect(
    "Symptoms",
    options=FEATURES,
    format_func=lambda value: value.replace("_", " ").strip().capitalize(),
    help="The model recognizes the symptom labels included in the student training dataset.",
)

if st.button("Explore possible condition", type="primary", disabled=not symptoms):
    sample = pd.DataFrame([[int(feature in symptoms) for feature in FEATURES]], columns=FEATURES)
    try:
        prediction = int(get_model().predict(sample)[0])
        if prediction < 1 or prediction > len(DISEASES):
            raise ValueError("Model returned an unexpected class label")
        condition = DISEASES[prediction - 1]
        st.markdown("### Model result")
        st.info(condition)
        st.caption(
            "This is a model output based on a classroom dataset, not a medical diagnosis. "
            "Do not use it to choose or delay treatment."
        )
    except Exception as exc:
        st.error(f"The model could not produce a result: {exc}")

with st.expander("About this project"):
    st.write(
        "Medi-Bot is a student project that demonstrates a machine-learning symptom "
        "classifier. Predictions depend on the supplied dataset and may be inaccurate. "
        "No account or personal information is collected by this Streamlit version."
    )
