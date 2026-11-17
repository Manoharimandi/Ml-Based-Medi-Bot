from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent
DATA = pd.read_csv(ROOT / "Training.csv")
DATA = DATA.loc[:, ~DATA.columns.astype(str).str.match(r"^Unnamed:")].copy()
constant_features = [
    column
    for column in DATA.columns
    if column != "prognosis" and DATA[column].nunique(dropna=False) <= 1
]
DATA = DATA.drop(columns=constant_features)
DATA = DATA.rename(
    columns={
        column: column[:-2]
        for column in DATA.columns
        if column.endswith(".1") and column[:-2] not in DATA.columns
    }
)
FEATURES = [column for column in DATA.columns if column != "prognosis"]

st.set_page_config(page_title="Medi-Bot | Symptom Explorer", page_icon="🩺", layout="centered")
st.title("🩺 Medi-Bot")
st.subheader("Explore possible conditions from symptoms")
st.warning(
    "For educational demonstration only. This tool cannot diagnose illness or replace a "
    "licensed clinician. If you have severe or urgent symptoms, seek immediate medical care."
)
st.write("Choose the symptoms you are experiencing, then run the model.")

@st.cache_resource
def get_model_and_labels():
    """Train against this runtime's scikit-learn version to avoid stale pickle formats."""
    labels = LabelEncoder()
    target = labels.fit_transform(DATA["prognosis"].astype(str))
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(DATA[FEATURES], target)
    return model, labels.classes_

symptoms = st.multiselect(
    "Symptoms",
    options=FEATURES,
    format_func=lambda value: value.replace("_", " ").strip().capitalize(),
    help="The model recognizes the symptom labels included in the student training dataset.",
)

if st.button("Explore possible condition", type="primary", disabled=not symptoms):
    sample = pd.DataFrame([[int(feature in symptoms) for feature in FEATURES]], columns=FEATURES)
    try:
        model, disease_labels = get_model_and_labels()
        prediction = int(model.predict(sample)[0])
        if prediction < 0 or prediction >= len(disease_labels):
            raise ValueError("Model returned an unexpected class label")
        condition = disease_labels[prediction]
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
