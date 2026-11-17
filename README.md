# Medi-Bot

A Streamlit deployment of the student Medi-Bot symptom-classification project.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, choose **Create app** and select that repository.
3. Set the main file path to `app.py` and deploy.

The app uses `Training.csv` for symptom names and class labels, and `model.pkl` for predictions. No secrets are required.

## Medical and privacy notice

This is an educational demonstration, not a medical device or diagnostic service. Outputs may be wrong; consult a qualified clinician for health concerns. This version does not ask for or store personal information.
