# Medi-Bot

A Streamlit deployment of the student Medi-Bot symptom-classification project.

## Live
https://ml-based-medi-bot-cgeuaprjew8oxdhhfmd4w4.streamlit.app/

## Run locally

1. Clone this repository and open its folder.
2. Create and activate a Python virtual environment.
3. Install dependencies with pip install -r requirements.txt.
4. Start the app with streamlit run app.py.

## Deploy on Streamlit Community Cloud

Select this repository, set the main file path to app.py, then deploy.

The app trains a Random Forest classifier from Training.csv the first time it runs and caches the classifier for the life of the app process. This avoids loading the original pickle, which uses an incompatible scikit-learn format. No secrets are required.

## Medical and privacy notice

This is an educational demonstration, not a medical device or diagnostic service. Outputs may be wrong; consult a qualified clinician for health concerns. This version does not ask for or store personal information.
