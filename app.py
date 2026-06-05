import streamlit as st
import pandas as pd
import numpy as np

from src.data_processing import load_data
from src.data_processing import engineer_features

from src.model import train_model

from src.visualization import feature_importance_plot


st.set_page_config(
    page_title="Placement Eligibility Predictor",
    layout="wide"
)

# ---------------------------------------------------
# CSS
# ---------------------------------------------------

try:
    with open("assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
except:
    pass

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("🎓 Placement Eligibility Predictor")

st.markdown(
    "Predict student placement readiness using Machine Learning."
)

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

df = load_data("data/student_placement_data.csv")

processed_df, scaler, skill_columns = engineer_features(df)

model, accuracy, feature_names = train_model(processed_df)

# ---------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------

st.sidebar.header("Student Profile")

cgpa = st.sidebar.slider(
    "CGPA",
    0.0,
    10.0,
    8.0,
    0.1
)

internships = st.sidebar.number_input(
    "Internships",
    min_value=0,
    max_value=10,
    value=1
)

backlogs = st.sidebar.selectbox(
    "Backlogs",
    [0, 1]
)

selected_skills = []

for skill in skill_columns:
    if st.sidebar.checkbox(skill):
        selected_skills.append(skill)

# ---------------------------------------------------
# Create Input Row
# ---------------------------------------------------

input_data = {}

input_data["CGPA"] = cgpa
input_data["Internships"] = internships
input_data["Backlogs"] = backlogs

for skill in skill_columns:
    input_data[skill] = 1 if skill in selected_skills else 0

input_df = pd.DataFrame([input_data])

input_df["CGPA"] = scaler.transform(
    input_df[["CGPA"]]
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

prediction = model.predict(input_df)[0]

probability = model.predict_proba(input_df)[0][1]

# ---------------------------------------------------
# Dashboard
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Model Accuracy",
        f"{accuracy:.2%}"
    )

with col2:

    st.metric(
        "Placement Probability",
        f"{probability:.2%}"
    )

# ---------------------------------------------------
# Result
# ---------------------------------------------------

if prediction == 1:
    st.success(
        "✅ Eligible for Placement"
    )
else:
    st.error(
        "❌ Needs Improvement Before Placement"
    )

# ---------------------------------------------------
# Feature Importance
# ---------------------------------------------------

st.subheader("Feature Importance")

fig = feature_importance_plot(
    model,
    feature_names
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------
# Dataset Preview
# ---------------------------------------------------

st.subheader("Dataset Preview")

st.dataframe(df.head())

# ---------------------------------------------------
# Statistics
# ---------------------------------------------------

st.subheader("Dataset Statistics")

st.dataframe(df.describe())
