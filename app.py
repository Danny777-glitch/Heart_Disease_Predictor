import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.markdown("""
<style>
.stApp{
background:linear-gradient(135deg,#0f172a,#1e293b,#111827);
}
.title{text-align:center;font-size:40px;font-weight:bold;color:#ff4b4b;}
.subtitle{text-align:center;color:white;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">❤️ Heart Disease Prediction using KNN</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hyperparameter Tuning + Cross Validation</div>', unsafe_allow_html=True)

st.sidebar.title("Model")
st.sidebar.write("Algorithm: KNN")
st.sidebar.write("Best K: 9")
st.sidebar.write("Accuracy: 80.33%")
st.sidebar.write("CV Score: 83.00%")

c1, c2 = st.columns(2)

with c1:
    age = st.slider("Age", 20, 100, 50)

    sex = st.selectbox(
        "Gender",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-Anginal Pain",
            3: "Asymptomatic"
        }[x]
    )

    trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 250, 120)

    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

    fbs = st.selectbox(
        "Fasting Blood Sugar",
        options=[0, 1],
        format_func=lambda x: "≤ 120 mg/dL" if x == 0 else "> 120 mg/dL"
    )

    restecg = st.selectbox(
        "Resting ECG Result",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }[x]
    )

with c2:
    thalach = st.slider("Maximum Heart Rate", 60, 220, 150)

    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    oldpeak = st.slider("ST Depression (Old Peak)", 0.0, 6.5, 1.0, 0.1)

    slope = st.selectbox(
        "ST Segment Slope",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }[x]
    )

    ca = st.selectbox(
        "Major Blood Vessels",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: f"{x} Vessel(s)"
    )

    thal = st.selectbox(
        "Thalassemia Test",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Normal",
            1: "Fixed Defect",
            2: "Reversible Defect",
            3: "Unknown"
        }[x]
    )

if st.button("Predict", use_container_width=True):
    x = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    x = scaler.transform(x)
    pred = model.predict(x)[0]
    prob = model.predict_proba(x)[0]
    
    if pred == 1:
        st.error("Heart Disease Detected")
        st.progress(int(prob[1] * 100))
        st.metric("Risk", f"{prob[1] * 100:.2f}%")
        #st.balloons()
    else:
        st.success("No Heart Disease Detected")
        st.progress(int(prob[0] * 100))
        st.metric("Healthy", f"{prob[0] * 100:.2f}%")

    st.write("Heart Disease Probability:", f"{prob[1] * 100:.2f}%")
    st.write("No Heart Disease Probability:", f"{prob[0] * 100:.2f}%")
