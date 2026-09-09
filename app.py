import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open('log_CHD_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler_log.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(
    page_title='CHD',
    page_icon="🩺",
    layout='centered'
)

st.title("Coronary Heart Disease Prediction")
st.write("Enter Patient's Details below: ")

st.markdown('---')

# User Inputs
male = st.selectbox('Gender', [0, 1], 
                    format_func=lambda x: 'Female' if x==0 else 'Male')
age = st.number_input('Age', 18, 100, 45)
education = st.selectbox("Education Level",
                         [1, 2, 3, 4],
                         help = '1=Primary School, 2=High School, 3=College, 4=PG')
CurrentSmoker = st.selectbox('Current Smoker', [0, 1])
cigsPerDay = st.number_input('Cigarette Per Day',
                             min_value=0,
                             max_value=100,
                             value=0)
BPMeds = st.selectbox("BP Medication", [0, 1])
prevalentStroke = st.selectbox("Previous Stroke", [0, 1])
prevalentHyp = st.selectbox("Hypertension", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])
totChol = st.number_input("Total Cholesterol",
                          100,
                          700,
                          220
)
sysBP = st.number_input("Systolic BP", 70, 300, 120)
diaBP = st.number_input("Diastolic BP", 40, 200, 80)
BMI = st.number_input("BMI", 10.0, 70.0, 25.0)
heartRate = st.number_input("Heart Rate", 30, 200, 75)
glucose = st.number_input("Glucose", 40, 500, 80)

# Prediction
if st.button('Predict'):
    input_data = pd.DataFrame([[
        male,
        age,
        education,
        CurrentSmoker,
        cigsPerDay,
        BPMeds,
        prevalentStroke,
        prevalentHyp,
        diabetes,
        totChol,
        sysBP,
        diaBP,
        BMI,
        heartRate,
        glucose
    ]],
    columns=[
        'male',
        'age',
        'education',
        'currentSmoker',
        'cigsPerDay',
        'BPMeds',
        'prevalentStroke',
        'prevalentHyp',
         'diabetes',
        'totChol',
        'sysBP',
        'diaBP',
        'BMI',
        'heartRate',
        'glucose'
    ])
    # Scale
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown('---')

    if prediction[0] == 1:
        st.error("High Risk of Coronary Heart Disease")
        # st.snow()
    else:
        st.success("Low risk of Coronary Heart Disease")
        st.balloons()

    st.write(f"**Risk Probability : {probability*100:.2f}%**")