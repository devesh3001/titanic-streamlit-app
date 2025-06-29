import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details to predict survival:")

# Input features
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 80, 25) 
Fare = st.slider("Fare", 0.0, 600.0, 32.0)

# Feature transformation
sex_encoded = 0 if Sex == "male" else 1

# Prediction
input_data = pd.DataFrame([[Pclass, sex_encoded, Age, Fare]],
                          columns=['Pclass', 'Sex', 'Age', 'Fare'])

if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 The passenger is likely to **Survive**!")
    else:
        st.error("💀 The passenger is likely to **Not Survive**.")
