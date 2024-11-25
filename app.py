import streamlit as st
import numpy as np
import pickle

# Set the background image
background_image_url = "https://img.freepik.com/free-photo/medical-banner-with-doctor-wearing-goggles_23-2149611193.jpg"  
page_bg = f"""
<style>
.stApp {{
    background-image: url("{background_image_url}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Load the trained model
model = pickle.load(open("trained_model.sav", "rb"))

# App title and description
st.title("Diabetes Prediction Model")
st.subheader("Enter the required details to check if the person is diabetic:")

# Input fields for user data
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300)
blood_pressure = st.number_input("Blood Pressure Level", min_value=0, max_value=200)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=100.0, step=0.1)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=10.0, step=0.01)
age = st.number_input("Age", min_value=0, max_value=120, step=1)

# Predict button
if st.button("Predict"):
    # Convert inputs to numpy array
    input_data = np.asarray([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])
    input_data_reshaped = input_data.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data_reshaped)
    
    # Display the result
    if prediction[0] == 0:
        st.success("The person is not diabetic.")
    else:
        st.error("The person is diabetic.")
