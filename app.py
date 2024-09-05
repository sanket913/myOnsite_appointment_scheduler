import streamlit as st
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data_preprocessing import load_and_preprocess_data 

# Load 
def load_model():
    with open("models/schedule_optimizer.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def predict_show(model, patient_age, scheduled_day, appointment_day, gender):
    prediction = model.predict_proba([[patient_age, scheduled_day, appointment_day, gender]])[0][1]
    return prediction

def patient_info_form():
    st.write("## Patient Information")
    patient_age = st.slider("Patient Age", 18, 75, 30)

    gender = st.selectbox("Gender", ["Female", "Male"])
    gender_encoded = 0 if gender == "Female" else 1

    return patient_age, gender_encoded

def show_calendar():
    st.write("## Select Appointment Dates")
    appointment_date = st.date_input("Appointment Date")
    scheduled_date = st.date_input("Scheduled Date")

    if scheduled_date > appointment_date:
        st.error("Scheduled date cannot be after the appointment date.")
        return None, None
    scheduled_day = scheduled_date.weekday()
    appointment_day = appointment_date.weekday()

    return scheduled_day, appointment_day, appointment_date, scheduled_date

def display_summary(patient_age, gender, scheduled_date, appointment_date):
    st.write("## Appointment Summary")
    st.write(f"Patient Age: {patient_age}")
    st.write(f"Gender: {'Female' if gender == 0 else 'Male'}")
    st.write(f"Scheduled Date: {scheduled_date}")
    st.write(f"Appointment Date: {appointment_date}")

def main():
    st.title("Phlebotomy Appointment Scheduling")
    patient_age, gender = patient_info_form()
    scheduled_day, appointment_day, appointment_date, scheduled_date = show_calendar()

    if scheduled_day is not None and appointment_day is not None:
        model = load_model()
        show_up_prob = predict_show(model, patient_age, scheduled_day, appointment_day, gender)
        if show_up_prob > 0.7:
            st.success(f"Appointment likely to succeed! Show-up probability: {show_up_prob:.2f}")
        else:
            st.warning(f"Risk of no-show! Show-up probability: {show_up_prob:.2f}")

        display_summary(patient_age, gender, scheduled_date, appointment_date)

        if st.button("Confirm Appointment"):
            st.success(f"Appointment confirmed for {appointment_date}!")
    else:
        st.warning("Please ensure the scheduled date is before or on the appointment date.")

if __name__ == "__main__":
    main()
