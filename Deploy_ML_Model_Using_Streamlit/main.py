import numpy as np
import pickle
import streamlit as st

with open("C:/Users/Wei/Downloads/MLProjects/ML_YT/Deploy_ML_Model_Using_Streamlit/trained_model.sav", "rb") as f:
    loaded_model = pickle.load(f)
    loaded_scaler = pickle.load(f)


# create a function for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # standardised input data
    std_data = loaded_scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(std_data)

    if prediction[0] == 1:
        return "This person is diabetic"
    else:
        return "This person is not diabetic"


def main():
    # giving a title
    st.title("Diabetes Prediction App")

    # getting the input from the user
    pregnancies = st.text_input("Number of pregnancies:")
    glucose = st.text_input("Glucose level:")
    blood_pressure = st.text_input("Blood Pressure value:")
    skin_thickness = st.text_input("Skin Thickness value:")
    insulin = st.text_input("Insulin level:")
    BMI = st.text_input("BMI value:")
    diabetes_pedigree_function = st.text_input("Diabetes Pedigree Function value:")
    age = st.text_input("Age of the person:")

    # code for prediction
    diagnosis = ''

    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([pregnancies,
                                         glucose,
                                         blood_pressure,
                                         skin_thickness,
                                         insulin,
                                         BMI,
                                         diabetes_pedigree_function,
                                         age])

        st.success(diagnosis)


if __name__ == '__main__':
    main()
