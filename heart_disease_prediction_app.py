import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open("C:/Users/Shantanu/Desktop/Machine Learning/heart_model.sav", 'rb'))

#creating the function for prediction
def heart_prediction(input_data):
    # change the data to a numpy array
    input_data_as_numpy_array = np.array(input_data, dtype=float)

    # reshape the numpy array as we are prediction for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        return"The person does not have a heart disease"
    else:
        return"The person has a heart disease"


def main():
    #Giving the title for the app
    st.title('Heart Disease Prediction Web App')

    #getting the input data from the user
    age=st.text_input("Age of the patient")
    sex = st.text_input("sex (1 = male; 0 = female)")
    cp = st.text_input("chest pain type  Value 0: typical angina Value 1: atypical angina Value 2: non-anginal pain Value 3: asymptomatic")
    trestbps = st.text_input(" resting blood pressure (in mm Hg on admission to the hospital)")
    chol = st.text_input("serum cholestoral in mg/dl")
    fbs = st.text_input("(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)")
    restecg = st.text_input("resting electrocardiographic results  Value 0: normal Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria")
    thalach = st.text_input("maximum heart rate achieved")
    exang = st.text_input("exercise induced angina (1 = yes; 0 = no)")
    oldpeak = st.text_input("ST depression induced by exercise relative to rest")
    slope = st.text_input("the slope of the peak exercise ST segment")
    ca = st.text_input("number of major vessels (0-3) colored by flourosopy")
    thal = st.text_input("thal: 1 = normal; 2 = fixed defect; 3 = reversable defect")

    #code for prediction
    diagnosis=''

    #creating a button for prediction
    if st.button("Heart Disease Test Result"):
        diagnosis=heart_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])

    st.success(diagnosis)

if __name__ == "__main__":
    main()