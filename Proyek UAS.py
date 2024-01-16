import streamlit as st
import pickle
import numpy as np
import time
from sklearn.preprocessing import StandardScaler

# Load your trained Random Forest model
model = pickle.load(open('xbg.pkl', 'rb'))

# Load StandardScaler from pickle file
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define a function to preprocess user input
def preprocess_input(input_data):
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    
    # Standardize the input data using the loaded scaler
    standardized_data = scaler.transform(input_data_as_numpy_array)

    return standardized_data

# Streamlit page config
st.set_page_config(
    page_title = "Heart Disease Prediction",
    page_icon = '❤️'
)

# Streamlit application layout
st.title('Heart Disease Prediction')

# Input fields for user data
age = st.number_input('Age', min_value=0, max_value=120, value=30)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
cp = st.selectbox('Chest Pain Type', options=[1, 2, 3, 4])
trestbps = st.number_input('Resting Blood Pressure', min_value=0)
chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0)
exang = st.selectbox('Exercise Induced Angina', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Predict button
if st.button('Predict Heart Disease'):

    # create progres bar widget with initial progress is 0%
    bar = st.progress(0)
        # create an empty container or space
    status_text = st.empty()
    for i in range(1,101):
        # create a text to showing a percentage process
        status_text.text("%i%% complete" %i)
        # give bar progress values
        bar.progress(i)
        # give bar progress time to execute the values
        time.sleep(0.01)

    # Preprocess input
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]
    preprocessed_input = preprocess_input(input_data)
    
    # Make prediction
    prediction = model.predict(preprocessed_input)

    # Determine the color based on the prediction result
    color = 'red' if prediction[0] == 1 else 'green'

    # Display prediction with color using Markdown
    prediction_result = 'Presence of Heart Disease' if prediction[0] == 1 else 'No Heart Disease'
    st.markdown(f'<p style="color:{color}; font-size:20px">{prediction_result}</p>', unsafe_allow_html=True)

    