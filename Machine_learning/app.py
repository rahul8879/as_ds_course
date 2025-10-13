# lets build very application using streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# loading the model
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)
st.title("KNN Classifier for Loan Default Prediction")
st.write("This app predicts whether a person will default on their loan based on their income and credit score.")   
income = st.number_input("Enter Income:", min_value=0)
credit_score = st.number_input("Enter Credit Score:", min_value=0, max_value=850)
if st.button("Predict"):
    input_data = np.array([[income, credit_score]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("The person is likely to default on their loan.")
    else:
        st.success("The person is unlikely to default on their loan.")