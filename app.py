import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from flask import Flask, render_template, request, redirect, url_for
import joblib
import os
import warnings
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import streamlit as st


scaler = joblib.load('scaler.pkl')

linear_model = joblib.load('linear_model.pkl')
ridge_model = joblib.load('ridge_model.pkl')
mlp_model = joblib.load('mlp_model.pkl')
stacking_model = joblib.load('stacking_model.pkl')



def score_prediction(input_data, model):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]
    return prediction

def nse(observed, predicted):
    return 1 - (np.sum((observed - predicted)**2) / np.sum((observed - np.mean(predicted))**2))

# def evaluate_model(model):
#     y_pred = model.predict(X_test_scaled)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     nses = nse(y_test, y_pred) 
#     return mse, r2, mae, nses

# print("\t\tmse\t\tr2\t\tmae\t\tnse")    
# print(f"linear: {evaluate_model(linear_model)}")
# print(f"ridge: {evaluate_model(ridge_model)}")
# print(f"mlp: {evaluate_model(mlp_model)}")
# print(f"stack: {evaluate_model(stacking_model)}")


models = {
    'Linear': linear_model,
    'Ridge': ridge_model,
    'MLP': mlp_model,
    'Stacking': stacking_model,
    
}
# Tiêu đề ứng dụng
st.title('Prediction Using ML Models')

# Tạo form để nhập dữ liệu
with st.form(key='my_form'):
    # Nhập dữ liệu
    school = st.selectbox('School:', ['GP (Gabriel Pereira)', 'MS (Mousinho da Silveira)'])
    school = 1 if school == 'GP (Gabriel Pereira)' else 0

    gender = st.selectbox('Gender:', ['Male', 'Female'])
    gender = 1 if gender == 'Male' else 0

    traveltime = st.number_input('Travel Time (1-4 hours):', min_value=1, max_value=4, step=1)

    schoolsup = st.selectbox('School Support:', ['Yes', 'No'])
    schoolsup = 1 if schoolsup == 'Yes' else 0

    famsup = st.selectbox('Family Support:', ['Yes', 'No'])
    famsup = 1 if famsup == 'Yes' else 0

    famrel = st.number_input('Family Relations (1-5):', min_value=1, max_value=5, step=1)
    goout = st.number_input('Going Out (1-5):', min_value=1, max_value=5, step=1)
    health = st.number_input('Health (1-5):', min_value=1, max_value=5, step=1)
    absences = st.number_input('Absences (0-93):', min_value=0, max_value=93, step=1)
    G1 = st.number_input('G1 (0-20):', min_value=0, max_value=20, step=1)
    G2 = st.number_input('G2 (0-20):', min_value=0, max_value=20, step=1)

    # Nút submit
    submit_button = st.form_submit_button(label='Predict')

# Khi người dùng nhấn nút "Predict"
if submit_button:
    # Tạo một đối tượng chứa các dữ liệu từ form
    input = {
        'school': school,
        'gender': gender,
        'traveltime': traveltime,
        'schoolsup': schoolsup,
        'famsup': famsup,
        'famrel': famrel,
        'goout': goout,
        'health': health,
        'absences': absences,
        'G1': G1,
        'G2': G2,
    }

    input_data = np.array(list(input.values())).reshape(1, -1)

    # Dự đoán
    linear_pred = score_prediction(input_data, linear_model)
    ridge_pred = score_prediction(input_data, ridge_model)
    mlp_pred = score_prediction(input_data, mlp_model)
    stacking_pred = score_prediction(input_data, stacking_model)

    # Hiển thị kết quả dự đoán từ các model
    st.subheader('Prediction Results:')
    st.write(f'Linear Model Prediction: {linear_pred}')
    st.write(f'Ridge Model Prediction: {ridge_pred}')
    st.write(f'MLP Model Prediction: {mlp_pred}')
    st.write(f'Stacking Model Prediction: {stacking_pred}')