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





data = pd.read_csv('score-mat.csv', sep=',')
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])
    label_ecoders = le


target = 'G3'
features = [col for col in data.columns if col!=target]
X = data[features]
y = data['G3']

scaler = StandardScaler()

linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
mlp_model = MLPRegressor(hidden_layer_sizes=(128,128),activation='relu', max_iter=1500, early_stopping=True, random_state=42)

def iterative_feature_elimination(X, y, threshold=0):
    prev_column_count = 0  
    current_column_count = X.shape[1]  

    while current_column_count != prev_column_count:
        prev_column_count = current_column_count

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ridge_model.fit(X_train_scaled, y_train)

        coefficients = ridge_model.coef_

        # features = X_filtered.columns
        # indices = np.argsort(coefficients)

        # plt.figure(figsize=(10, 6))
        # plt.title('Feature Importance (Linear Regression Coefficients)')
        # plt.barh(range(len(indices)), coefficients[indices], color='b', align='center')
        # plt.yticks(range(len(indices)), [features[i] for i in indices])
        # plt.xlabel('Coefficient Value')
        # plt.show()

        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': coefficients})

        important_features = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()

        X = X[important_features]

        current_column_count = len(important_features)

        # print(f"Số lượng cột sau khi lặp: {current_column_count}")

    return X, coefficients

X_filtered ,coef = iterative_feature_elimination(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#train model
linear_model.fit(X_train_scaled, y_train)
ridge_model.fit(X_train_scaled, y_train)
mlp_model.fit(X_train_scaled, y_train)

#stacking
estimators = [
    ('linear', linear_model),
    ('mlp', mlp_model)
]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0))
stacking_model.fit(X_train_scaled, y_train)


def score_prediction(input_data, model):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]
    return prediction

def nse(observed, predicted):
    return 1 - (np.sum((observed - predicted)**2) / np.sum((observed - np.mean(predicted))**2))

def evaluate_model(model):
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    nses = nse(y_test, y_pred) 
    return mse, r2, mae, nses

print("\t\tmse\t\tr2\t\tmae\t\tnse")    
print(f"linear: {evaluate_model(linear_model)}")
print(f"ridge: {evaluate_model(ridge_model)}")
print(f"mlp: {evaluate_model(mlp_model)}")
print(f"stack: {evaluate_model(stacking_model)}")


import streamlit as st
import requests

# Tiêu đề ứng dụng
st.title('Prediction Using ML Models')

# Tạo form để nhập dữ liệu
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

# Khi người dùng nhấn nút "Predict"
if st.button('Predict'):
    # Tạo một đối tượng chứa các dữ liệu từ form
    input_data = {
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
    
    
    linear_pred = score_prediction(input_data,linear_model)
    ridge_pred = score_prediction(input_data,ridge_model)
    mlp_pred = score_prediction(input_data,mlp_model)
    stacking_pred = score_prediction(input_data,stacking_model)

        
    # Hiển thị kết quả dự đoán từ các model
    st.subheader('Prediction Results:')
    st.write(f'Linear Model Prediction: {linear_pred[0]}')
    st.write(f'Ridge Model Prediction: {ridge_pred[0]}')
    st.write(f'MLP Model Prediction: {mlp_pred[0]}')
    st.write(f'Stacking Model Prediction: {stacking_pred[0]}')
    
