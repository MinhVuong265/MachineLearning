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
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

scaler = StandardScaler()
data = joblib.load('data.pkl')

scaler.fit_transform(data)

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
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    input_data = request.json
    school = float(input_data['school'])
    gender = float(input_data['gender'])
    traveltime = float(input_data['traveltime'])
    schoolsup = float(input_data['schoolsup'])
    famsup = float(input_data['famsup'])
    famrel = float(input_data['famrel'])
    goout = float(input_data['goout'])
    health = float(input_data['health'])
    absences = float(input_data['absences'])
    G1 = float(input_data['G1'])
    G2 = float(input_data['G2'])

    # Tạo một numpy array từ dữ liệu nhập
    input_features = np.array([[school,gender, traveltime, schoolsup, famsup, famrel, goout, health, absences, G1, G2]])

    # Chuẩn hóa dữ liệu đầu vào
    input_scaled = scaler.transform(input_features)

    # Dự đoán với các mô hình
    linear_pred = linear_model.predict(input_scaled)[0]
    ridge_pred = ridge_model.predict(input_scaled)[0]
    mlp_pred = mlp_model.predict(input_scaled)[0]
    stacking_pred = stacking_model.predict(input_scaled)[0]

    # Trả kết quả về client
    return jsonify({
        'linear': linear_pred.round(2),
        'ridge': ridge_pred.round(2),
        'mlp': mlp_pred.round(2),
        'stacking': stacking_pred.round(2),
    })
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000)) 
    app.run(debug=True, host='0.0.0.0', port=port)
