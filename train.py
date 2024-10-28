from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

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
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': coefficients})

        important_features = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()

        X = X[important_features]

        current_column_count = len(important_features)

        # print(f"Số lượng cột sau khi lặp: {current_column_count}")

    return X

X_filtered= iterative_feature_elimination(X, y)
joblib.dump(X_filtered, 'data.pkl')


X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')

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

joblib.dump(linear_model, 'linear_model.pkl')
joblib.dump(ridge_model, 'ridge_model.pkl')
joblib.dump(mlp_model, 'mlp_model.pkl')
joblib.dump(stacking_model, 'stacking_model.pkl')

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

