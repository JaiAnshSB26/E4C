#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 19:39:05 2024

@author: kaylcolardelle
"""

# Import necessary libraries
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

# Local import
from challenge_utils import build_training_data, relative_squared_error, train_test_split, save_onnx, load_onnx

# Building training/testing set data
student_data_path = 'students_drahi_production_consumption_hourly.csv'
target_time, targets, predictors = build_training_data(student_data_path)

# Since predictors and targets are directly obtained, we consider them as x_all and y_all for clarity
x_all = predictors
y_all = targets

# Generate Polynomial Features for the entire dataset
degree = 2  # Example degree
poly = PolynomialFeatures(degree=degree)
x_all_poly = poly.fit_transform(x_all.reshape(-1, x_all.shape[-1]))

# Separating train/test sets with polynomial features
n = 250
test_ind = np.arange(n, len(y_all))
x_train, y_train, x_test, y_test = train_test_split(x_all_poly, y_all, test_ind)

# Update model training to use polynomial features
reg = Ridge(alpha=1e8)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

# Calculate and print RSE for the trained model
RelativeMSE = relative_squared_error(y_pred, y_test)
print('Trained polynomial model RSE:', RelativeMSE)

# Searching for optimal coefficients (validation) with polynomial features
param_grid = {'alpha': np.logspace(-2, 10, 21)}
grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(x_all_poly, y_all)  # Use the polynomial features version for validation

best_alpha = grid.best_params_['alpha']
best_rse = -grid.best_score_ / np.mean((y_all - y_all.mean()) ** 2)
print('Best alpha from validation:', best_alpha)

best_estimator = grid.best_estimator_
# Predict using the best estimator on the entire dataset with polynomial features
y_pred = best_estimator.predict(x_all_poly)
RelativeMSE = relative_squared_error(y_pred, y_all)
print('Best linear model RSE:', RelativeMSE)

# Save best version of the model
print('Saving in ONNX format')
save_onnx(best_estimator, 'linear_model.onnx', x_all_poly)  # Note: Ensure this step is compatible with your saving function

# Load and run saved simple model
# Ensure to use polynomial-transformed data when predicting with the loaded model
y_pred_onnx = load_onnx('linear_model.onnx', x_all_poly)
RelativeMSE = relative_squared_error(y_pred_onnx, y_all)
print('Loaded from ONNX file RSE:', RelativeMSE)
