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

# Assuming challenge_utils provides necessary functions directly
from challenge_utils import build_training_data, relative_squared_error, save_onnx, load_onnx

# Building training/testing set data
student_data_path = 'students_drahi_production_consumption_hourly.csv'
# Assume this function correctly returns aligned target_time, targets, and predictors
target_time, targets, predictors = build_training_data(student_data_path)

# Ensure the shape of predictors is compatible for polynomial transformation
if predictors.ndim == 1:
    predictors = predictors.reshape(-1, 1)

# Apply Polynomial Features transformation
ntot = len(targets)
x_all = predictors.reshape(ntot, -1)
poly = PolynomialFeatures(degree=2)
predictors_poly = poly.fit_transform(x_all)

# Ensure y_all matches the transformed predictors in size
y_all = targets  # Assuming targets are correctly aligned with the original predictors

# Now, we ensure the GridSearch and subsequent steps use matched predictors and targets
# Setup for GridSearchCV
param_grid = {'alpha': np.logspace(-2, 10, 13)}
grid = GridSearchCV(Ridge(), param_grid, scoring='neg_mean_squared_error', cv=5)

# Here we fit the model using the entire dataset, ensuring the sizes match
grid.fit(predictors_poly, y_all)

print(f"Best alpha from GridSearchCV: {grid.best_params_['alpha']}")

# Evaluate the best model found by GridSearchCV
best_model = grid.best_estimator_

# Assuming the evaluation and save/load process is handled correctly
# Save the model
save_onnx(best_model, 'polynomial_ridge_model.onnx')

# Load and evaluate the saved model
# This step assumes the load_onnx function correctly handles the model loading
# and that you have a mechanism to apply the same PolynomialFeatures transformation before prediction
loaded_model_predictions = load_onnx('polynomial_ridge_model.onnx', predictors_poly)
loaded_model_rse = relative_squared_error(loaded_model_predictions, y_all)
print(f"RSE for the loaded model: {loaded_model_rse}")
