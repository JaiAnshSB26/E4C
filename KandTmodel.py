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

# Correctly handle the reshaping for polynomial features transformation
# Ensure we're using the correct shape for predictors before transformation
predictors_reshaped = predictors.reshape(-1, predictors.shape[-1]) if len(predictors.shape) > 1 else predictors.reshape(-1, 1)

# Generate Polynomial Features for the reshaped predictors
degree = 2  # Example degree
poly = PolynomialFeatures(degree=degree)
predictors_poly = poly.fit_transform(predictors_reshaped)

# Here we assume targets and predictors are already aligned correctly
x_all_poly = predictors_poly
y_all = targets

# Assuming the previous split (train_test_split) is correct and doesn't need to be revised
# Directly move to the model fitting and validation phase with correct shapes

# Model fitting and validation
param_grid = {'alpha': np.logspace(-2, 10, 21)}
grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(x_all_poly, y_all)  # Ensure this uses the entire dataset correctly

best_alpha = grid.best_params_['alpha']
print('Best alpha from validation:', best_alpha)

best_estimator = grid.best_estimator_
# Ensure we're using the correct dataset for predictions
y_pred = best_estimator.predict(x_all_poly)
RelativeMSE = relative_squared_error(y_pred, y_all)
print('Best linear model RSE:', RelativeMSE)

# Save the model
print('Saving in ONNX format')
save_onnx(best_estimator, 'linear_model.onnx', x_all_poly)

# Load and run saved model for a consistency check
y_pred_onnx = load_onnx('linear_model.onnx', x_all_poly)
RelativeMSE = relative_squared_error(y_pred_onnx, y_all)
print('Loaded from ONNX file RSE:', RelativeMSE)
