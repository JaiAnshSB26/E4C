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

# Generate Polynomial Features
degree = 2  # Example degree
poly = PolynomialFeatures(degree=degree)
predictors_poly = poly.fit_transform(predictors.reshape(-1, predictors.shape[-1]))

# separating train/test sets with polynomial features
n = 250
test_ind = np.arange(n, len(targets))
x_train, y_train, x_test, y_test = train_test_split(predictors_poly, targets, test_ind)

# Update model training to use polynomial features
reg = Ridge(alpha=1e8)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

#  ──────────────────────────────────────────────────────────────────────────
# Apply polynomial transformation to the entire dataset for consistency
x_all_poly = poly.transform(x_all.reshape(-1, x_all.shape[-1]))

# Searching for optimal coefficients (validation) with polynomial features
param_grid = {'alpha': np.logspace(-2, 10, 21)}
grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(x_all_poly, y_all)  # Use the transformed features

best_alpha = grid.best_params_['alpha']
# Recalculate RSE based on polynomial features
best_rse = -grid.best_score_/np.mean((y_all - y_all.mean()) ** 2)
print('Best alpha from validation:', best_alpha)

best_estimator = grid.best_estimator_
# Predictions also need to be done on the polynomial-transformed features
y_pred = best_estimator.predict(x_all_poly)
RelativeMSE = relative_squared_error(y_pred, y_all)
print('Best linear model RSE:', RelativeMSE)

# Save best version of simple model
# Note: When saving and later loading the model, ensure that the input data is transformed with the same polynomial features
print('Saving in ONNX format')
# This step assumes the model pipeline includes polynomial feature transformation or that it's applied to the data before prediction
save_onnx(best_estimator, 'linear_model.onnx', x_train_poly)  # Note: Adjust saving function if necessary to handle polynomial features

# Load and run saved simple model
# Ensure to transform the input data with polynomial features before prediction
y_pred_onnx = load_onnx('linear_model.onnx', x_all_poly)
RelativeMSE = relative_squared_error(y_pred_onnx, y_all)
print('Loaded from ONNX file RSE:', RelativeMSE)

