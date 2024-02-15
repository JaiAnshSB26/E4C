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


