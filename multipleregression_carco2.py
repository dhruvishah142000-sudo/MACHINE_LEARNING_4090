from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv("D:\\ML_4090\\16_07_26\\cardata.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# 3. Create and Fit the model (cleaned up duplicate instantiations)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Printing the model coefficients
print('Intercept: ', reg_model.intercept_)

# 4. Predict on training and test sets
y_pred = reg_model.predict(X_test)  
x_pred = reg_model.predict(X_train)

print("Prediction for test set: {}".format(y_pred))

reg_model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})

# 5. Evaluate the model (Updated to modern root_mean_squared_error function)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = metrics.root_mean_squared_error(y_test, y_pred) 

print('Mean Absolute Error:', mae)
print('Mean Square Error:', mse)
print('Root Mean Square Error:', rmse)

# FIXED LINE: Pass custom input data as a DataFrame with matching feature names
custom_input = pd.DataFrame([[2300, 1300]], columns=['Weight', 'Volume'])
predictedCO2 = reg_model.predict(custom_input)

print("Predicted CO2 for Weight 2300 & Volume 1300:", predictedCO2)
