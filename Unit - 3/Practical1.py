'''
Unit -3 -> Supervised Learning – Classification :

Practical – 1 :  Write a python code to apply Naive Bayesian algorithm to classify that whether a personcan buy computer or not based on given test data :

'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB

# 1. Load the dataset
df = pd.read_excel("D:\\ML_4090\\Unit-3\\Buying_or_not_computer_dataset.xls")

# 2. Separate features (X) and target (y)
X = df.drop(columns=["Buyscomputer"])
y = df["Buyscomputer"]

# 3. Encode text categories into numbers
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# 4. Train the Naive Bayes Model
# CategoricalNB is perfect for text/category-based data like this
model = CategoricalNB()
model.fit(X_encoded, y)

# 5. Define a new test person to predict
# Let's predict for: Age='Youth', Income='Medium', Student='Yes', Creditrating='Fair'
test_data = pd.DataFrame(
    [["Youth", "Medium", "Yes", "Fair"]],
    columns=["Age", "Income", "Student", "Creditrating"],
)

# Encode the test data using the same rules
test_encoded = encoder.transform(test_data)

# 6. Make the prediction
prediction = model.predict(test_encoded)
print(f"Prediction for the test person: Will they buy a computer? -> {prediction[0]}")
