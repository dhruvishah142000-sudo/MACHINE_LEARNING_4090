'''
Practical – 2 :Write a python code to implement decision tree for below given dataset. Identify Job offered or not
'''

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

# 1. Load the dataset (using a clean path with double backslashes)
df = pd.read_excel("D:\\ML_4090\\Unit-3\\Job_Offered_or_not_dataset.xls")

# 2. Separate features (X) and target (y)
X = df.drop(columns=["Job_offer"])
y = df["Job_offer"]

# 3. Encode text categories into numbers
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# 4. Train the Decision Tree Classifier
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_encoded, y)

# 5. Define a new test case to predict
# Let's predict for: CGPA='High', Communication='Good', Apptitude='High', Programming_skill='Good'
test_data = pd.DataFrame(
    [["High", "Good", "High", "Good"]],
    columns=["CGPA", "Communication ", "Apptitude", "Programming_skill"],
)

# Encode the test data using the exact same rules
test_encoded = encoder.transform(test_data)

# 6. Make the prediction
prediction = model.predict(test_encoded)
print(f"Prediction for the candidate: Will they get a Job Offer? -> {prediction[0]}")
