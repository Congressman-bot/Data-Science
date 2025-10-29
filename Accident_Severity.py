# Code written by Alex Mwangi

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

# Creating a dataset
np.random.seed(42)
data_size = 100

Vehicle_Speed = np.random.randint(30, 130, data_size)
Weather_Condition = np.random.choice([0, 1, 2], data_size)
Road_Surface = np.random.choice([0, 1], data_size)
Driver_Age = np.random.randint(18, 70, data_size)
Visibility = np.random.randint(50, 1000, data_size)

Accident_Severity = (
    0.05 * Vehicle_Speed +
    1.5 * Weather_Condition +
    2 * Road_Surface -
    0.03 * Visibility -
    0.02 * Driver_Age +
    np.random.normal(0, 2, data_size)
)

# Creating a DataFrame (no spaces in column names)
df = pd.DataFrame({
    'Vehicle_Speed': Vehicle_Speed,
    'Weather_Condition': Weather_Condition,
    'Road_Surface': Road_Surface,
    'Driver_Age': Driver_Age,
    'Visibility': Visibility,
    'Accident_Severity': Accident_Severity
})

print("Sample of the dataset:")
display(df.head())

# Define features (X) and target (y)
X = df[['Vehicle_Speed', 'Weather_Condition', 'Road_Surface', 'Driver_Age', 'Visibility']]
y = df['Accident_Severity']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluating model performance
y_pred = model.predict(X_test)
print(f"\nModel Performance:")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}")

# Saving the model
joblib.dump(model, 'accident_severity_model.pkl')
print("\nModel saved as 'accident_severity_model.pkl'")

# Example prediction
example = pd.DataFrame({
    'Vehicle_Speed': [90],
    'Weather_Condition': [1],  # Rainy
    'Road_Surface': [1],       # Wet
    'Driver_Age': [30],
    'Visibility': [200]
})

predicted_severity = model.predict(example)[0]
print("\nPredicted Accident Severity (Scale 1–10):", round(predicted_severity, 2))

# Display coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
display(coefficients)

