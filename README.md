# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries (numpy, pandas, sklearn).

2. Load the dataset (50_Startups.csv).

3. Split into features X and target y.

4. Encode categorical data (OneHotEncoder).

5. Scale features using StandardScaler.

6. Add intercept term and initialize θ as zeros.

7. For given iterations:

   Compute predictions (X·θ).
   Calculate error (pred - y).
   Compute gradient.
   Update parameters (θ = θ – α * gradient).

8. Get final θ values after training.

9. Preprocess new input data.

10. Predict profit using learned θ. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Anu Nivitha U
RegisterNumber: 212223040016
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def linear_regression(X, y, iters=1000, learning_rate=0.01):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept term
    theta = np.zeros((X.shape[1], 1))
    
    for _ in range(iters):
        predictions = X.dot(theta)
        errors = predictions - y.reshape(-1, 1)
        gradient = (1 / X.shape[0]) * X.T.dot(errors)
        theta -= learning_rate * gradient
    
    return theta

data = pd.read_csv('50_Startups.csv', header=0)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), [3])  
], remainder='passthrough')

X = ct.fit_transform(X)

y = y.astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

theta = linear_regression(X_scaled, y, iters=1000, learning_rate=0.01)

new_data = np.array([165349.2, 136897.8, 471784.1, 'New York']).reshape(1, -1)  # Example new data
new_data_scaled = scaler.transform(ct.transform(new_data))

new_prediction = np.dot(np.append(1, new_data_scaled), theta)

print(f"Predicted value: {new_prediction[0]}")
data.head()

```

## Output:
<img width="829" height="378" alt="Screenshot 2025-10-04 at 10 12 02 AM" src="https://github.com/user-attachments/assets/4d729c4a-7656-4c1f-bfd6-638bd2937048" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
