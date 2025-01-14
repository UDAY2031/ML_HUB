from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# 1. Import dataset
data = fetch_california_housing(as_frame=True).frame

# 2. Display first 5 rows
print(data.head())

# 3. Check for null values
print("\nNull values:\n", data.isnull().sum())

# 4. Visualize data
data.plot.scatter(x='MedInc', y='MedHouseVal', title="Median Income vs House Value")
plt.show()

# 5. Covariance and Correlation
print("\nCovariance (partial):\n", data.cov().iloc[:3, :3])  # Partial display
print("\nCorrelation (partial):\n", data.corr().iloc[:3, :3])  # Partial display

# 6. Train-test split
X = data.drop('MedHouseVal', axis=1)
y = data['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Gradient Descent
model = SGDRegressor(max_iter=1000, tol=1e-3).fit(X_train, y_train)

# 8. Predict and evaluate
y_pred = model.predict(X_test)
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
