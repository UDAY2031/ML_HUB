# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Import dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Step 2: Display first 5 rows
print("First 5 rows of the dataset:\n", df.head())

# Step 3: Check for null values
print("\nNull values in the dataset:\n", df.isnull().sum())

# Step 4: Visualize data
sns.pairplot(df.sample(500))  # Sample for visualization
plt.show()
sns.histplot(df['MedHouseVal'], kde=True, bins=30).set_title("Target Variable Distribution")
plt.show()

# Step 5: Covariance and Correlation
cov_matrix = df.cov()
correlation_matrix = df.corr()

print("\nCovariance Matrix (partial):\n", cov_matrix.head())
print("\nCorrelation Matrix (partial):\n", correlation_matrix.head())

# Correlation heatmap
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm').set_title("Correlation Heatmap")
plt.show()

# Step 6: Train-Test Split
X = df.drop(columns='MedHouseVal')  # Features
y = df['MedHouseVal']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train and evaluate model
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 8: Display results
print(f"\nMean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
