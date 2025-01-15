import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# Load and normalize data
data = load_iris()
X = MinMaxScaler().fit_transform(data.data[:, :1])  # Normalize Sepal Length
y = data.data[:, 1]  # Sepal Width

# Kernel and LWR computation
tau = 0.1  # Bandwidth
y_pred = []
for xi in X:
    weights = np.exp(-np.sum((X - xi) ** 2, axis=1) / (2 * tau ** 2))
    W = np.diag(weights)
    theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    y_pred.append(xi @ theta)

# Plotting
sns.scatterplot(x=X.ravel(), y=y, color='blue', label='Data')
sns.lineplot(x=X.ravel(), y=y_pred, color='red', label='LWR Fit')
plt.title('Locally Weighted Regression on Iris Dataset')
plt.xlabel('Normalized Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()
