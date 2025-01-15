import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data
d = load_iris()
x, y = d.data[:, :2], d.target  # Use the first two features

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train SVM
svm = SVC(kernel='linear')
svm.fit(x_train, y_train)

# Predict and calculate accuracy
y_pred = svm.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Plot decision boundaries
xx, yy = np.meshgrid(np.linspace(x[:, 0].min(), x[:, 0].max(), 200),
                     np.linspace(x[:, 1].min(), x[:, 1].max(), 200))
z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y, palette='muted', s=40)
plt.contour(xx, yy, z, levels=np.unique(y), colors='k', linestyles='--', linewidths=0.5)
plt.show()
