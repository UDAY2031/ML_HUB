from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
d = load_iris()
x, y = d.data, d.target

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train KNN
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

# Predict and evaluate
y_pred = knn.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Predict a single sample
sample = x_test[0].reshape(1, -1)  # Take the first test sample
pred = knn.predict(sample)
print(f"Sample: {sample[0]}, Predicted Class: {d.target_names[pred[0]]}")
