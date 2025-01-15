from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
d = load_iris()
X, y = d.data, d.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

# k-NN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Predictions
for i, (x, pred, actual) in enumerate(zip(X_test, y_pred, y_test)):
    if pred == actual:
        print(f"Correct: {x}, Predicted: {d.target_names[pred]}")
    else:
        print(f"Wrong: {x}, Predicted: {d.target_names[pred]}, Actual: {d.target_names[actual]}")
