from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset and split into training and testing sets
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train the decision tree using ID3 algorithm
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Display tree structure and evaluate model
print("Decision Tree Rules:\n", export_text(model, feature_names=data.feature_names))
print(f"Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

# Classify a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]
print(f"Predicted class: {data.target_names[model.predict(sample)[0]]}")
