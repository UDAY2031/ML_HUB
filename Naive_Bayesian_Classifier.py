import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "Outlook": ["Sunny", "Sunny", "Rainy", "Sunny"],
    "Temperature": ["Warm", "Warm", "Cold", "Warm"],
    "Humidity": ["Normal", "High", "High", "High"],
    "Wind": ["Strong", "Strong", "Strong", "Strong"],
    "Water": ["Warm", "Warm", "Warm", "Cool"],
    "Forecast": ["Same", "Same", "Change", "Change"],
    "PlayTennis": ["Yes", "Yes", "No", "Yes"]
}

# Encode data
df = pd.DataFrame(data)
df = df.apply(LabelEncoder().fit_transform)

# Split data
X = df.drop(columns="PlayTennis")
y = df["PlayTennis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and predict
model = CategoricalNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy and results
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nActual vs Predicted:")
print(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}))
