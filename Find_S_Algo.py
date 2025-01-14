import pandas as pd

# Load dataset
data = pd.read_csv("your_dataset.csv")
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Target variable

# Find-S Algorithm
def find_s(X, y):
    s = None  # Initialize hypothesis
    for i in range(len(y)):
        if y[i] == "Yes":  # Check for positive examples
            if s is None:
                s = X[i].copy()  # Set the first positive example as the hypothesis
            else:
                # Generalize hypothesis for mismatched attributes
                s = [h if h == x else '?' for h, x in zip(s, X[i])]
    return s

# Apply Find-S
print("Most Specific Hypothesis:", find_s(X, y))
