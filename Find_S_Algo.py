import pandas as pd

# Load dataset into a DataFrame
data = pd.DataFrame({
    'O': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],  # Outlook
    'T': ['Warm', 'Warm', 'Cool', 'Warm'],      # Temperature
    'H': ['Normal', 'High', 'High', 'High'],    # Humidity
    'W': ['Strong', 'Strong', 'Strong', 'Strong'],  # Wind
    'A': ['Warm', 'Warm', 'Warm', 'Cool'],      # Air quality
    'F': ['Same', 'Same', 'Change', 'Change'],  # Forecast
    'P': ['Yes', 'Yes', 'No', 'Yes']            # Play (target variable)
})

# Separate features (X) and target variable (y)
X = data.iloc[:, :-1].values  # Features (all columns except the last)
y = data.iloc[:, -1].values   # Target variable (last column)

# Find-S Algorithm
def find_s(X, y):
    s = None  # Initialize hypothesis as None
    for i in range(len(y)):
        if y[i] == "Yes":  # Consider only positive examples
            if s is None:
                s = X[i].copy()  # First positive example becomes the initial hypothesis
            else:
                # Generalize hypothesis: Replace mismatched attributes with '?'
                s = [h if h == x else '?' for h, x in zip(s, X[i])]
    return s

# Apply the Find-S algorithm and display the result
print("Most Specific Hypothesis:", find_s(X, y))
