import pandas as pd
import numpy as np

def ce(data):
    x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values  # Extract features and labels
    s = x[0].copy()  # Initialize specific hypothesis with the first instance
    g = [["?" for _ in range(len(s))] for _ in range(len(s))]  # Initialize general hypotheses
    
    for i, instance in enumerate(x):
        if y[i] == "Yes":  # Positive instance
            for j in range(len(s)):
                if s[j] != instance[j]:
                    s[j] = "?"  # Generalize the specific hypothesis
        else:  # Negative instance
            for j in range(len(instance)):
                if s[j] != instance[j]:
                    g[j][j] = s[j]  # Update general hypotheses with more specific values
                else:
                    g[j][j] = "?"
    
    # Remove duplicates in general_hypotheses
    g = [h for h in g if any(f != "?" for f in h)]  # Remove fully generalized hypotheses
    
    return s, g

data = pd.DataFrame({
    'O':['Sunny','Sunny','Rainy','Sunny'],
    'T':['Warm','Warm','Cool','Warm'],
    'H':['Normal','High','High','High'],
    'W':['Strong','Strong','Strong','Strong'],
    'A':['Warm','Warm','Warm','Cool'],
    'F':['Same','Same','Change','Change'],
    'P':["Yes",'Yes','No','Yes']
})

# Run the algorithm
s, g = ce(data)

# Output the specific hypothesis
print("Specific hypothesis:\n", s)

# Output the general hypotheses
print("General Hypothesis:\n")
for h in g:
    print(h)
