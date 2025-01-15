import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load and preprocess data
df = pd.read_csv('heart.csv')
df['a'] = pd.qcut(df['age'], 3, labels=['young', 'middle', 'elderly'])
df['b'] = pd.qcut(df['trestbps'], 3, labels=['low', 'normal', 'high'])
df['c'] = pd.qcut(df['chol'], 3, labels=['low', 'normal', 'high'])
df = df[['a', 'sex', 'cp', 'b', 'c', 'target']]
df['sex'] = df['sex'].map({0: 'F', 1: 'M'})
df['cp'] = df['cp'].astype(str)
df['target'] = df['target'].astype(str)

# Create Bayesian Network
model = BayesianNetwork([('a', 'c'), ('a', 'b'), ('sex', 'c'),
                         ('sex', 'cp'), ('c', 'target'), ('b', 'target'),
                         ('cp', 'target')])

# Train model
mle = MaximumLikelihoodEstimator(model, df)
cpds = [mle.estimate_cpd(node) for node in model.nodes()]
model.add_cpds(*cpds)

if model.check_model():
    print("Bayesian Network successfully created and validated")

    # Make inference for a sample
    evidence = {'a': 'elderly', 'sex': 'M', 'cp': '3', 'b': 'high', 'c': 'high'}
    infer = VariableElimination(model)
    result = infer.query(variables=['target'], evidence=evidence)
    print("\nProbability Distribution for Heart Disease given evidence:")
    print(result)

    # Model accuracy
    correct, total = 0, len(df)
    for _, row in df.iterrows():
        evidence = {'a': row['a'], 'sex': row['sex'], 'cp': row['cp'],
                    'b': row['b'], 'c': row['c']}
        result = infer.query(variables=['target'], evidence=evidence)
        predicted_class = '1' if result.values[1] > 0.5 else '0'
        correct += int(predicted_class == row['target'])

    accuracy = correct / total
    print(f"\nModel Accuracy: {accuracy:.2f}")
else:
    print("Error: Invalid Bayesian Network")
