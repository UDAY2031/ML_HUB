import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

df = pd.read_csv('heart.csv')
df['x'] = pd.qcut(df['age'],3,labels = ['y','m','o'])
df['y'] = pd.qcut(df['trestbps'],3,labels = ['l','n','h'])
df['z'] = pd.qcut(df['chol'],3,labels=['l','n','h'])
df['s'] = df['sex'].map({0:'F',1:'M'})
df['c'] = df['cp'].astype(str)
df['t'] = df['target'].astype(str)
df = df[['x','y','z','s','c','t']]

model = BayesianNetwork([('x','z'),('x','y'),('s','c'),('z','t'),('y','t'),('c','t')])
mle = MaximumLikelihoodEstimator(model,df)
model.add_cpds(*[mle.estimate_cpd(n) for n in model.nodes()])

if model.check_model():
    print('model ok')
    infer = VariableElimination(model)
    print(infer.query(['t'],evidence = {'x' :'o','y':'h','z':'h','s':'M','c':'3'}))
    
    c = 0
    for _,i in df.iterrows():
        pred = infer.query(['t'],evidence = {'x':i['x'],'y':i['y'],'z':i['z'],'s':i['s'],'c':i['c']})
        c += (pred.values[1]>0.5) == (i['t'] == '1')
        
    print(f"Accuracy : {c / len(df):.2f}")
else:
    print('error')
