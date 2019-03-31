import numpy as np
import pandas as pd
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


girls = ["Anagha", "Deepika", "Deepti", "Devyani", "Juhi", "Nehal", "Prachi", "Pragya", "Shiloni", "Sowmya", "Sravya", "Tripti"]
X = pd.read_csv('features/reps.csv',header=None)
df_labels = pd.read_csv('features/labels.csv',header=None)

y = df_labels.iloc[:,1].apply(lambda x: x.split('/')[-2]).apply(lambda x: 1 if x in girls else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

C_range = 10.0 ** np.arange(-3, 3)
gamma_range = 10.0 ** np.arange(-3, 3)
param_grid = dict(gamma = gamma_range.tolist(), C = C_range.tolist())
model = GridSearchCV(SVC(kernel='linear'), param_grid, cv = 4, n_jobs = -1)

model.fit(X_train,y_train)
score = model.score(X_test,y_test)
print(score)
