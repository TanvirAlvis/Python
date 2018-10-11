import pandas as pd
df = pd.read_csv('dset/aps_failure_training_set.csv', skiprows=21, header=None, na_values='na')

l = df.iloc[:,0].values
X = df.iloc[0:100,1:]

from fancyimpute import KNN

X_filled_knn = KNN(k=3).complete(X)

print(X_filled_knn)
# print(X)
import numpy as np
np.save('output_training.npy',X_filled_knn)
# print(np.load('output_training.npy'))