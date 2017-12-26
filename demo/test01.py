import numpy as np
print(np.random.randint(2))
list = [(1,5),(2,6),(3,7),(4,8),(5,9),(6,4)]
y = [i for i, j in list if j > 6]
print (y)

from sklearn.datasets import load_iris
#X, y = np.loadtxt("X_classification.txt"), np.loadtxt("y_classification.txt")
dataset = load_iris()
X = dataset.data
y = dataset.target
print(dataset.DESCR)
n_samples, n_features = X.shape




