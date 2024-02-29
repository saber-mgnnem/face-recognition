# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:43:19 2023

@author: R I B
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

np.random.seed(1)
x, y = make_classification(n_samples=100,n_features=2, n_redundant=0, n_informative=1,
n_clusters_per_class=1)

# Visualisation des données
plt.figure(num=None, figsize=(8, 6))
plt.scatter(x[:,0], x[:, 1], marker = 'o', c=y, edgecolors='k')
plt.xlabel('X0')
plt.ylabel('X1')
x.shape
model = SGDClassifier(max_iter=1000, eta0=0.001, loss='log_loss')
model.fit(x, y)
print('score:', model.score(x, y))