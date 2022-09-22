#!/usr/bin/env python
# -*- coding: latin-1 -*-

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import time
start_time = time.time()

#obteniendo los datos aleatorios
x, y = make_classification(n_samples=5000, n_features=10, 
                           n_classes=3, 
                           n_clusters_per_class=1)
#dividir entre train y test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)

#entrenando
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(xtrain, ytrain)

#obteniendo la precisión
score = clf.score(xtrain, ytrain)
print(f'Precisión: {score:.6f}')

tiempo=(time.time() - start_time)

print(f'Tiempo de proceso: {tiempo:.6f}')

