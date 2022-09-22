#!/usr/bin/env python
# -*- coding: latin-1 -*-

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm 

import time
start_time = time.time()

#obteniendo los datos aleatorios
x, y = make_classification(n_samples=5000, n_features=10, 
                           n_classes=3, 
                           n_clusters_per_class=1)
#dividir entre train y test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)

#entrenando
clf = svm.SVC() 
clf.fit(xtrain,ytrain)
score = clf.score(xtrain, ytrain) 

#obteniendo la precisión
print(f'Precisión: {score:.6f}')

tiempo=(time.time() - start_time)

print(f'Tiempo de proceso: {tiempo:.6f}')

