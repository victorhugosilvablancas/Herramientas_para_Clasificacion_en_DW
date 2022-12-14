#!/usr/bin/env python
# -*- coding: latin-1 -*-

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.datasets import fetch_20newsgroups_vectorized
import pandas as pd

import time
start_time = time.time()

#obteniendo los datos
n_samples=5000
x, y = fetch_20newsgroups_vectorized(subset="all", return_X_y=True)
x = x[:n_samples]
y = y[:n_samples]

#dividir entre train y test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)

#entrenando
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
clf.fit(xtrain, ytrain) 

#obteniendo el reporte
ypred = clf.predict(xtest) 

reporte = classification_report(ytest, ypred, output_dict=True)
df = pd.DataFrame(reporte).transpose()
df.to_csv('reporte3.csv')

#matriz de confusión
cm = confusion_matrix(ytest, ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.savefig('matriz3.png')

tiempo=(time.time() - start_time)
print(f'Tiempo de proceso: {tiempo:.6f}')

