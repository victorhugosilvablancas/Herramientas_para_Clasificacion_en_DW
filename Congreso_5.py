#!/usr/bin/env python
# -*- coding: latin-1 -*-

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay 

import time
start_time = time.time()

#obteniendo los datos aleatorios
x, y = make_classification(n_samples=5000, n_features=10, 
                           n_classes=3, 
                           n_clusters_per_class=1)
#dividir entre train y test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)

#entrenando
rc = GaussianProcessClassifier()
rc.fit(xtrain, ytrain)

#obteniendo el reporte
ypred = rc.predict(xtest) 
 
print(classification_report(ytest, ypred))

#matriz de confusión
cm = confusion_matrix(ytest, ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=rc.classes_)
disp.plot()
plt.savefig('matriz5.png')

tiempo=(time.time() - start_time)
print(f'Tiempo de proceso: {tiempo:.6f}')
