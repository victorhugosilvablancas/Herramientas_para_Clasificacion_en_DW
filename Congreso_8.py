#!/usr/bin/env python
# -*- coding: latin-1 -*-

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay 
import pandas as pd

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

#obteniendo el reporte
ypred = clf.predict(xtest)

reporte = classification_report(ytest, ypred, output_dict=True)
df = pd.DataFrame(reporte).transpose()
df.to_csv('reporte8.csv')

#matriz de confusión
cm = confusion_matrix(ytest, ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.savefig('matriz8.png')

tiempo=(time.time() - start_time)
print(f'Tiempo de proceso: {tiempo:.6f}')