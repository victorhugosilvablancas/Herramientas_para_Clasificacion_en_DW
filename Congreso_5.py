#!/usr/bin/env python
# -*- coding: latin-1 -*-

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.datasets import load_iris
import pandas as pd

import time
start_time = time.time()

#obteniendo los datos
x, y = load_iris(return_X_y=True)

#dividir entre train y test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)

#entrenando
kernel = 1.0 * RBF(1.0)
rc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(x, y)
rc.fit(xtrain, ytrain)

#obteniendo el reporte
ypred = rc.predict(xtest)

reporte = classification_report(ytest, ypred, output_dict=True)
df = pd.DataFrame(reporte).transpose()
df.to_csv('reporte5.csv')

#matriz de confusión
cm = confusion_matrix(ytest, ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=rc.classes_)
disp.plot()
plt.savefig('matriz5.png')

tiempo=(time.time() - start_time)
print(f'Tiempo de proceso: {tiempo:.6f}')
