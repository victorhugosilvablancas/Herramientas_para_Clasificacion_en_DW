#!/usr/bin/env python
# -*- coding: latin-1 -*-

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

import time
start_time = time.time()

#obteniendo los datos aleatorios
x, y = make_classification(n_samples=5000, n_features=10, 
                           n_classes=3, 
                           n_clusters_per_class=1)
#dividir entre train y test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)

votin1 = LogisticRegression(random_state=1)
votin2 = RandomForestClassifier(n_estimators=50, random_state=1)
votin3 = GaussianNB()

votin = VotingClassifier(
    estimators=[('lr', votin1), ('rf', votin2), ('gnb', votin3)],
    voting='hard')

for clf, label in zip([votin1, votin2, votin3, votin], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, xtrain, ytrain, scoring='accuracy', cv=5)
    print("Precisión: %0.6f (+/- %0.6f) [%s]" % (scores.mean(), scores.std(), label))

tiempo=(time.time() - start_time)

print(f'Tiempo de proceso: {tiempo:.6f}')

