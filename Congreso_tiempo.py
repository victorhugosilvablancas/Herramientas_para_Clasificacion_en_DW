#!/usr/bin/env python
# -*- coding: latin-1 -*-

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("resultados.csv", parse_dates=True, index_col="num")
df = df.drop(columns = 'algoritmo')

#df['tiempo'].plot()
df.plot(subplots=True, figsize=(10,12))
#plt.show()
plt.savefig('resultados.png')