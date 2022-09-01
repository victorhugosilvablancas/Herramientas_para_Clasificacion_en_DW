#!/usr/bin/env python
# -*- coding: latin-1 -*-

import numpy as np #arreglos
import pandas as pd #datasets 
import matplotlib.pyplot as plt #graficar
import seaborn as sns #graficar con estadísticas

conto_data = pd.read_csv('contopolimov.csv', index_col=0, parse_dates=True)
print(conto_data.head())
#conto_data['cargo'].plot()
#conto_data['abono'].plot()
#conto_data.plot.scatter(x='cargo', y='abono', alpha=0.5)
conto_data.plot.box()

#ejes = conto_data.plot.area(figsize=(12,4), subplots=True)

figura,ejes = plt.subplots(figsize=(12,4))
conto_data.plot.area(ax=ejes) 
ejes.set_ylabel("movimientos")
plt.show() 
figura.savefig("contopolimov.png")


conto_data['saldo']=conto_data['cargo'] + conto_data['abono']
print(conto_data.head())
