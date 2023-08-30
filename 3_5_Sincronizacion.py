# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 19:03:17 2023

@author: jaime
"""

# %% Librerías

import numpy as np
import matplotlib.pyplot as plt

from FuncionesLatex import(AbrirCSV,
                           ConversionSegundos,
                           IntervalosSenal,
                           Sincronizar)

# %%%% 3.5 Sincronización

ruta3D = '3_7_Metricas/cooDer0.csv' 
ruta3D = '3_7_Metricas/cooIzq0.csv' 

# %%
Pts3D = AbrirCSV(ruta3D)
rutaROI = '3_7_Metricas/METROI0.csv' 
ROI = AbrirCSV(rutaROI)

# %% Ficheros

ruta3D = '3_5_Sincronizacion/3DFT.csv'
Pts3D = AbrirCSV(ruta3D)

rutaROI = 'sujetos/METROI7.csv'
ROI = AbrirCSV(rutaROI)

# %% Análisis de frecuencia de muestreo

tO = ConversionSegundos(ROI, 1)
yO = ROI[:,0]

fps = 1/(np.diff(tO))
fig = plt.figure()
plt.scatter(tO[:-1], fps)
plt.xlabel('t (s)')
plt.ylabel('Frecuencia de muestreo (fps)')

# %% Sincronización por intervalos
I = IntervalosSenal(yO)

ini = I[0,0]
fin = I[-1,1]

Pts3DR = Pts3D[ini:fin,:]
tR = tO[ini:fin]
yR = yO[ini:fin]

N = I.shape[0]
T = 0.5
Pts3DS = np.empty((15*N+1,Pts3D.shape[1]))

for i in range(Pts3D.shape[1]):
    tS, Pts3DS[:,i] = Sincronizar(tR, Pts3DR[:,i], T, I)

# %% Comparación original vs sincronizado
fig, ax = plt.subplots(2,1, sharex = True)
ax[0].plot(tO, Pts3D[:,25])
ax[1].plot(tS, Pts3DS[:,25])
ax[0].set_xlabel('Tiempo (s)')
ax[0].set_ylabel('Posición (m)')
ax[1].set_xlabel('Tiempo (s)')
ax[1].set_ylabel('Posición (m)')
