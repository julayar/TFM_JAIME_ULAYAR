# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 19:22:37 2023

@author: jaime
"""

# %% Librerías

import numpy as np
import matplotlib.pyplot as plt

from FuncionesLatex import (AbrirCSV,
                            ConversionSegundos,
                            DerivarArray,
                            ModuloArray,
                            EncontrarMovimientos,
                            EncontrarPulsaciones)

# %% Ficheros

ruta3DS = '3_6_Acelerometro/3DACCSync.csv'
rutatS = '3_6_Acelerometro/tACCSync.csv'
Pts3DS = AbrirCSV(ruta3DS)
tS = AbrirCSV(rutatS)[:,0]

rutaACC = '3_6_Acelerometro/ACCacc0.csv'
ACC = AbrirCSV(rutaACC)

tACC = ConversionSegundos(ACC, 4)
yACC = ACC[:,3]
ModACC = np.sqrt(np.sum((ACC[:,0:3]**2),1))*9.8

# %% Distancia, velocidad y aceleración

rows = Pts3DS.shape[0]
cols = Pts3DS.shape[1]

nodos = int(cols/3)

DistS = np.empty((Pts3DS.shape[0],
                 nodos,
                 nodos,
                 3))

for fin in range(nodos):
    for ini in range(nodos):
        ptfin = Pts3DS[:, fin*3:fin*3+3]
        ptini = Pts3DS[:, ini*3:ini*3+3]
        DistS[:, ini, fin, :] = ptfin-ptini
      
VeloS = DerivarArray(np.copy(DistS), tS)
AcceS = DerivarArray(np.copy(VeloS), tS)

ModDS = ModuloArray(np.copy(DistS))
ModVS = ModuloArray(np.copy(VeloS))
ModAS = ModuloArray(np.copy(AcceS))

# %% Comparación aceleraciones

nfin = 8
nini = 4
eje = 1

Distnodo = DistS[:,nfin,nini,eje]
Velonodo = VeloS[:,nfin,nini,eje]
Accenodo = AcceS[:,nfin,nini,eje]

fig, ax = plt.subplots(4,
                      1,
                      sharex = True)
lineD, = ax[0].plot(tS, Distnodo)
ax[0].set_ylabel('Distancia (m)')
ax[0].title.set_text('Distancia (MediaPipe)')
lineV, = ax[1].plot(tS, Velonodo)
ax[1].set_ylabel('Velocidad (m/s)')
ax[1].title.set_text('Velocidad (MediaPipe)')
lineA, = ax[2].plot(tS, Accenodo)
ax[2].set_ylabel('Aceleración (m/s^2)')
ax[2].title.set_text('Aceleración (MediaPipe)')
LineAcc, = ax[3].plot(tACC,ModACC)
ax[3].set_ylabel('Aceleración (m/s^2)')
ax[3].title.set_text('Aceleración (Acelerómetro)')
ax[3].set_xlabel('Tiempo (s)')

# %% Encontrar intervalos de movimiento

ventCAM = 50
ventACC = 200
umbral = 2.5

interCAM, stdCAM, condCAM = EncontrarMovimientos(Accenodo,
                                                 ventCAM,
                                                 umbral)
interACC, stdACC, condACC = EncontrarMovimientos(ModACC,
                                                 ventACC,
                                                 umbral)

fig, ax = plt.subplots(2,2, sharex = True)
ax[0,0].plot(tS, Accenodo)
ax[1,0].plot(tS, stdCAM)
ax[1,0].plot(tS, condCAM*np.max(stdCAM))

ax[0,1].plot(tACC, ModACC)
ax[1,1].plot(tACC, stdACC)
ax[1,1].plot(tACC, condACC*np.max(stdACC))
ax[1,0].set_ylabel('Desviación estándar')
ax[0,0].set_ylabel('Aceleración (m/s^2)')
ax[1,0].set_xlabel('Tiempo (s)')
ax[1,1].set_xlabel('Tiempo (s)')
ax[0,0].title.set_text('Intervalos detectados (MediaPipe)')
ax[0,1].title.set_text('Intervalos detectados (Acelerómetro)')

for i in range(interCAM.shape[0]):
    aux1 = tS[interCAM[i,0]]
    aux2 = tS[interCAM[i,1]]
    aux3 = tS[ventCAM//2]
    ax[0,0].axvline(x = aux1,
                    color = 'g')
    ax[0,0].axvline(x=aux2,
                    color = 'r')
    
    aux1 = tACC[interACC[i,0]]
    aux2 = tACC[interACC[i,1]]
    aux3 = tACC[ventACC//2]
    ax[0,1].axvline(x=aux1,
                    color = 'g')
    ax[0,1].axvline(x=aux2,
                    color = 'r')
    
# %% Encontrar máximos

fig, ax = plt.subplots(interCAM.shape[0], 2)
fig.tight_layout(pad=1)

titCAM = 'Movimiento {0} (MediaPipe). # Pulsaciones: {1}'
titACC = 'Movimiento {0} (Acelerómetro). # Pulsaciones: {1}'
ax[2,0].set_xlabel('Tiempo (s)')
ax[2,1].set_xlabel('Tiempo (s)')
ax[0,0].set_ylabel('Aceleración (m/s^2)')
ax[1,0].set_ylabel('Aceleración (m/s^2)')
ax[2,0].set_ylabel('Aceleración (m/s^2)')

for i in range(interCAM.shape[0]):
    xfCAM, yfCAM, maxCAM = EncontrarPulsaciones(Accenodo*(-1),
                                                tS,
                                                interCAM[i,:],
                                                4,
                                                5)
    ax[i,0].plot(xfCAM, yfCAM)
    ax[i,0].plot(xfCAM[maxCAM], yfCAM[maxCAM],
                 'o')
    ax[i,0].title.set_text(titCAM.format(i+1, len(maxCAM)))
    xfACC, yfACC, maxACC = EncontrarPulsaciones(ModACC,
                                                tACC,
                                                interACC[i,:],
                                                20,
                                                20)


    ax[i,1].plot(xfACC, yfACC)
    ax[i,1].plot(xfACC[maxACC], yfACC[maxACC],
                 'o')
    ax[i,1].title.set_text(titACC.format(i+1, len(maxACC)))