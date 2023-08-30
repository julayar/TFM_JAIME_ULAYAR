# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 08:06:11 2023

@author: julayar.1
"""

# %% Librerías

from FuncionesLatex import (AbrirCSV,
                            ModuloArray,
                            DerivarArray,
                            EncontrarMovimientos,
                            EncontrarPulsaciones)

import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg

# %% Ficheros

rutaDer = '3_8_UnaCamara/2DDer0.csv'
rutaIzq = '3_8_UnaCamara/2DIzq0.csv'
ruta3D = '3_8_UnaCamara/3DFTSync.csv'

rutatD = '3_8_UnaCamara/tDer0.csv'
rutatI = '3_8_UnaCamara/tIzq0.csv'
rutat = '3_8_UnaCamara/tFTSync.csv'

PtsDS = AbrirCSV(rutaDer)
PtsIS = AbrirCSV(rutaIzq)
Pts3D = AbrirCSV(ruta3D)

tDS = AbrirCSV(rutatD)
tIS = AbrirCSV(rutatI)
tS = AbrirCSV(rutat)[:,0]

# %% Módulo de distancias

nodos = 21
nini = 8
nfin = 4

DistDer = np.empty((PtsDS.shape[0],
                 nodos,
                 nodos,
                 2))

DistIzq = np.empty((PtsIS.shape[0],
                 nodos,
                 nodos,
                 2))

Dist = np.empty((Pts3D.shape[0],
                 nodos,
                 nodos,
                 3))

for fin in range(nodos):
    for ini in range(nodos):
        ptfin = PtsDS[:, fin*3:fin*3+2]
        ptini = PtsDS[:, ini*3:ini*3+2]
        DistDer[:, ini, fin, 0:2] = ptfin-ptini
        ptfin = PtsIS[:, fin*3:fin*3+2]
        ptini = PtsIS[:, ini*3:ini*3+2]
        DistIzq[:, ini, fin, 0:2] = ptfin-ptini
        ptfin = Pts3D[:, fin*3:fin*3+3]
        ptini = Pts3D[:, ini*3:ini*3+3]
        Dist[:, ini, fin, :] = ptfin-ptini
        
MDD = ModuloArray(DistDer)
MDI = ModuloArray(DistIzq)
MD3 = ModuloArray(Dist)

DistnodoD = MDD[:,nfin,nini]
DistnodoI = MDI[:,nfin,nini]
Distnodo = MD3[:,nfin,nini]

Velo = DerivarArray(np.copy(Dist), tS)
Acce = DerivarArray(np.copy(Velo), tS)

Accenodo = Acce[:,nfin,nini,1]
ventana = 50
umbral = 2.5

interCAM, stdCAM, condCAM = EncontrarMovimientos(Accenodo,
                                                 ventana,
                                                 umbral)

# %% Representación

fig, ax = plt.subplots(3,1, sharex = True)
fig.tight_layout(pad=1)

ax[0].plot(tDS, DistnodoD)
ax[1].plot(tS, Distnodo)
ax[2].plot(tIS, DistnodoI)

# %% Extraer pulsaciones

for i in range(interCAM.shape[0]):
    xfCAM, yfCAM, maxCAM = EncontrarPulsaciones(DistnodoD,
                                                tDS,
                                                interCAM[i,:],
                                                4,
                                                np.min(DistnodoD))
    xMAX = xfCAM[maxCAM]
    yMAX = yfCAM[maxCAM]
    xfCAM, yfCAM, minCAM = EncontrarPulsaciones(-DistnodoD,
                                                tDS,
                                                interCAM[i,:],
                                                4,
                                                np.min(-DistnodoD))
    xMIN = xfCAM[minCAM]
    yMIN = yfCAM[minCAM]
    
    largo = np.min((len(yMAX), len(yMIN)))
    yMAX = yMAX[:largo]
    xMAX = xMAX[:largo]
    yMIN = yMIN[:largo]
    xMIN = xMIN[:largo]
    ampD = yMAX - yMIN

for i in range(interCAM.shape[0]):
    xfCAM, yfCAM, maxCAM = EncontrarPulsaciones(Distnodo,
                                                tS,
                                                interCAM[i,:],
                                                4,
                                                np.min(Distnodo))
    xMAX = xfCAM[maxCAM]
    yMAX = yfCAM[maxCAM]
    xfCAM, yfCAM, minCAM = EncontrarPulsaciones(-Distnodo,
                                                tS,
                                                interCAM[i,:],
                                                4,
                                                np.min(-Distnodo))
    xMIN = xfCAM[minCAM]
    yMIN = yfCAM[minCAM]
    
    largo = np.min((len(yMAX), len(yMIN)))
    yMAX = yMAX[:largo]
    xMAX = xMAX[:largo]
    yMIN = yMIN[:largo]
    xMIN = xMIN[:largo]
    amp = yMAX - yMIN
    
for i in range(interCAM.shape[0]):
    xfCAM, yfCAM, maxCAM = EncontrarPulsaciones(DistnodoI,
                                                tIS,
                                                interCAM[i,:],
                                                4,
                                                np.min(DistnodoI))
    xMAX = xfCAM[maxCAM]
    yMAX = yfCAM[maxCAM]
    xfCAM, yfCAM, minCAM = EncontrarPulsaciones(-DistnodoI,
                                                tIS,
                                                interCAM[i,:],
                                                4,
                                                np.min(-DistnodoI))
    xMIN = xfCAM[minCAM]
    yMIN = yfCAM[minCAM]
    
    largo = np.min((len(yMAX), len(yMIN)))
    yMAX = yMAX[:largo]
    xMAX = xMAX[:largo]
    yMIN = yMIN[:largo]
    xMIN = xMIN[:largo]
    ampI = yMAX - yMIN


# %% Correlación y Bland-Altman

NampD = ampD/np.sqrt(np.sum(ampD**2))
Namp = amp/np.sqrt(np.sum(amp**2))
NampI = ampI/np.sqrt(np.sum(ampI**2))

pg.corr(Namp, NampD, method="bicor")
pg.corr(Namp, NampI, method="bicor")
pg.corr(NampD, NampI, method="bicor")

pg.plot_blandaltman(amp, ampD)
pg.plot_blandaltman(amp, ampI)
pg.plot_blandaltman(ampI, ampD)
