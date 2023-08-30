# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 19:59:32 2023

@author: jaime
"""

# %% Librerías

import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from FuncionesLatex import (AbrirCSV,
                            DerivarArray,
                            ModuloArray,
                            EncontrarMovimientos,
                            EncontrarPulsaciones)

# %% Ficheros

ruta3D = "3_7_Metricas/3DSync0.csv"
rutat = "3_7_Metricas/tSync0.csv"
rutas3D = sorted(glob.glob("3_7_Metricas/*3DSync*.csv"))
rutast = sorted(glob.glob("3_7_Metricas/*tSync*.csv"))

# %% Distancia, Velocidad, Aceleración

Pts3D = AbrirCSV(ruta3D)
tS = AbrirCSV(rutat)[:,0]

nodos = Pts3D.shape[1]//3

Dist = np.empty((Pts3D.shape[0],
                  nodos,
                  nodos,
                  3))

for fin in range(nodos):
    for ini in range(nodos):
        ptfin = Pts3D[:, fin*3:fin*3+3]
        ptini = Pts3D[:, ini*3:ini*3+3]
        Dist[:, ini, fin, :] = ptfin-ptini
        
Velo = DerivarArray(np.copy(Dist), tS)
Acce = DerivarArray(np.copy(Velo), tS)

ModD = ModuloArray(np.copy(Dist))
ModV = ModuloArray(np.copy(Velo))
ModA = ModuloArray(np.copy(Acce))

# %% Distancia, velocidad, aceleracion entre nodos

nfin = 8
nini = 4

Dnodo = ModD[:,nfin,nini]
Vnodo = ModV[:,nfin,nini]
Anodo = ModA[:,nfin,nini]

# %% Hallar métricas en movimientos

ventana = 50
umbral = 2.5

inter, std, cond = EncontrarMovimientos(Anodo,
                                        ventana,
                                        umbral)

fig, ax = plt.subplots(inter.shape[0],
                       3)

fig.tight_layout(pad=1)

tit = 'Movimiento {0}. # Pulsaciones: {1}-{2}.'
titfre = 'Movimiento {0}. Frecuencia media: {1:.3f}. CV = {2:.3f}.'
titamp = 'Movimiento {0}. Amplitud media: {1:.3f}. CV = {2:.3f}.'

ax[0,0].set_ylabel('Distancia (m)')
ax[1,0].set_ylabel('Amplitud (m)')
ax[2,0].set_ylabel('Frecuencia (m)')
ax[1,0].set_ylabel('Amplitud (m)')
ax[2,0].set_ylabel('Frecuencia (m)')

ax[2,0].set_xlabel('Tiempo (s)')
ax[2,1].set_xlabel('Tiempo (s)')
ax[2,2].set_xlabel('Tiempo (s)')

ax[0,0].set_ylim(0,0.13)
ax[0,1].set_ylim(0,0.13)
ax[0,2].set_ylim(0,0.13)

ax[1,0].set_ylim(0.05,0.17)
ax[1,1].set_ylim(0.05,0.17)
ax[1,2].set_ylim(0.05,0.17)

ax[2,0].set_ylim(2,8)
ax[2,1].set_ylim(2,8)
ax[2,2].set_ylim(2,8)

for i in range(inter.shape[0]):
    xfM, yfM, M = EncontrarPulsaciones(Dnodo,
                                       tS,
                                       inter[i,:],
                                       4,
                                       np.min(Dnodo))
    xM = xfM[M]
    yM = yfM[M]
    ax[0,i].plot(xfM, yfM)
    ax[0,i].plot(xM, yM,
                 'o')
    
    xfm, yfm, m = EncontrarPulsaciones(-Dnodo,
                                       tS,
                                       inter[i,:],
                                       4,
                                       np.min(-Dnodo))
    xm = xfm[m]
    ym = yfm[m]
    ax[0,i].plot(xm, -ym,
                 'o')
    ax[0,i].title.set_text(tit.format(i+1,len(M),len(m)))
    
    largo = np.min((len(yM), len(ym)))
    
    yM = yM[:largo]
    xM = xM[:largo]
    ym = ym[:largo]
    xm = xm[:largo]
    amp = yM - ym
    Mamp = np.mean(amp)
    CVA = np.std(amp)/Mamp
    per = np.diff(xM)
    fre = 1/per
    Mfre = np.mean(fre)
    CVF = np.std(fre)/Mfre
    ax[1,i].plot(xm, amp)
    ax[1,i].title.set_text(titamp.format(i+1, Mamp, CVA))
    ax[2,i].plot(xm[:-1], fre)
    ax[2,i].title.set_text(titfre.format(i+1, Mfre, CVF))

# %% Varios sujetos

Data = np.zeros((len(rutas3D),
                 3,
                 2,
                 3))

ventana = 50
umbral = [2.5,
          2.5,
          1.25,
          2.5,
          2.5,
          2.5,
          2.5,
          2.5]

nfin = 8
nini = 4

for j in range(len(rutas3D)):
    print('Sujeto:')
    print(j)
    Pts3D = AbrirCSV(rutas3D[j])
    tS = AbrirCSV(rutast[j])[:,0]

    nodos = Pts3D.shape[1]//3

    Dist = np.empty((Pts3D.shape[0],
                      nodos,
                      nodos,
                      3))

    for fin in range(nodos):
        for ini in range(nodos):
            ptfin = Pts3D[:, fin*3:fin*3+3]
            ptini = Pts3D[:, ini*3:ini*3+3]
            Dist[:, ini, fin, :] = ptfin-ptini
            
    Velo = DerivarArray(np.copy(Dist), tS)
    Acce = DerivarArray(np.copy(Velo), tS)

    ModD = ModuloArray(np.copy(Dist))
    ModV = ModuloArray(np.copy(Velo))
    ModA = ModuloArray(np.copy(Acce))
    
    Dnodo = ModD[:,nfin,nini]
    Vnodo = ModV[:,nfin,nini]
    Anodo = ModA[:,nfin,nini]
    
    inter, std, cond = EncontrarMovimientos(Anodo,
                                            ventana,
                                            umbral[j])
    i = 0
    while i < 3:
        print('Movimiento:')
        print(i)
        while np.abs(inter[i,1] - inter[i,0]) < 250:
            inter = np.delete(inter,i,0)
        
        xfM, yfM, M = EncontrarPulsaciones(Dnodo,
                                           tS,
                                           inter[i,:],
                                           4,
                                           np.min(Dnodo))
        xM = xfM[M]
        yM = yfM[M]
        
        xfm, yfm, m = EncontrarPulsaciones(-Dnodo,
                                           tS,
                                           inter[i,:],
                                           4,
                                           np.min(-Dnodo))
        xm = xfm[m]
        ym = yfm[m]
    
        largo = np.min((len(yM), len(ym)))
        
        yM = yM[:largo]
        xM = xM[:largo]
        ym = ym[:largo]
        xm = xm[:largo]
        amp = yM - ym
        Mamp = np.mean(amp)
        CVA = np.std(amp)/Mamp
        FanA = np.var(amp)/Mamp
        per = np.diff(xM)
        fre = 1/per
        Mfre = np.mean(fre)
        CVF = np.std(fre)/Mfre
        FanF =np.var(fre)/Mfre
    
        Data[j,i,0,0] = Mamp
        Data[j,i,0,1] = CVA
        Data[j,i,0,2] = FanA
        
        Data[j,i,1,0] = Mfre
        Data[j,i,1,1] = CVF
        Data[j,i,1,2] = FanF
        
        i = i+1
        
# %% Plot varios sujetos

fig, ax = plt.subplots(2, 2)
fig.tight_layout(pad=1)

labels = ['Mov 1','Mov 2','Mov 3']
x = [0,1,2]
ax[0,0].set_xticks(x, labels)
ax[0,0].title.set_text('Amplitud media')
ax[0,0].set_ylabel('Amplitud (cm)')
ax[0,1].set_xticks(x, labels)
ax[0,1].title.set_text('Frecuencia media')
ax[0,1].set_ylabel('Frecuencia (Hz)')
ax[1,0].set_xticks(x, labels)
ax[1,0].title.set_text('Factor Fano amplitud')
ax[1,0].set_ylabel('Factor Fano')
ax[1,1].set_xticks(x, labels)
ax[1,1].title.set_text('Factor Fano frecuencia')
ax[1,1].set_ylabel('Factor Fano')


ax[0,0].set_xlim(right=3,
                 left=-0.5)
ax[0,1].set_xlim(right=3,
                 left=-0.5)
ax[1,0].set_xlim(right=3,
                 left=-0.5)
ax[1,1].set_xlim(right=3,
                 left=-0.5)

for j in range(len(rutas3D)):
    ax[0,0].plot(Data[j,:,0,0]*100,
                 '-o',
                 label = 'Sujeto {0}'.format(j))
    ax[0,0].legend()
    ax[0,1].plot(Data[j,:,1,0],
                 '-o',
                 label = 'Sujeto {0}'.format(j))
    ax[0,1].legend()
    ax[1,0].plot(Data[j,:,0,2],
                 '-o',
                 label = 'Sujeto {0}'.format(j))
    ax[1,0].legend()
    ax[1,1].plot(Data[j,:,1,2],
                 '-o',
                 label = 'Sujeto {0}'.format(j))
    ax[1,1].legend()
    
