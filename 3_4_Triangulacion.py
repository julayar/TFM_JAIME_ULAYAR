# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:24:47 2023

@author: jaime
"""

# %% Librerias

import numpy as np
import matplotlib.pyplot as plt

from FuncionesLatex import (Triangulacion)

# %%%% 3.4 Triangulación

# %% Ficheros

# Descomentar un par de los siguientes

## Golpeteo de dedos. "FT".
#coordDer = np.genfromtxt('3_4_Triangulacion/cooDerFT.csv',
#                         delimiter=',')
#coordIzq = np.genfromtxt('3_4_Triangulacion/cooIzqFT.csv',
#                         delimiter=',')

## Apertura de manos. "OC".
#coordDer = np.genfromtxt('3_4_Triangulacion/cooDerOC.csv',
#                         delimiter=',')
#coordIzq = np.genfromtxt('3_4_Triangulacion/cooIzqOC.csv',
#                         delimiter=',')

## Rotación de manos. "PS"
#coordDer = np.genfromtxt('3_4_Triangulacion/cooDerPS.csv',
#                         delimiter=',')
#coordIzq = np.genfromtxt('3_4_Triangulacion/cooIzqPS.csv',
#                         delimiter=',')

coordDer = np.genfromtxt('sujetos/cooDer7.csv',
                         delimiter=',')
coordIzq = np.genfromtxt('sujetos/cooIzq7.csv',
                         delimiter=',')

size = coordDer.shape
rows = size[0]
cols = size[1]

Pts3D = np.empty((rows,0))     
res = (480, 640)

MP1 = np.genfromtxt('3_4_Triangulacion/MP1.csv',
                    delimiter=',')
MP2 = np.genfromtxt('3_4_Triangulacion/MP2.csv',
                    delimiter=',')

# %% Triangulación
      
for i in range(0,cols, 3):
    print(int(i/3)+1,"/", int(cols/3))
    PtsDer = np.multiply(coordDer[:,[i,i+1]].astype('float64'),
                            [res])
    PtsIzq = np.multiply(coordIzq[:,[i,i+1]].astype('float64'),
                            [res])
    Coord3D = Triangulacion(PtsDer,
                            PtsIzq,
                            MP1, MP2)
    Pts3D = np.concatenate((Pts3D, Coord3D.reshape(-1,3)),
                           axis = 1)
    
# %% Definición del esqueleto

line_list = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [5, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [9, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [13, 17],
             [17, 18],
             [18, 19],
             [19, 20],
             [17, 0]]

# %% Representaciones 3D

ax = plt.figure().add_subplot(projection='3d')

for i in range(len(Pts3D)):
    ax.set_xlim(-0.25,0.25)
    ax.set_ylim(-0.25,0.25)
    ax.set_zlim(0.25,0.75)
    for line in line_list:
        ax.plot([Pts3D[i, line[0]*3],
                 Pts3D[i, line[1]*3]],
                [Pts3D[i, line[0]*3+1],
                 Pts3D[i, line[1]*3+1]],
                [Pts3D[i, line[0]*3+2],
                 Pts3D[i, line[1]*3+2]],
                'k-')
    ax.plot(Pts3D[i,0::3],
            Pts3D[i,1::3],
            Pts3D[i,2::3],
            'r.',
            markersize = 5)
    plt.pause(0.01)
    ax.cla()
 
# %% Verificación distancia conocida

nodoini = Pts3D[:,6*3:6*3+3]
nodofin = Pts3D[:,5*3:5*3+3]
dist = nodofin - nodoini
falange = np.sqrt(np.sum(dist**2,1))
mean = np.mean(falange)
sd = np.std(falange)

print(mean)
print(sd)