# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:16:25 2023

@author: jaime
"""

# %% Librerías

import numpy as np

from FuncionesLatex import (CalibracionMono,
                            CalibracionEstereo,
                            MatricesProyeccion)

# %% 3.3 Calibración

# %% Parámetros del patrón

filas = 4
columnas = 7
tamano = 0.024

# %% Abrir ficheros de imágenes

fichero1 = "3_3_Calibracion/Camara1/*"
fichero2 = "3_3_Calibracion/Camara2/*"

# %% Calibraciones individuales

MK1, dis1 = CalibracionMono(fichero1,
                            filas,
                            columnas,
                            tamano,
                            False)
MK2, dis2 = CalibracionMono(fichero2,
                            filas,
                            columnas,
                            tamano,
                            False)

print(MK1)
print(MK2)

# %% Calibración estéreo

MR, vt = CalibracionEstereo(fichero1,
                            fichero2,
                            filas,
                            columnas,
                            tamano,
                            MK1,
                            dis1,
                            MK2,
                            dis2,
                            False)

print(MR)
print(vt)

print(np.sqrt(np.sum(vt**2)))

# %% Construcción de matrices de proyección

MP1, MP2 = MatricesProyeccion(MR,
                           vt,
                           MK1,
                           MK2)

print(MP1)
print(MP2)