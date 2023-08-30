# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 17:53:27 2023

@author: jaime
"""
# %% Librerías

import mediapipe as mp
import numpy as np
import cv2 as cv

from FuncionesLatex import (AbrirVideo,
                            PredecirFrame,
                            ExtraerCoordenadas,
                            TratarNA)


# %% 3.1 Grabaciones

# Descomentar un par de los siguientes

## Golpeteo de dedos. "FT".
#videoDer = '3_12_GrabacionesMediaPipe/FTDer0.avi'
#videoIzq = '3_12_GrabacionesMediaPipe/FTIzq0.avi'

## Apertura de manos. "OC".
#videoDer = '3_12_GrabacionesMediaPipe/OCDer0.avi'
#videoIzq = '3_12_GrabacionesMediaPipe/OCIzq0.avi'

## Rotación de manos. "PS"
#videoDer = '3_12_GrabacionesMediaPipe/PSDer0.avi'
#videoIzq = '3_12_GrabacionesMediaPipe/PSIzq0.avi'

videoDer = 'sujetos/METDer0.avi'
videoIzq = 'sujetos/METIzq0.avi'

camDer, NframesDer, fpsDer = AbrirVideo(videoDer)
camIzq, NframesIzq, fpsIzq = AbrirVideo(videoIzq)

if NframesDer != NframesIzq:
    print("Error: videos de distinto tamaño.")
    print("Nº frames video derecho: ",
          NframesDer,
          "\n","Nº frames video izquierdo:" ,
          NframesIzq)
    
# %% 3.2 MediaPipe


# Construir modelos
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

hand1 = mp_hands.Hands(min_detection_confidence = 0.6,
                       min_tracking_confidence = 0.5,
                       max_num_hands = 1)
hand2 = mp_hands.Hands(min_detection_confidence = 0.6,
                       min_tracking_confidence = 0.5,
                       max_num_hands = 1)

# Procesar vídeos
coordDer = np.empty((NframesDer,
                     21*3))*np.nan
coordIzq = np.empty((NframesIzq,
                     21*3))*np.nan

size = coordDer.shape
rows = size[0]
cols = size[1]

for frame in range(NframesDer):
    resDer, imDer = PredecirFrame(camDer,
                                  hand1)
    resIzq, imIzq = PredecirFrame(camIzq,
                                  hand2)
    
    aux = resDer.multi_hand_landmarks
    if aux:
        mp_drawing.draw_landmarks(imDer,
                                  aux[0],
                                  mp_hands.HAND_CONNECTIONS)
        coordDer[frame,:] = ExtraerCoordenadas(resDer)
    
    aux = resIzq.multi_hand_landmarks
    if aux:
        mp_drawing.draw_landmarks(imIzq,
                                  aux[0],
                                  mp_hands.HAND_CONNECTIONS)
        coordIzq[frame, :] = ExtraerCoordenadas(resIzq)

            
    cv.imshow(videoDer, imDer)
    cv.imshow(videoIzq, imIzq)
    
    cv.waitKey(1)
    print(frame+1,"/", NframesDer)
        
cv.destroyWindow(videoDer)
cv.destroyWindow(videoIzq)

coordDer = TratarNA(coordDer)
coordIzq = TratarNA(coordIzq)