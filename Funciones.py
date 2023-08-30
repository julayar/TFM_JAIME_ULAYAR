# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 12:09:02 2023

@author: jaime
"""

import glob
import cv2 as cv
import numpy as np
from scipy import linalg
import pandas as pd
import csv
from scipy.signal import find_peaks
import mediapipe as mp

def CalibracionMono(fichero, filas, columnas, tamano, check):
    """
    Toma todas las fotos dentro del fichero seleccionado
    y realiza un calibrado de la cámara con la que se han
    tomado especificando las características del
    patrón de ajedrez como variables de entrada.                                                                              
    """
    # Crear listado de fotos
    rutas = sorted(glob.glob(fichero))
    imagenes = []
    
    for imagen in rutas[0:len(rutas)-1]:
        im = cv.imread(imagen,1)
        imagenes.append(im)
    
    # Definir criterios de terminación del
    # algoritmo iterativo calibrateCamera
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                100,
                0.001)
    
    # Describir el patrón en coordenadas 
    # coplanares medidas en metros
    patron = np.zeros((filas*columnas, 3), np.float32)
    patron[:,:2] = np.mgrid[0:filas, 0:columnas].T.reshape(-1,2)
    patron = tamano * patron
    
    ancho = imagenes[0].shape[1]
    alto = imagenes[0].shape[0]
    
    pts_img = []
    pts_cal = []
    
    # Recorrer cada foto y detectar las 
    # esquinas interiores del patrón
    for im in imagenes:
        BW = cv.cvtColor(im,
                         cv.COLOR_BGR2GRAY)
        ret, esquinas = cv.findChessboardCorners(BW,
                                                 (filas,columnas),
                                                 None)
        # Si se encuentra el patrón en la 
        # foto, ret == True
        if ret == True:
            # Tamaño de convolución para 
            # detectar las esquinas. 
            # Si es muy grande produce errores.
            convolucion = (5, 5)
            
            # Refinar la detección de 
            # esquinas y mostrarlas 
            # superpuestas en la foto
            esquinas = cv.cornerSubPix(BW,
                                       esquinas,
                                       convolucion,
                                       (-1, -1),
                                       criteria)
            cv.drawChessboardCorners(im,
                                     (filas, columnas),
                                     esquinas,
                                     ret)
            if check:
                cv.imshow('Patron detectado',
                          im)
                cv.waitKey()
            
            # Guardar las coordenadas obtenidas
            pts_cal.append(patron)
            pts_img.append(esquinas)
    
    # Calibración propiamente dicha. Estima 
    # la mejor matriz de calibración asociada
    # a las fotos y la distorsión de la cámara. 
    # También devuelve las matrices de rotación
    # y los vectores de traslación asociadas a 
    # cada foto.
    ret, MK, dis, MR, vt = cv.calibrateCamera(pts_cal,
                                              pts_img,
                                              (ancho, alto),
                                              None,
                                              None)
    if check:
        cv.destroyWindow('Patron detectado')
    return MK, dis
            

def CalibracionEstereo(fichero1,
                       fichero2,
                       filas,
                       columnas,
                       tamano,
                       MK1,
                       dis1,
                       MK2,
                       dis2,
                       check):
    """
    Toma dos conjuntos de imágenes, correspondientes 
    a dos cámaras diferentes, de los ficheros 
    especificados. Realiza con ellos una calibración
    estéreo, tomando como variables de entrada las 
    características del patrón usado.
    """
    
    rutas1 = sorted(glob.glob(fichero1))
    rutas2 = sorted(glob.glob(fichero2))
    
    imagenes1 = []
    imagenes2 = []
    
    for im1, im2 in zip(rutas1[0:len(rutas1)-1],
                        rutas2[0:len(rutas2)-1]):
        aux = cv.imread(im1,1)
        imagenes1.append(aux)
        
        aux = cv.imread(im2,1)
        imagenes2.append(aux)
        
    # Definir criterios de terminación del 
    # algoritmo iterativo calibrateCamera
    crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                100,
                0.0001)
    
    # Describir el patrón en coordenadas 
    # coplanares medidas en metros
    patron = np.zeros((filas*columnas, 3), np.float32)
    aux = np.mgrid[0:filas, 0:columnas]
    patron[:,:2] = aux.T.reshape(-1,2)
    patron = tamano * patron
    
    ancho = imagenes1[0].shape[1]
    alto = imagenes1[0].shape[0]
    
    pts_img1 = []
    pts_img2 = []
    
    pts_cal = []
    
    # Recorrer cada pareja de fotos y detectar 
    # las esquinas interiores del patrón
    for im1, im2 in zip(imagenes1,
                        imagenes2):
        BW1 = cv.cvtColor(im1,
                          cv.COLOR_BGR2GRAY)
        BW2 = cv.cvtColor(im2,
                          cv.COLOR_BGR2GRAY)
        ret1, esq1 = cv.findChessboardCorners(BW1,
                                              (filas, columnas),
                                              None)
        ret2, esq2 = cv.findChessboardCorners(BW2,
                                              (filas, columnas),
                                              None)
        # Si se encuentra el patrón en 
        # ambas fotos, ret1 == ret2 == True
        if ret1 == True and ret2 == True:
            
            # Tamaño de convolución para 
            # detectar las esquinas. Si 
            # es muy grande produce errores.
            convolucion = (5, 5)
            
            # Refinar la detección de esquinas 
            # y mostrarlas superpuestas en 
            # las fotos
            esquinas1 = cv.cornerSubPix(BW1,
                                        esq1,
                                        convolucion,
                                        (-1, -1),
                                        crit)
            esquinas2 = cv.cornerSubPix(BW2,
                                        esq2,
                                        convolucion,
                                        (-1, -1),
                                        crit)
            cv.drawChessboardCorners(im1,
                                     (filas, columnas),
                                     esq1,
                                     ret1)
            cv.drawChessboardCorners(im2,
                                     (filas, columnas),
                                     esq2,
                                     ret2)
            if check:
                cv.imshow('Patron detectado 1',
                          im1)
                cv.imshow('Patron detectado 2',
                          im2)
                cv.waitKey()
            
            # Guardar las coordenadas obtenidas
            pts_cal.append(patron)
            pts_img1.append(esquinas1)
            pts_img2.append(esquinas2)
    
    # Calibración estéreo propiamente dicha. 
    # Estima la mejor matriz de rotación R
    # y el mejor vector de traslación T 
    # entre las dos cámaras que mejor se 
    # adapten a los datos de calibración. 
    # Proporciona también las matrices de
    # proyección. 
    flags = cv.CALIB_FIX_INTRINSIC
    ret,MP1,dis1,MP2,dis2,MR,vt,E,F=cv.stereoCalibrate(pts_cal,
                                                       pts_img1,
                                                       pts_img2,
                                                       MK1,
                                                       dis1,
                                                       MK2,
                                                       dis2,
                                                       (ancho, alto),
                                                       criteria = crit,
                                                       flags = flags)
    if check:
        cv.destroyWindow('Patron detectado 1')
        cv.destroyWindow('Patron detectado 2')
    return MR, vt
    
def MatricesProyeccion(MR, vt, MK1, MK2):
    """
    Impone la primera cámara en el origen 
    de coordenadas globales y la segunda 
    en relación a la primera y devuelve 
    las matrices de proyección correspondientes.
    """
    MP1 = MK1 @ np.concatenate([np.eye(3),
                                [[0], [0], [0]]],
                               axis = -1)
    MP2 = MK2 @ np.concatenate([MR, vt],
                               axis = -1)
    
    return MP1, MP2

def Triangulacion(Pts1, Pts2, MP1, MP2):
    """
    Para cada par de puntos dados por dos 
    listas de coordenadas, calcula el punto 
    en 3 dimensiones que produce al 
    proyectarse mediante las matrices de 
    proyección dichos puntos.
    """
    PtsTri = []
    for Pt1, Pt2 in zip(Pts1,
                        Pts2):
        A = [Pt1[1]*MP1[2,:] - MP1[1,:],
             MP1[0,:] - Pt1[0]*MP1[2,:],
             Pt2[1]*MP2[2,:] - MP2[1,:],
             MP2[0,:] - Pt2[0]*MP2[2,:]]
        A = np.array(A).reshape((4,4))
        MCorr = A.transpose() @ A
        U, S, VT = linalg.svd(MCorr,
                              full_matrices = False)
        
        aux = VT[3,0:3]/VT[3,3]
        PtsTri.append(aux)
        
    return np.array(PtsTri)

def ProcesarMediaPipe(videoDer, videoIzq):
    camDer, NframesDer, fpsDer = AbrirVideo(videoDer)
    camIzq, NframesIzq, fpsIzq = AbrirVideo(videoIzq)
    
    if NframesDer != NframesIzq:
        print("Error: videos de distinto tamaño.")
        print("Nº frames video derecho: ",
              NframesDer,
              "\n","Nº frames video izquierdo:" ,
              NframesIzq)
        return
    
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hand1 = mp_hands.Hands(min_detection_confidence = 0.6,
                           min_tracking_confidence = 0.5,
                           max_num_hands = 1)
    hand2 = mp_hands.Hands(min_detection_confidence = 0.6,
                           min_tracking_confidence = 0.5,
                           max_num_hands = 1)
    
    coordDer = np.empty((NframesDer,
                         21*3))*np.nan
    coordIzq = np.empty((NframesIzq,
                         21*3))*np.nan
    
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
        
        res = imDer.shape[:2]
        
    cv.destroyWindow(videoDer)
    cv.destroyWindow(videoIzq)
    
    camDer.release()
    camIzq.release()
        
    return coordDer, coordIzq, res

def PredecirFrame(cam, modelo):
    ret, frame = cam.read()
    frame = cv.flip(frame,1)
    imagen = cv.cvtColor(frame,
                         cv.COLOR_BGR2RGB)
    imagen.flags.writeable = False
    resultados = modelo.process(imagen)
    imagen.flags.writeable = True
    imagen = cv.cvtColor(imagen,
                         cv.COLOR_BGR2RGB)
    
    return resultados, imagen
    
def AbrirVideo(ruta):
    cam = cv.VideoCapture(ruta)
    Nframes = int(cam.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cam.get(cv.CAP_PROP_FPS)
    
    return cam, Nframes, fps
    
def ExtraerCoordenadas(resultados):
    coo = np.zeros((21*3))
    for coord in np.asarray(range(21)):
        array = resultados.multi_hand_landmarks[0]
        equis = array.landmark[coord].x
        griega = array.landmark[coord].y
        zeta = array.landmark[coord].z*(-1)
        coo[3*coord:3*coord+3] = np.array([equis,
                                           griega,
                                           zeta])
    
    return coo
    
def TratarNA(array):
    dataframe = pd.DataFrame(array)
    array = dataframe.interpolate().bfill().to_numpy()

    return array

def AbrirCSV(ruta):
    with open(ruta, 'r') as f:
        reader = csv.reader(f)
        CSV = np.array(list(reader),
                       dtype=float)
        return CSV
    
def DerivarArray(array, t):
    a = array
    nodos = range(array.shape[1])
    ejes = range(array.shape[3])
    for eje in ejes:
        for fin in nodos:
            for ini in nodos:
                aux = array[:,
                            fin,
                            ini,
                            eje]
                a[:, fin, ini, eje] = np.gradient(aux,
                                                  t)
    return a

def ModuloArray(array):
    nodos = array.shape[1]
    frames = array.shape[0]
    Mod = np.empty((frames,
                    nodos,
                    nodos))
    for frame in range(frames):
        for fin in range(nodos):
            for ini in range(fin,nodos):
                vec2 = array[frame, fin, ini,:]**2
                ModVec2 = np.sum(vec2)
                Mod[frame, fin, ini] = np.sqrt(ModVec2)
                Mod[frame, ini, fin] = Mod[frame, fin, ini]
    return Mod

def ConversionSegundos(array, colTiempo):
    milis = array[:, colTiempo]
    inicio = np.min(milis)
    tiempo = (milis - inicio)/1000
    
    return tiempo
    
def EncontrarMovimientos(vector, ventana, umbral):
    serie = pd.Series(vector)
    stds = serie.rolling(ventana, center = True).std()
    cond =(stds > umbral).to_numpy()
    pos = np.int8(cond[1:])
    pre = np.int8(cond[:-1])
    limites = pos - pre
    inis = np.where(limites == 1)[0]
    fins = np.where(limites == -1)[0]
    crop = ventana//2
    intervalos = np.array((inis, fins)).transpose()
    intervalos[:,0] = intervalos[:,0] + crop
    intervalos[:,1] = intervalos[:,1] - crop
    
    return intervalos, stds, cond
    
def EncontrarPulsaciones(vector, tiempo, inter, dist, umbral):
    ini = inter[0]
    fin = inter[1]
    condi = tiempo > tiempo[ini]
    condf = tiempo < tiempo[fin]
    indice = np.where(condi*condf)[0]
    fragy = vector[indice]
    fragx = tiempo[indice]
    picos, _ = find_peaks(fragy,
                          height = umbral,
                          distance = dist) 
    
    return fragx, fragy, picos
    
def IntervalosSenal(reloj):
    diff = np.diff(reloj)
    inter = np.where(diff == 1)[0]
    intervalos = np.array((inter[:-1], inter[1:])).transpose()
    
    return intervalos
    
    
def Sincronizar(tR, yR, T, I):
    tAsync = np.zeros(len(tR))
    
    N = len(I[:,0])
    
    for i in range(I.shape[0]):
        ini = I[i,0] - I[0,0]
        fin = I[i,1] - I[0,0]
        largo = fin - ini
        frag = np.linspace(i*T, (i+1)*T, largo + 1)[:-1]
        tAsync[ini:fin] = frag
    tSync = np.linspace(0, T*N,15*N+1)
    
    ySync = np.interp(tSync, tAsync, yR)
    
    return tSync, ySync
    
    
