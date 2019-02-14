#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:31:09 2018

intento de detectar dados con opencv

@author: sebalander
"""
# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans
from glob import glob

class separoDadosDeFondo():
    def __init__(self, sigmaColor=10, sigmaSpace=10, kE=5, kD=6):
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace
        self.kE = kE
        self.kD = kD
        self.img = None
        self.imgShape = None
        self.imgLab = None
        self.denoised = None
        self.features = None

    def denoiseBilateral(self, img):
        '''
        paso un filtro bilateral para sacar ruido en gral en espacio LAB
        se guarda la imagen filtrada en self.denoised y se sacan los features
        self.imgShape = img.shape[:2]
        '''
        self.img = img # guardo imagen
        self.imgShape = img.shape[:2]
        self.imgLab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        self.denoised = cv2.bilateralFilter(self.imgLab,  -1, self.sigmaColor, self.sigmaSpace)
        self.features = np.double(self.denoised[:,:,1:].reshape((-1, 2)))

    def segmentacion(self):
        # semilla de centroides
        book = np.array([np.min(self.features, 0), np.max(self.features, 0)])

        # segmento imagen
        self.centroids, dist = kmeans(self.features, book)
        code, dist = vq(self.features, self.centroids)
        self.imgSeg = np.bool8(code.reshape(self.imgShape))

        # elijo la clase que tenga menos pixeles como la de los dados
        if np.prod(self.imgShape) < 2 * self.imgSeg.sum():
            # es que se eligio el fondo como True. lo invierto
            self.imgSeg = np.uint8(~ self.imgSeg)
        else:
            self.imgSeg = np.uint8(self.imgSeg)

    def morphological(self):
        '''
        hago la limpieza de ruido con operaciones morfologicas
        open+close = erode+dilate+dilate+erode
        y saco los contornos de los blobs resultantes
        '''
        kerEr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.kE, self.kE))
        kerDi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.kD, self.kD))

        dadosImg = self.imgSeg
        dadosImg = cv2.morphologyEx(dadosImg, cv2.MORPH_ERODE, kerEr)
        dadosImg = cv2.morphologyEx(dadosImg, cv2.MORPH_DILATE, kerDi)
        dadosImg = cv2.morphologyEx(dadosImg, cv2.MORPH_DILATE, kerDi)
        self.dadosImg = cv2.morphologyEx(dadosImg, cv2.MORPH_ERODE, kerEr)


        self.contours, hierarchy = cv2.findContours(self.dadosImg, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_LIST)

    def getDiceContours(self, img):
        self.denoiseBilateral(img)
        self.segmentacion()
        self.morphological()
        # paso extra de filtrado, supongo convexidad
        self.contours = [cv2.convexHull(con) for con in self.contours]
        # sepued e


    def plotContours(self):
        '''
        grafico los contornos sobre la imagen original
        '''
        plt.figure()
        plt.imshow(self.img[:,:,::-1])
        for cont in self.contours:
            contX, contY = np.array(cont).reshape((-1,2)).T
            plt.plot(contX, contY)

    def dadosIndividuales(self):
        '''
        saco imagenes mas chicas de los dados individuales estan en la
        resolucion y orientacion original, se adjunta la mascara del objeto
        '''

# %%
imgFilePath = "dataset/"
imgFiles = glob(imgFilePath + "*.jpg")
separador = separoDadosDeFondo(sigmaColor=10, sigmaSpace=10, kE=5, kD=6)


for fil in imgFiles:
    img = cv2.imread(fil)
    separador.getDiceContours(img)
    separador.plotContours()

