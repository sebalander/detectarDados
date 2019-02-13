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
import scipy as sc
import scipy.cluster as cl


imgFile = "dataset/img15.jpg"
img = cv2.imread(imgFile)

# %%
imgLab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("sarasa0", np.concatenate(cv2.pyrDown(imgLab).transpose((2,0,1)),1))

cv2.imshow("sarasa0", np.concatenate(cv2.pyrDown(imgLab).transpose((2,0,1)),1))

# %%
from scipy.cluster.vq import vq, kmeans, whiten

useRGB = False
if useRGB:
    features = np.double(img.reshape((-1,3)))
else:
    features = np.double(imgLab[:,:,1:].reshape((-1, 2)))

book = np.array([np.min(features, 0), np.max(features, 0)])


centroids, dist = kmeans(features, book)
code, dist = vq(features, centroids)

imgSeg = np.uint8(code.reshape(img.shape[:2]))

plt.imshow(imgSeg, cmap='Set1')
#
## %% elijo como fondo la clase que toque mas los bordes
#
#bordes = np.concatenate([imgSeg[0],imgSeg[-1], imgSeg[:, 0],imgSeg[:, -1]])
#clasBKG = np.int(np.round(np.mean(bordes)))

# %%
#dadosImg = np.uint8(imgSeg != clasBKG)

ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
dadosImg = cv2.morphologyEx(imgSeg, cv2.MORPH_ERODE, ker)

ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
dadosImg = cv2.morphologyEx(dadosImg, cv2.MORPH_CLOSE, ker)

ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
dadosImg = cv2.morphologyEx(dadosImg, cv2.MORPH_DILATE, ker)

plt.figure()
plt.imshow(dadosImg, cmap='gray')

contours, hierarchy = cv2.findContours(dadosImg, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_LIST)

plt.figure()
plt.imshow(img, cmap='gray')
for cont in contours:
    contX, contY = np.array(cont).reshape((-1,2)).T
    plt.plot(contX, contY)

triangles = [cv2.minEnclosingTriangle(con) for con in contours]

plt.figure()
plt.imshow(img  * np.transpose([dadosImg], (1,2,0)))
for tr in triangles:
    print(tr[1])
    plt.plot(tr[1][:,0, 0], tr[1][:,0, 1])

retval, labels, stats, centroids = cv2.connectedComponentsWithStats(imgSeg)

diceArea = np.mean(np.sort(stats[:,4])[1:11])
