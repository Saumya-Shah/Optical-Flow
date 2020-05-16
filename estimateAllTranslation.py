from scipy import signal
import numpy as np
from estimateFeatureTranslation import *
import cv2
from utils import *

def estimateAllTranslation(startXs,startYs,valid,img1,img2,distanceThreshold=4):
    img1G = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2G = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    Imag1,Ix1,Iy1,Iori1 = findDerivatives(img1G)
    newXs = np.zeros(startXs.shape)
    newYs = np.zeros(startYs.shape)
    for i in range(startXs.shape[0]):
        if valid[i]:
            newXs[i],newYs[i] = estimateFeatureTranslation(startXs[i],startYs[i],Ix1,Iy1,img1,img2)
            if newXs[i]==-1 or newYs[i]==-1:
                valid[i] = False
            else:
                xdelta = newXs[i]-startXs[i]
                ydelta = newYs[i]-startYs[i]
                res = np.sqrt(np.square(ydelta)+np.square(xdelta))
                valid[i] = res < distanceThreshold
    return newXs,newYs,valid
