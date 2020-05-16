import numpy as np
import cv2
from utils import *
from ransac_est_homography import ransac_est_homography

def applyBoxTransform(prevX, prevY, newX, newY, coords, valid):
    prevX = prevX[valid]
    prevY = prevY[valid]
    newX = newX[valid]
    newY = newY[valid]
    H = ransac_est_homography(prevX,prevY,newX,newY,0.001)
    coordsT = coords.T
    boxStk = np.vstack((coordsT,np.ones(4)))
    boxPts = H@boxStk
    boxPts = boxPts/boxPts[2:]
    shiftedBox = boxPts[0:2,:].T
    return shiftedBox