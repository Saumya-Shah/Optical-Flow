from scipy import signal
import numpy as np
import pdb
import cv2

def interp2(v, xq, yq):
    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()

    h = v.shape[0]
    w = v.shape[1]
    if xq.shape != yq.shape:
        raise 'query coordinates Xq Yq should have same shape'

    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)

    x_floor[x_floor < 0] = 0
    y_floor[y_floor < 0] = 0
    x_ceil[x_ceil < 0] = 0
    y_ceil[y_ceil < 0] = 0

    x_floor[x_floor >= w-1] = w-1
    y_floor[y_floor >= h-1] = h-1
    x_ceil[x_ceil >= w-1] = w-1
    y_ceil[y_ceil >= h-1] = h-1

    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4
    return interp_val

def GaussianPDF_1D(mu, sigma, length):
    half_len = length / 2
    if np.remainder(length, 2) == 0:
        ax = np.arange(-half_len, half_len, 1)
    else:
        ax = np.arange(-half_len, half_len + 1, 1)

        ax = ax.reshape([-1, ax.size])
        denominator = sigma * np.sqrt(2 * np.pi)
        nominator = np.exp( -np.square(ax - mu) / (2 * sigma * sigma) )
    return nominator / denominator

def GaussianPDF_2D(mu, sigma, row, col):
    g_row = GaussianPDF_1D(mu, sigma, row)
    g_col = GaussianPDF_1D(mu, sigma, col).transpose()
    return signal.convolve2d(g_row, g_col, 'full')

def findDerivatives(I_gray):
    kernelsize = 5
    Gauss2D = GaussianPDF_2D(0,0.5,kernelsize,kernelsize)   #standard deviation = 1 for canny_dataset (bright images), 0.1 for Extra images(low light images)
    dx,dy = np.gradient(Gauss2D,axis=(1,0))
    Ix = signal.convolve2d(I_gray,dx,'same')
    Iy = signal.convolve2d(I_gray,dy,'same')
    Imag = np.sqrt(Ix*Ix + Iy*Iy)
    Iori = np.arctan(Iy/Ix)
    return Imag, Ix, Iy,Iori

def getMinBox(coords):
    height = min(coords[3,1]-coords[0,1],coords[2,1]-coords[1,1])
    width = min(coords[1,0]-coords[0,0],coords[2,0]-coords[3,1])
    minDeltaIdx = 0
    corner = coords[minDeltaIdx].copy()
    if minDeltaIdx == 0:
        height = height
        width = width
    elif minDeltaIdx == 1:
        corner[0] -= width
    elif minDeltaIdx == 2:
        corner[1] -= height
        corner[0] -= width
    elif minDeltaIdx == 3:
        corner[1] -= height
    return int(height), int(width), corner.astype('int')

def getMinPointsBox(X,Y):
    minx = int(np.min(X[X>0]))
    miny = int(np.min(Y[Y>0]))
    maxx = int(np.max(X[X>0]))
    maxy = int(np.max(Y[Y>0]))
    h=maxy-miny
    w=maxx-minx
    corner=np.array([minx,miny])
    return h,w,corner