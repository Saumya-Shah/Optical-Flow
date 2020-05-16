'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
import cv2
import numpy as np

def corner_detector(cimg):
    # [xmin,ymin,xmax,ymax]
    print(cimg.shape)
    cimg = cv2.cornerHarris(cimg,2,3,0.04)
    cimg[cimg<0.01*cimg.max()]=0
    return cimg
