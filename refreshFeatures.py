from anms_mod import *
from corner_detector import *
from matplotlib import pyplot as plt
def refreshFeatures(gray,box,x_oldd, y_oldd, valid_oldd,pts):
    x_oldd = x_oldd - box[0]
    y_oldd = y_oldd - box[1]
    res = corner_detector(gray)
    x, y, valid = anms_mod(x_oldd,y_oldd,valid_oldd,res,pts)
    return x+box[0],y+box[1],valid
