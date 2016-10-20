import cv2
import numpy as np
import math
import subprocess

import sklearn
import glob
from PIL import Image
#for saving the model
from sklearn.externals import joblib

#Some global variables 
label_dict = {'AN':1, 
                    'DI':2, 
                    'FE':3, 
                    'HA':4,
                    'SA':5,
                    'SU':6,
                    'NE':7}


#start vidro capture 
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, img = cap.read()
    cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
    crop_img = img[100:300, 100:300]
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    value = (35, 35)
    #blurred = cv2.GaussianBlur(grey, value, 0)
    #_, thresh1 = cv2.threshold(blurred, 127, 255,
    #to find contours and for better accuracy we need to convert the image in binary and apply gaussian blur!
    str = "Basic structure for recognizing the emotions"
    cv2.putText(img, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    #start vidro capture 
    cv2.imshow('Here', img)
    #cv2.imshow('end', crop_img)
    k = cv2.waitKey(10)
    if k == 27:
        break
