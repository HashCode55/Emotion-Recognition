import cv2 
import numpy as np 
from PIL import Image 
import glob 
import matplotlib.pyplot as plt
#Another module will be imprted written by puneet 
FILEPATH = 'data/Landmarks/S010/001/S010_001_00000001_landmarks.txt'

###################
#  GET LANDMARKS  #   
###################
def get_landmarks(filepath):
	land_imp = open(filepath, 'r').read().split('\n')
	land_imp_sp = list(map(lambda x : x.split('   '), land_imp))
	land_imp_fin = [(int(float(k[1])), int(float(k[2]))) for k in land_imp_sp if len(k) == 3]
	return land_imp_fin


#read the image 
img = cv2.imread('data/cohn-kanade-images/S010/001/S010_001_00000001.png')
#process the image 
#SKIP 
landmarks = get_landmarks(FILEPATH)

#display the landmarks 

for landmark in landmarks:
	cv2.circle(img, landmark, 1, (0, 0, 255), -1)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

