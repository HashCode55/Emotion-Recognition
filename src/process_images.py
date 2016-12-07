"""
The Slave.
Module for processing the images.
"""
import os
import cv2
import numpy as np
import copy


def get_file_locations():	
	"""
	::returns file_locs:: 
	"""
	file_locs = []
	for root, dirs, files in os.walk('/Users/mehul/Desktop/ED thorugh FD/data/Dataset_images'):
		fin = list(map(lambda x : root+"/"+x, files))
		file_locs += fin

	file_locs = [file for file in file_locs if 'DS_Store' not in file]

	return file_locs	


def process_dataset():
	"""
	processes the whole dataset
	"""
	image_locs = get_file_locations()
	image_locs = image_locs[:5]
	for image_loc in image_locs:
		image_loc = image_loc.replace("\\", "/") 	
		img = cv2.imread(image_loc, 0)
		faces = process_image(img)
		for face in faces:	
			cv2.imwrite(image_loc, face)


def process_image(img = list()):
	"""
	Extracts faces from the image using haar cascade, resizes and applies filters. 
	:param img: image matrix. Must be grayscale
	::returns faces:: list contatining the cropped face images
	"""
	face_cascade = cv2.CascadeClassifier('/Users/mehul/opencv-3.0.0/build/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')	

	faces_location = face_cascade.detectMultiScale(img, 1.3, 5)
	faces = []
	
	for (x,y,w,h) in faces_location:
		img = img[y:(y+h), x:(x+w)]
		try:
			img = cv2.resize(img, (256, 256))
		except:
			exit(1)
		img = cv2.bilateralFilter(img,15,10,10)
		img = cv2.fastNlMeansDenoising(img,None,4,7,21)
		faces.append(img)

	return faces
	
