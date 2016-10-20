import cv2
import numpy as np
from sklearn.cluster import KMeans 
import pandas as pd 

#SURF
#surf = cv2.xfeatures2d.SURF_create(400)
#surf.setHessianThreshold(5000)
#kp, des = surf.detectAndCompute(img,None)
#img = cv2.drawKeypoints(gray,kp,None,(255,0,0),4)
#cv2.imwrite('surf_keypoints.jpg',img)
#print (len(des))

def export_siftkeypoints(images_filenames, extractor = 'sift', hes_thresh = 4000):
	"""
	This function takes the images, extracts the keypoints 
	and then create a complete 2-d array of all sift points for 
	further k-means clustering.
	:param images_filenames: image file names 
	:param extractor: algorithm used to extract the keypoints 
	:param hes_thresh: hessian threshhold, only for surf
	"""
	descriptors = []
	extract_ = None
	if extractor == 'sift':
		extract_ = cv2.xfeatures2d.SIFT_create()
	elif extractor == 'surf':
		extract_ = cv2.xfeatures2d.SURF_create()
		extract_.setHessianThreshold(5000)
	else:
		raise ValueError('Descriptor must be sift or surf')	

	#Make the array of keypoints 
	for file in images_filenames:
		img = cv2.imread(file)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		kp, des = extract_.detectAndCompute(gray, None)
		descriptors.append(des)

	return descriptors

def kmeans(images_filenames, num_classes, extractor = 'sift', hes_thresh = 4000):
	"""
	Apply kmeans on on the descriptors to get a fixed sized vector for the 
	bag of words mode.
	:param images_filenames: image file names 
	:param num_classes: number of classes in K-means 
	:param extractor: algorithm used to extract the keypoints 	
	:param hes_thresh: hessian threshhold, only for surf	
	"""
	descriptors = export_siftkeypoints(images_filenames, )
	print (descriptors)
	kmeans = KMeans(n_clusters=num_classes, random_state=0)
	kmeans.fit(descriptors)
	#save this kmeans model for further use 
	return (kmeans.labels_)

