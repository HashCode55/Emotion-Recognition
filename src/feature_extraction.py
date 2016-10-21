import cv2
import numpy as np
from sklearn.cluster import KMeans 
import pandas as pd 
import time 

#SURF
#surf = cv2.xfeatures2d.SURF_create(400)
#surf.setHessianThreshold(5000)
#kp, des = surf.detectAndCompute(img,None)
#img = cv2.drawKeypoints(gray,kp,None,(255,0,0),4)
#cv2.imwrite('surf_keypoints.jpg',img)
#print (len(des))

#TODO : work on bag of words and fisher vector approach 

class feature_extraction(object):
	"""
	class containing the functions necessary for feature extraction 
	"""
	def __init__(self, filenames):
		self.filenames = filenames
		self.kmeans_model = None 
		self.num_classes = None

	def export_keypoints(self, extractor = 'sift', hes_thresh = 4000):
		"""
		This function takes the images, extracts the keypoints 
		and then create a complete 2-d array of all sift points for 
		further k-means clustering.
		:param images_filenames: image file names 
		:param extractor: algorithm used to extract the keypoints 
		:param hes_thresh: hessian threshhold, only for surf
		::returns descriptors:: a pandas dataframe containg all the points together 
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
		for file in self.filenames:
			img = cv2.imread(file)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			kp, des = extract_.detectAndCompute(gray, None)
			descriptors.append(des)

		descrip_df = self.make_dataframe(descriptors)	
		return descrip_df


	def kmeans(self, num_classes, extractor = 'sift', hes_thresh = 4000):
		"""
		Apply kmeans on on the descriptors to get a fixed sized vector for the 
		bag of words mode.
		:param images_filenames: image file names 
		:param num_classes: number of classes in K-means 
		:param extractor: algorithm used to extract the keypoints 	
		:param hes_thresh: hessian threshhold, only for surf	
		"""
		#mark the starting time 
		self.num_classes = num_classes
		start = time.time()
		print 'Getting the descriptor data...'
		descriptors = self.export_keypoints()
		print 'Data successfully fetched.\n'
		print ('Applying kmeans on %d keypoints with number of clusters being %d \n'
			% (descriptors.shape[0], num_classes))
		self.kmeans_model = KMeans(n_clusters=num_classes, random_state=0)
		self.kmeans_model.fit(descriptors)
		print 'Operation finished \n'

		end = time.time()
		#mark the ending time 
		print 'Time taken - %f seconds' % (end-start)
		#save this kmeans model for further use 
		return (self.kmeans_model.labels_)


	def bag_of_words(self):
		"""
		For creating a fixed sized vector 
		- take one image 
		- get its sift keypoints 
		- classify each and every keypoint into one class using the 
		  kmeans trained model
		- make a fixed sized vector and add it in the final dataset list  
		"""	
		final_df = []
		for file in self.filenames:
			img = cv2.imread(file)	
			keypoints = self.get_imagekeypoints(img)	
			#predict the keypoints 
			predict_labels = self.kmeans_model.predict(keypoints)
			#make a bow vector 
			bow_vector = self.make_vector(predict_labels, self.num_classes)
			final_df.append(bow_vector)

		return pd.DataFrame(final_df)


	def fisher_vector(self, ):
		"""
		A second approach for creating a fixed sized vector 
		"""		

	@staticmethod	
	def make_dataframe(images):
		"""
		Converts the 3 dimentional array to a pandas data frame
		:param images: the array containing descriptors of images 
		"""
		df = pd.DataFrame()
		for image in images:
			temp_df = pd.DataFrame(image)
			df = pd.concat([df, temp_df])
		return df
		
	@staticmethod 
	def get_imagekeypoints(image):
		"""
		Similar to export_keypoints but just for one image 
		"""	
		sift_ = cv2.xfeatures2d.SIFT_create()
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		_, des = sift_.detectAndCompute(gray, None)
		return des


	@staticmethod
	def make_vector(labels, num_classes):
		"""
		Convert the labels into a fixed size vectors 
		"""		
		bow_vector = np.zeros(num_classes)
		for lab in labels:
			bow_vector[lab] += 1
		return bow_vector	