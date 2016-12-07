import numpy as np 
import cv2 
import sys

# read the test image 
TEST_IMAGE  = cv2.imread('/Users/mehul/Desktop/ED thorugh FD/tests/testImage.jpg', 0)
TEST_IMAGE_LIST = [TEST_IMAGE]
DUMMY_PATH = ['/Users/mehul/Desktop/ED thorugh FD/tests/DummyData/testImage.jpg']


# Testing process_images module  
sys.path.insert(0, '/Users/mehul/Desktop/ED thorugh FD/src')

import process_images as pi
from feature_extraction import feature_extraction 

def test_get_file_location():
	# Check that it doesn't return a null list 
	assert  pi.get_file_locations() != None 

def test_process_image():
	assert pi.process_image(TEST_IMAGE) != None 

def test_process_dataset():
	try:
		pi.process_dataset()
	except:
		raise 


# Testing the feature_extraction module 
fe = feature_extraction(DUMMY_PATH)

def test_export_keypoints():
	assert len(fe.export_keypoints()) != 0

def test_kmeans():
	assert len(fe.kmeans(10)) != 0

def test_make_labels():
	assert len(feature_extraction.make_vector([0, 1, 2, 3, 4], 5)) != 0

def test_get_sift_keypoints():
	assert len(feature_extraction.get_imagekeypoints(TEST_IMAGE)) != 0


# Testing the predict module 
sys.path.insert(0, '/Users/mehul/Desktop/ED thorugh FD/')				

import predict 

def test_predict():	
	get_label = predict.predict(TEST_IMAGE)
	assert get_label != None