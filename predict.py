import src.process_images as pi
from src.feature_extraction import feature_extraction
import pickle 
from sklearn.cluster import KMeans 

NUM_CLASSES = 20

EMO_DICT = {
    0:'Neutral',
    1:'Anger',
    2:'Contempt',
    3:'Disgust',
    4:'Fear',
    5:'Happy',
    6:'Sad',
    7:'Surprise'
}

def predict(image):
	"""
	Image -> Processed Image -> Kmeans for features -> SVM Predict
	:param image: numpy matrix 
	::returns label:: The predicted label 
	"""

	#Load the kmeans model and svm model
	pkl_file = open('kmeans_model.pkl', 'rb')
	kmeans_model = pickle.load(pkl_file)
	pkl_file.close()

	pkl_file = open('svm_model.pkl', 'rb')
	svm_model = pickle.load(pkl_file)
	pkl_file.close()

	#process the image
	processed_image = pi.process_image(image)
	#get the sift keypoints
	labels = []
	for image in processed_image:
		keypoints = feature_extraction.get_imagekeypoints(image)
		#predict the labels 
		predict_labels = kmeans_model.predict(keypoints)
		#make a BoW vector
		bow_vector = feature_extraction.make_vector(predict_labels, NUM_CLASSES)

		#predict the label using SVM
		lab = svm_model.predict(bow_vector.reshape(1, -1))
		#add the label to the global list
		labels.append(EMO_DICT[int(lab)])

	return labels