"""
The Master.
Module for training the model.
"""
import numpy as np
import pandas as pd
import time 
import pickle 

import src.data_loader as dl
from src.feature_extraction import feature_extraction

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# TODO:
# - Number of kmeans classes hardcoded 
# - pickle svm model 
# - Clean the code

#global variables 
LABELS = ['Neutral', 'Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']
NUM_CLASSES_FOR_KMEANS = 20
TEST_SIZE = 0.3
FILES_TO_EXTRACT_PER_LABEL = 4

def grid_search_cv(clf, x, y, params, cv = 5):
    """
    :param clf: The classifier over which we want to perform 
    gridsearch.
    :param x: Features 
    :param y: Target
    :param params: Hyperparameters to perform gs on
    :cv: kfold cv parameter
    """
    gs = GridSearchCV(clf, param_grid = params, cv = cv)
    gs.fit(x, y)
    print 
    print 'BEST PARAMS:', gs.best_params_
    print 'BEST SCORE:', gs.best_score_
    print 
    best_estimator = gs.best_estimator_
    return best_estimator

######################
# PREPARING THE DATA #
######################

#get the last 4 images from each file
image_filenames, neutral_images = dl.get_file_names('data/Dataset_images', FILES_TO_EXTRACT_PER_LABEL)
#add the neutral faces 
fin_filenames = image_filenames + neutral_images[:150]
print 
print len(fin_filenames), 'files successfully extracted.\n'

emofiles = dl.get_emo_files('data/Emotion/')
#extract the labels 
labels_ = dl.make_label_series(emofiles)
label_rep = np.ndarray.flatten(np.array(list(map(lambda x : [x, x, x, x], labels_))))
neutral_labels = np.zeros(150)

final_labels = np.append(label_rep, neutral_labels)

# train the kmeans with the whole dataset 
fe = feature_extraction(fin_filenames)
fe.kmeans(NUM_CLASSES_FOR_KMEANS)
print (fe.kmeans_model)
print

# apply the bag of words to get another final 
# dataset for svm
final_df = fe.bag_of_words()

###################
# TRAIN TEST SPLIT #
###################

X_train, X_test, y_train, y_test = train_test_split(
    		final_df, final_labels, test_size=TEST_SIZE, random_state=0)

#####################
# GRID SEARCH + SVM #
#####################

print ('Starting Grid Search...\n')
start = time.time()
#parameters for grid-search
tuned_parameters = [{'kernel': ['rbf'], 
                                    'gamma': [1e-3, 1e-4],
                                    'C': [0.001, 0.01, 1, 10, 100, 1000]},
                                    {'kernel': ['linear'], 
                                    'C': [0.001, 0.01, 1, 10, 100, 1000]}]
clf = SVC()
best_clf = grid_search_cv(clf, X_train, y_train, tuned_parameters)

end = time.time()
m, s = divmod((end-start), 60)
h, m = divmod(m, 60)
print ('Grid Search successful.\nTime taken - %f hours %f minutes %f seconds \n' 
    % (h, m, s))

# Apply the model that best fits #
print 'Fitting the best model...\n'
best_clf.fit(X_train, y_train)
print 'Operation Successful!\n'

#Pickle the model for further use 
output = open('svm_model.pkl', 'wb')
pickle.dump(best_clf, output)
output.close()

#################
# PERFORMANCE #
#################

print('Detailed classification report:\n')
print('The model is trained on the full development set.')
print('The scores are computed on the full evaluation set.\n')
y_true, y_pred = y_test, best_clf.predict(X_test)
print(classification_report(y_true, y_pred, target_names=LABELS))
print

print('Confusion Matrix:\n')
print(confusion_matrix(y_true, y_pred))
print 

print('Accuracy:')
print(accuracy_score(y_true, y_pred))
print 
