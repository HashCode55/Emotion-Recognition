import numpy as np
import pandas as pd
import src.data_loader as dl
from src.feature_extraction import feature_extraction
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV 
import time 

# TODO:
# - Number of kmeans classes hardcoded 
# - pickle svm model 
# - Clean the code

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
    print ("BEST", gs.best_params_, gs.best_score_, gs.grid_scores_)
    best_estimator = gs.best_estimator_
    return best_estimator

######################
# PREPARING THE DATA #
######################

image_filenames, neutral_images = dl.get_file_names('data/Dataset_images', 4)
fin_filenames = image_filenames + neutral_images[:150]
print len(fin_filenames), 'files successfully extracted.\n'

emofiles = dl.get_emo_files('data/Emotion/')
labels_ = dl.make_label_series(emofiles)
label_new = np.ndarray.flatten(np.array(list(map(lambda x : [x, x, x, x], labels_))))
neutral_labels = np.zeros(150)
# Have these labels baby
fin_labels = np.append(label_new, neutral_labels)
#now apply kmeans on these images 
#for testing lets just on a subset 

# train the kmeans with the whole dataset 
fe = feature_extraction(fin_filenames)
labels_ = fe.kmeans(20)
print (fe.kmeans_model)

# apply the bag of words to get another final 
# dataset for svm
final_df = fe.bag_of_words()

###################
# TRAIN TEST SPLIT #
###################

X_train, X_test, y_train, y_test = train_test_split(
    		final_df, fin_labels, test_size=0.3, random_state=0)

#####################
# GRID SEARCH + SVM #
#####################

print ('Starting Grid Search...\n')
start = time.time()

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [0.001, 0.01, 1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.01, 1, 10, 100, 1000]}]
clf = SVC()
best_clf = grid_search_cv(clf, X_train, y_train, tuned_parameters)

end = time.time()
print 'Grid Search successful. Time taken - %f seconds \n' % (end - start)
# Apply the model that best fits #

print 'Fitting the final model...\n'
best_clf.fit(X_train, y_train)
print 'Operation Successful!'

#################
# PERFORMANCE #
#################

print("Detailed classification report:")
print
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print
y_true, y_pred = y_test, best_clf.predict(X_test)
print(classification_report(y_true, y_pred))
print
