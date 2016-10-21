import numpy as np
import pandas as pd
import src.data_loader as dl
import src.feature_extraction as fe


# get the image filenames 
image_filenames, neutral_images = dl.get_file_names('data/Dataset_images', 4)
fin_filenames = image_filenames + neutral_images
print len(fin_filenames), 'files successfully extracted.\n'

#now apply kmeans on these images 
#for testing lets just on a subset 

#get the descriptor dataframe
labels_ = fe.kmeans(fin_filenames[:100], 20)
print (len(labels_))




