import numpy as np
import pandas as pd
import src.data_loader as dl
from src.feature_extraction import feature_extraction


# get the image filenames 
image_filenames, neutral_images = dl.get_file_names('data/Dataset_images', 4)
fin_filenames = image_filenames + neutral_images
print len(fin_filenames), 'files successfully extracted.\n'

#now apply kmeans on these images 
#for testing lets just on a subset 

#get the descriptor dataframe
fe = feature_extraction(fin_filenames[:50])
labels_ = fe.kmeans(20)
print (fe.kmeans_model)

print ('test predicting...')
final_df = fe.bag_of_words()
print (final_df)


