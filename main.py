import data_loader
import kmeans

#get the image filenames 
image_filenames, neutral_images = data_loader.get_file_names('data/Dataset_images', 4)
fin_filenames = image_filenames + neutral_images
print (len(fin_filenames))

#now apply kmeans on these images 
#for testing lets just on a subset 

labels_ = kmeans.kmeans(fin_filenames[:500], 20)
print (labels_)




