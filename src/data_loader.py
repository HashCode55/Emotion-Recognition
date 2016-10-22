"""
The Slave.
Module for loading in the data.
"""
import os

# TODO: 
# Code clean up

def get_file_names(file_path, num_files):

    """
    Extract the images for which the label exists and 
    leave the one for which it doesn't.
    Finally return the filenames for further processing.
    :param num_files: number of files required from the end
    ::returns filenames:: The list of filenames except neutral emotion
    ::returns numfiles:: The list of neutral file names 
    """
    #'data/Landmarks/'
    filenames = []
    neutral_files = []
    #recursively explores the whole directory
    for root, dirs, files in os.walk(file_path):     
        spl_ = root.split('/')
        if len(spl_) == 4: 
            #if the corresponding emotion does not exist continue 
            try:
                if len(os.listdir('data/Emotion/' + spl_[2] + '/' + spl_[3])) == 0:
                    continue
            except:
                continue
        if len(files) != 0 and len(files) != 1:
            new_files = list(map(lambda x : root + '/' + x, files[:-(num_files+1):-1]))
            neutral_files.append(root + '/' + files[1])
            filenames += new_files
    #remove the DSStore files        
    filenames = [url for url in filenames if 'DS_Store' not in url]
    return (filenames, neutral_files)


def get_emo_files(file_path):
    """
    Extract the filenames for the emotion labels 
    ::returns filenames:: emotion filenames 
    """
    #'data/Emotion/''
    filenames = []
    for root, dirs, files in os.walk(file_path):
        fin = list(map(lambda x : root+"/"+x, files))
        filenames += fin
    #get rid of the DS store files     
    filenames = [url for url in filenames if 'DS_Store' not in url]
    return (filenames)   


def make_label_series(filenames):
    """
    Makes a list of labels, extracting the info from filenames
    :param filenames: list of filenames of the labels 
    """
    label_list = []
    for file in filenames:
        #open the file
        with open(file, 'r') as emoLabel:
            read_lab = emoLabel.read().split('   ')[1][:-1]
            label_list.append(int(float(read_lab)))
    return label_list      
